import numpy as np
import copy
from flwr.server.strategy import FedAvg as FlowerFedAvg
from flwr.server.client_manager import ClientManager
from src.utils import get_func_from_config
from pprint import pformat
from collections import defaultdict
from src.server.strategies.utils import aggregate_inplace
from src.apps.app_utils import cosine_decay_with_warmup
from typing import Dict, Optional, Tuple, List, Any
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Weights,
    Scalar,
    FitRes,
    FitIns,
    parameters_to_weights,
    weights_to_parameters,
)

import logging
logger = logging.getLogger(__name__)

import pdb
import numpy.typing as npt
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

class InclusiveFL(FlowerFedAvg):
    '''
        Reimplementation of No One Left Behind: Inclusive Federated Learning Over Heterogeneous Devices (KDD'22)
    '''
    def __init__(self, ckp, client_valuation, *args, aggregation='fedavg', aggregation_args={}, beta=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckp = ckp
        self.config = ckp.config
        self.beta = beta
        self.aggregation = aggregation
        self.aggregation_args = aggregation_args

        # get global sd keys corresponding to global parameters
        self.net_config = self.config.models.net
        arch_fn = get_func_from_config(self.net_config)
        global_net = arch_fn(device='cpu', **self.net_config.args)
        self.blks_to_exit = global_net.blks_to_exit
        self.global_sd_keys = global_net.all_state_dict_keys

        # get each local client sd keys
        no_of_clients = self.config.simulation.num_clients
        no_of_exits = self.ckp.config.models.net.args.no_of_exits
        depth = self.ckp.config.models.net.args.depth
        no_of_blocks_per_exit = depth // no_of_exits
        self.no_of_exits = no_of_exits
        self.max_exit = no_of_exits - 1

        # map exit to shared and local sd_keys
        self.exit_local_sd_keys = {} 
        self.exit_personalized_sd_values = {} # map exit to personalized values (f in paper)
        self.exit_shared_sd_keys = {} # map exit to personalized layers
        self.exit_momentum_sd_block_ids = {} # map exit to momentum distill block ids
        self.exit_momentum_sd_values = {} # map exit to momentum distill weights (m in paper)
        
        for exit_i in range(no_of_exits):
            arch_fn = get_func_from_config(self.net_config)
            net_args = copy.deepcopy(self.net_config.args)
            blk_to_exit = self.blks_to_exit[exit_i]
            net_args.depth = blk_to_exit + 1
            net_args.no_of_exits = exit_i + 1 # equal blks between exits
            local_net = arch_fn(device='cpu', **net_args)
            self.exit_local_sd_keys[exit_i] = local_net.trainable_state_dict_keys

            self.exit_personalized_sd_values[exit_i] = {}
            shared_sd_keys = []
            if exit_i != no_of_exits - 1:
                personalized_block_id = (exit_i + 1) * no_of_blocks_per_exit - 1
                for sd_key in local_net.trainable_state_dict_keys:
                    if f'blocks.{personalized_block_id}' in sd_key or f'accumulator.{exit_i}' in sd_key:
                        self.exit_personalized_sd_values[exit_i][sd_key] = None 
                    # elif sd_key.startswith('blocks.') or sd_key.startswith('patch_embed.ssf'):
                    elif (sd_key.startswith('blocks.')
                        or sd_key.startswith('patch_embed')
                        or sd_key in ['cls_token', 'pos_embed']
                        or sd_key.startswith('blocks_token_only')
                        or sd_key in ['norm.weight', 'norm.bias']):
                        shared_sd_keys.append(sd_key)
            else: # last exit
                personalized_block_id = depth - no_of_blocks_per_exit
                for sd_key in local_net.trainable_state_dict_keys:
                    # if sd_key.startswith('patch_embed.ssf'):
                    if (sd_key.startswith('patch_embed') 
                        or sd_key in ['cls_token', 'pos_embed'] 
                        or sd_key.startswith('blocks_token_only') 
                        or sd_key in ['norm.weight', 'norm.bias']):
                        shared_sd_keys.append(sd_key)
                        continue

                    block_id = int(sd_key.split('.')[1])
                    if block_id >= personalized_block_id or f'accumulator.{exit_i}' in sd_key:
                        self.exit_personalized_sd_values[exit_i][sd_key] = None
                    elif sd_key.startswith('blocks.'):
                        shared_sd_keys.append(sd_key)
                
            self.exit_momentum_sd_values[exit_i] = {}
            momentum_sd_block_ids = []
            if exit_i != 0: # not first exit
                earliest_momentum_block_id = (exit_i + 1) * no_of_blocks_per_exit - no_of_blocks_per_exit
                for sd_key in local_net.trainable_state_dict_keys:
                    # if sd_key.startswith('patch_embed'):
                    if (sd_key.startswith('patch_embed') 
                        or sd_key in ['cls_token', 'pos_embed'] 
                        or sd_key.startswith('blocks_token_only') 
                        or sd_key in ['norm.weight', 'norm.bias']):
                        continue
                    _, block_id, *layer = sd_key.split('.')
                    block_id = int(block_id)

                    if sd_key.startswith('blocks') and block_id >= earliest_momentum_block_id:
                        momentum_sd_block_ids.append(block_id)
                        layer = '.'.join(layer)
                        if layer not in self.exit_momentum_sd_values:
                            self.exit_momentum_sd_values[exit_i][layer] = np.zeros(local_net.state_dict()[sd_key].shape)

            self.exit_shared_sd_keys[exit_i] = shared_sd_keys
            self.exit_momentum_sd_block_ids[exit_i] = set(momentum_sd_block_ids)

            assert not set(self.exit_personalized_sd_values[exit_i].keys()).intersection(set(shared_sd_keys))
            assert set(self.exit_personalized_sd_values[exit_i].keys()) | set(shared_sd_keys) == set(local_net.trainable_state_dict_keys), \
                f'Exit {exit_i}: Difference: {set(local_net.trainable_state_dict_keys) - (set(self.exit_personalized_sd_values[exit_i].keys()) | set(shared_sd_keys))}'
        
        # map client to exit
        self.clients_exit = {}
        for i in range(no_of_clients):
            if self.config.app.args.mode == 'maximum':
                max_exit = no_of_exits - 1
            else:
                max_exit = i % no_of_exits
            self.clients_exit[str(i)] = max_exit
        
        # aggregation
        assert self.aggregation in ['fedavg', 'fedadam']

        if self.aggregation == 'fedadam':
            assert 'beta_1' in self.aggregation_args
            assert 'beta_2' in self.aggregation_args
            assert 'tau' in self.aggregation_args
            assert 'eta' in self.aggregation_args

            # each exit has its own adam buffer
            self.m_t = {}
            self.v_t = {}

    def get_personalized_exit_weights(self, exit_i: int, parameters: Parameters) -> List[NDArrays]:
        local_weights = []
        global_sd = dict(zip(self.global_sd_keys, parameters_to_weights(parameters)))

        for sd_key in self.exit_local_sd_keys[exit_i]: # get all sd_keys required
            if sd_key in self.exit_shared_sd_keys[exit_i]: # if key is shared key, take from global parameters
                local_weights.append(global_sd[sd_key])
            elif sd_key in self.exit_personalized_sd_values[exit_i]: # if key is personalized, take from local else global
                if self.exit_personalized_sd_values[exit_i][sd_key] is None:
                    local_weights.append(global_sd[sd_key]) # only happens during 1st round
                else:
                    local_weights.append(self.exit_personalized_sd_values[exit_i][sd_key])
            else:
                raise NotImplementedError()
        return local_weights

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_instructions = []

        assert len(self.global_sd_keys) == len(parameters_to_weights(parameters))

        # get sub-global model
        _tmp = {}
        for client in clients:
            exit_i = self.clients_exit[client.cid]
            if exit_i not in _tmp:
                local_weights = self.get_personalized_exit_weights(exit_i, parameters)
                _tmp[exit_i] = local_weights

            local_weights = _tmp[exit_i]
            client_instructions.append((client, FitIns(weights_to_parameters(local_weights), config)))

        return client_instructions

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
        current_parameters: Parameters, 
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        global_sd = dict(zip(self.global_sd_keys, parameters_to_weights(current_parameters)))
        global_sd_updates = defaultdict(int)

        # group clients by exit no.
        exit_clients = defaultdict(list)
        for client, fit_res in results:
            exit_clients[self.clients_exit[client.cid]].append((client, fit_res))

        # sort from earliest to latest exit
        exit_clients = dict(sorted(exit_clients.items()))

        if self.aggregation == 'fedadam':
            beta_1 = self.aggregation_args['beta_1']
            beta_2 = self.aggregation_args['beta_2']
            tau = self.aggregation_args['tau']
            eta = self.aggregation_args['eta']

        ## Homomorphic aggregation
        for exit_i, group_results in exit_clients.items():
            # print(f'Exit {exit_i}')
            group_params = aggregate_inplace(group_results)
            group_sd = dict(zip(self.exit_local_sd_keys[exit_i], group_params))
            group_sd_grad = {}

            # compute group aggregated gradients
            local_weights = self.get_personalized_exit_weights(exit_i, current_parameters)
            assert len(local_weights) == len(self.exit_local_sd_keys[exit_i]) == len(group_sd)
            for k, initial_weight in zip(self.exit_local_sd_keys[exit_i], local_weights):
                # updated model = initial model - grad
                group_sd_grad[k] = initial_weight - group_sd[k] 
            
            # add momentum distillation
            if exit_i != self.max_exit:
                # for k in group_sd_grad.keys():
                for k in self.exit_personalized_sd_values[exit_i].keys():
                    if k.startswith('blocks'): # last transformer block
                        # print(f'adding momentum to {k}')
                        assert k in group_sd_grad
                        
                        # retrieve momentum value
                        _, _, *layer = k.split('.')
                        layer = '.'.join(layer)
                        layer_mom = self.exit_momentum_sd_values[exit_i + 1][layer]

                        group_sd_grad[k] = self.beta * layer_mom + (1 - self.beta) * group_sd_grad[k]
            
            if self.aggregation == 'fedavg':
                # applying the grad
                for k, initial_weight in zip(self.exit_local_sd_keys[exit_i], local_weights):
                    group_sd[k] = initial_weight - group_sd_grad[k]
            else: # fedadam
                if exit_i not in self.m_t:
                    self.m_t[exit_i] = {k: np.zeros_like(x) for k, x in zip(self.exit_local_sd_keys[exit_i], local_weights)}
                    self.v_t[exit_i] = {k: np.zeros_like(x) for k, x in zip(self.exit_local_sd_keys[exit_i], local_weights)}
                
                for k in self.exit_local_sd_keys[exit_i]:
                    g = group_sd_grad[k]
                    self.m_t[exit_i][k] = np.multiply(beta_1, self.m_t[exit_i][k]) + (1 - beta_1) * g            
                    self.v_t[exit_i][k] = np.multiply(beta_2, self.v_t[exit_i][k]) + (1 - beta_2) * np.multiply(g, g)
                
                # applying the grad
                for k, initial_weight in zip(self.exit_local_sd_keys[exit_i], local_weights):
                    group_sd[k] = initial_weight - eta * self.m_t[exit_i][k] / (np.sqrt(self.v_t[exit_i][k]) + tau)
                    
            # Heterogeneous aggregation: update shared global weights and store them in global
            for sd_key in self.exit_shared_sd_keys[exit_i]:
                aggregated_update = len(group_results) * group_sd[sd_key]
                # print(f'Heterogeneous Aggregating {sd_key}')
                if global_sd_updates[sd_key] == 0: # no updates yet
                    global_sd[sd_key] = aggregated_update
                else:
                    global_sd[sd_key] += aggregated_update
                global_sd_updates[sd_key] += len(group_results)
            
            # update personalized weights
            for sd_key in self.exit_personalized_sd_values[exit_i].keys():
                self.exit_personalized_sd_values[exit_i][sd_key] = group_sd[sd_key]

            # saving momentum
            if exit_i != 0:
                total_blocks = len(self.exit_momentum_sd_block_ids[exit_i])
                for layer in self.exit_momentum_sd_values[exit_i].keys():
                    self.exit_momentum_sd_values[exit_i][layer] = np.zeros(self.exit_momentum_sd_values[exit_i][layer].shape)
                    for block_id in self.exit_momentum_sd_block_ids[exit_i]:
                        # print(f'[{exit_i}]saving momentum for {layer} from blocks.{block_id}')
                        k = f'blocks.{block_id}.{layer}'
                        self.exit_momentum_sd_values[exit_i][layer] += group_sd_grad[k]
                    self.exit_momentum_sd_values[exit_i][layer] /= total_blocks

        # Heterogenous aggregation: taking average
        for sd_key, no_of_client_updates in global_sd_updates.items():
            assert no_of_client_updates > 0
            global_sd[sd_key] /= no_of_client_updates
            
        # reset momentum distillation for exits that are not sampled
        for exit_i in range(self.no_of_exits):
            if exit_i not in exit_clients:
                for layer in self.exit_momentum_sd_values[exit_i].keys():
                    self.exit_momentum_sd_values[exit_i][layer] = np.zeros(self.exit_momentum_sd_values[exit_i][layer].shape)
        
        # log training loss
        train_summary = defaultdict(list)
        for _, fit_res in results:
            for m, v in fit_res.metrics.items():
                train_summary[m].append(v)
        for k, v in train_summary.items():
            self.ckp.log({f'mean_{k}': np.mean(v)}, step=rnd, commit=False)
        
        return weights_to_parameters(list(global_sd.values())), {}

    def evaluate(
        self, parameters: Parameters, partition: str = 'test',
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None

        mean_loss, mean_acc = 0.0, 0.0
        metrics = {}
        for exit_i in range(self.no_of_exits):
            blk_to_exit = self.blks_to_exit[exit_i]
            local_weights = self.get_personalized_exit_weights(exit_i, parameters)

            _, _metrics = self.eval_fn(local_weights, partition, exit_i, blk_to_exit)
            mean_loss += _metrics[f'centralized_{partition}_exit{exit_i}_loss']
            mean_acc += _metrics[f'centralized_{partition}_exit{exit_i}_acc']
            metrics = {**metrics, **_metrics}
        
        mean_loss /= self.no_of_exits
        mean_acc /= self.no_of_exits

        metrics[f"centralized_{partition}_exit_all_loss"] = mean_loss
        metrics[f"centralized_{partition}_exit_all_acc"] = mean_acc

        return mean_loss, metrics


    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients, all_available=True
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        client_results = {}
        for client, evaluate_res in results:
            client_results[client.cid] = (
                evaluate_res.num_examples,
                evaluate_res.loss,
                evaluate_res.metrics,
            )

        loss_aggregated, accuracy_results = weighted_loss_avg(
            client_results,
            self.test_alpha
        )
        return loss_aggregated, accuracy_results

def weighted_loss_avg(results: Dict[str, Tuple[int, float, Optional[Dict[str, float]]]], personalized_fl_groups: Dict[str, int]) -> Tuple[float, float]:
    """Aggregate evaluation results obtained from multiple clients.
    TODO: rename variables to include other metrics apart from accuracy. 
    """
    accuracy_results = {}
    if personalized_fl_groups is not None and len(personalized_fl_groups) > 1:
        from_id = 0
        for group, to_id in personalized_fl_groups.items():
            group_examples = 0
            group_correct_preds = defaultdict(float)
            group_loss = 0
            for cid in range(from_id, from_id + int(to_id)):
                num_examples, loss, metrics = results[str(cid)]
                group_examples += num_examples
                for k, acc in metrics.items():
                    if 'test_acc' in k or 'accuracy' in k:
                        group_correct_preds[k] += num_examples * acc
                    else:
                        group_correct_preds[k] += acc
                group_loss += num_examples * loss
            from_id += to_id
            for k, v in group_correct_preds.items():
                if 'test_acc' in k or 'accuracy' in k:
                    accuracy_results[f'ps_{k}_alpha{group}({to_id} clients)'] = v / group_examples * 100
                else:
                    accuracy_results[f'mean_{k}_alpha{group}({to_id} clients)'] = v / float(to_id)
    
    # overall accuracy    
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _, _ in results.values()]
    )
    weighted_losses = [num_examples * loss for num_examples, loss, _ in results.values()]
    num_correct_preds = defaultdict(list)
    for num_examples, _, metrics in results.values():
        for k, acc in metrics.items():
            if 'test_acc' in k or 'accuracy' in k:
                num_correct_preds[k].append(num_examples * acc)
            else:
                num_correct_preds[k].append(acc)

    # num_correct_preds = [num_examples * accuracy for num_examples, _, accuracy in results.values()]
    for k, v in num_correct_preds.items():
        if 'test_acc' in k or 'accuracy' in k:
            accuracy_results[f'ps_{k}'] =  sum(v) / num_total_evaluation_examples * 100
        else:
            accuracy_results[f'mean_{k}'] = np.mean(v)


    return sum(weighted_losses) / num_total_evaluation_examples, accuracy_results