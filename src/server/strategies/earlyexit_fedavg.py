import numpy as np
import copy
from flwr.server.strategy import FedAvg as FlowerFedAvg
from flwr.server.client_manager import ClientManager
from src.utils import get_func_from_config
from pprint import pformat
from collections import defaultdict
from src.server.strategies.utils import aggregate_inplace_early_exit, aggregate_inplace_early_exit_fedsparseadam

from typing import Dict, Optional, Tuple, List
from flwr.server.client_proxy import ClientProxy
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

class EarlyExitFedAvg(FlowerFedAvg):
    '''
        Similar to flwr.server.strategy.FedAvg. 
        Includes parameter mapping from local to global model parameters and
        parameter aggregation of different model sizes
    '''
    def __init__(self, ckp, client_valuation, *args, aggregation='fedavg', aggregation_args={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckp = ckp
        self.config = ckp.config

        # get global sd keys corresponding to global parameters
        self.net_config = self.config.models.net
        arch_fn = get_func_from_config(self.net_config)
        global_net = arch_fn(device='cpu', **self.net_config.args)
        self.global_sd_keys = global_net.all_state_dict_keys

        # get each local client sd keys
        self.no_of_clients = self.config.simulation.num_clients
        no_of_exits = self.ckp.config.models.net.args.no_of_exits

        self.exit_local_sd_keys = {}
        for exit_i in range(no_of_exits):
            arch_fn = get_func_from_config(self.net_config)
            net_args = copy.deepcopy(self.net_config.args)
            blk_to_exit = global_net.blks_to_exit[exit_i]
            net_args.depth = blk_to_exit + 1
            net_args.blks_to_exit = global_net.blks_to_exit[:exit_i+1]
            # net_args.no_of_exits = exit_i + 1
            local_net = arch_fn(device='cpu', **net_args)
            self.exit_local_sd_keys[exit_i] = local_net.trainable_state_dict_keys

        self.clients_exit = {}
        for i in range(self.no_of_clients):
            if self.config.app.args.mode == 'maximum':
                max_exit = no_of_exits - 1
            else:
                max_exit = i % no_of_exits
            self.clients_exit[str(i)] = max_exit

        self.aggregation = aggregation
        self.aggregation_args = aggregation_args
        assert self.aggregation in ['fedavg', 'fedadam']

        if self.aggregation == 'fedadam':
            assert 'beta_1' in self.aggregation_args
            assert 'beta_2' in self.aggregation_args
            assert 'tau' in self.aggregation_args
            assert 'eta' in self.aggregation_args

            self.m_t = {}
            self.v_t = {}
            gn_sd = global_net.state_dict()
            for sd_key in self.global_sd_keys:
                self.m_t[sd_key] = np.zeros_like(gn_sd[sd_key])
                self.v_t[sd_key] = np.zeros_like(gn_sd[sd_key])

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
        global_sd = dict(zip(self.global_sd_keys, parameters_to_weights(parameters)))
        for client in clients:
            local_sd_keys = self.exit_local_sd_keys[self.clients_exit[client.cid]]
            local_weights = [global_sd[k] for k in local_sd_keys]

            client_instructions.append((client, FitIns(weights_to_parameters(local_weights), config)))

        # fit_ins = FitIns(parameters, config)
        # Return client/config pairs
        # return [(client, fit_ins) for client in clients]
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
        
        # Convert results for fedavg
        clients_local_sd_keys = {client.cid: self.exit_local_sd_keys[self.clients_exit[client.cid]] for client, _ in results}

        if self.aggregation == 'fedavg':
            aggregated_weights = aggregate_inplace_early_exit(global_sd, clients_local_sd_keys, results)
        elif self.aggregation == 'fedadam':
            beta_1 = self.aggregation_args['beta_1']
            beta_2 = self.aggregation_args['beta_2']
            tau = self.aggregation_args['tau']
            eta = self.aggregation_args['eta']

            aggregated_weights = aggregate_inplace_early_exit_fedsparseadam(global_sd, 
                                    clients_local_sd_keys, 
                                    results, 
                                    self.m_t, 
                                    self.v_t,
                                    beta_1,
                                    beta_2,
                                    tau,
                                    eta)

        # log training loss
        train_summary = defaultdict(list)
        for _, fit_res in results:
            for m, v in fit_res.metrics.items():
                train_summary[m].append(v)
        for k, v in train_summary.items():
            self.ckp.log({f'mean_{k}': np.mean(v)}, step=rnd, commit=False)
        
        return weights_to_parameters(aggregated_weights), {}

    def evaluate(
        self, parameters: Parameters, partition: str = 'test',
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = parameters_to_weights(parameters)
        eval_res = self.eval_fn(weights, partition)
        if eval_res is None:
            return None
        loss, other = eval_res
        if isinstance(other, float):
            metrics = {"accuracy": other}
        else:
            metrics = other
        return loss, metrics


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