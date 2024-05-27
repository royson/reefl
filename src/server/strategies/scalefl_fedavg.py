from src.server.strategies import EarlyExitFedAvg
import torch
import numpy as np
import copy

from src.server.strategies.utils import aggregate_scalefl
from typing import Dict, Optional, Tuple, List, Any
from flwr.server.client_proxy import ClientProxy
from src.utils import get_func_from_config
from collections import defaultdict
from src.models.model_utils import prune
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

import numpy.typing as npt
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]

class ScaleFLFedAvg(EarlyExitFedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.aggregation == 'fedavg', 'only fedavg is supported'

        self.width_scaling = self.ckp.config.app.args.width_scaling
        self.no_of_exits = self.ckp.config.models.net.args.no_of_exits

        self.is_weight = {}
        arch_fn = get_func_from_config(self.net_config)
        global_net = arch_fn(device='cpu', **self.net_config.args)
        self.blks_to_exit = global_net.blks_to_exit
        for k in global_net.state_dict().keys():
            self.is_weight[k] = 'num' not in k

        # get param_ids for each width
        self.param_idxs = {}
        assert len(self.width_scaling) == self.net_config.args.no_of_exits
        for exit_i, width_scale in enumerate(self.width_scaling):
            net_args = copy.deepcopy(self.net_config.args)
            blk_to_exit = self.blks_to_exit[exit_i]
            net_args.depth = blk_to_exit + 1
            net_args.blks_to_exit = self.blks_to_exit[:exit_i+1]
            net_args.width_scale = width_scale
            local_net = arch_fn(device='cpu', **net_args)

            param_idx = {}
            local_state_dict = local_net.state_dict()

            for k in local_state_dict.keys():
                param_idx[k] = [
                    torch.arange(size) for size in local_state_dict[k].shape
                ]  
            self.param_idxs[exit_i] = param_idx

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

        aggregated_weights = aggregate_scalefl(global_sd, clients_local_sd_keys, results, self.is_weight)

        # log training loss
        train_summary = defaultdict(list)
        for _, fit_res in results:
            for m, v in fit_res.metrics.items():
                train_summary[m].append(v)
        for k, v in train_summary.items():
            self.ckp.log({f'mean_{k}': np.mean(v)}, step=rnd, commit=False)
        
        return weights_to_parameters(aggregated_weights), {}
    
    def get_personalized_exit_weights(
        self, exit_i: int, parameters: Parameters,
    ) -> List[NDArrays]:
        local_weights = []
        global_sd = dict(zip(self.global_sd_keys, parameters_to_weights(parameters)))
        global_sd = prune(global_sd, self.param_idxs[exit_i]) # prune depth and width

        for sd_key in self.exit_local_sd_keys[exit_i]: # get keys by depth
            assert sd_key in global_sd # sanity check
            local_weights.append(global_sd[sd_key])

        return local_weights

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
            local_weights = self.get_personalized_exit_weights(exit_i, parameters)

            _, _metrics = self.eval_fn(local_weights, partition, exit_i, self.blks_to_exit)
            mean_loss += _metrics[f'centralized_{partition}_exit{exit_i}_loss']
            mean_acc += _metrics[f'centralized_{partition}_exit{exit_i}_acc']
            metrics = {**metrics, **_metrics}
        
        mean_loss /= self.no_of_exits
        mean_acc /= self.no_of_exits

        metrics[f"centralized_{partition}_exit_all_loss"] = mean_loss
        metrics[f"centralized_{partition}_exit_all_acc"] = mean_acc

        return mean_loss, metrics

