import numpy as np
import copy
from flwr.server.client_manager import ClientManager
from src.utils import get_func_from_config
from pprint import pformat
from collections import defaultdict
from src.server.strategies.utils import aggregate_inplace_early_exit_feddyn
from src.server.strategies import EarlyExitFedAvg

from typing import Dict, Optional, Tuple, List
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.common import (
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

class EarlyExitFedDyn(EarlyExitFedAvg):
    '''
        Modified from: https://github.com/adap/flower/blob/main/baselines/depthfl/depthfl/strategy.py
    '''
    def __init__(self, ckp, client_valuation, *args, alpha=0.1, **kwargs):
        super().__init__(ckp, client_valuation, *args, **kwargs)
        arch_fn = get_func_from_config(self.net_config)
        global_net = arch_fn(device='cpu', **self.net_config.args)

        self.h_variate = {}
        for k in self.global_sd_keys:
            if k in global_net.all_state_dict_keys:
                self.h_variate[k] = np.zeros_like(global_net.state_dict()[k])

        self.alpha = alpha

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
        aggregated_weights = aggregate_inplace_early_exit_feddyn(global_sd, 
                                                                clients_local_sd_keys, 
                                                                results,
                                                                self.h_variate,
                                                                self.no_of_clients,
                                                                self.alpha)

        # log training loss
        train_summary = defaultdict(list)
        for _, fit_res in results:
            for m, v in fit_res.metrics.items():
                train_summary[m].append(v)
        for k, v in train_summary.items():
            self.ckp.log({f'mean_{k}': np.mean(v)}, step=rnd, commit=False)
        
        return weights_to_parameters(aggregated_weights), {}
