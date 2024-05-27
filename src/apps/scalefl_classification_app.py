import flwr as fl
import torch

import copy
from typing import Dict, Callable, Optional, Tuple, List
from flwr.server.server import Server

from src.apps import ReeFLClassificationApp
from src.models.model_utils import set_partial_weights
from src.apps.clients import test
from src.utils import get_func_from_config

import logging
logger = logging.getLogger(__name__)

class ScaleFLClassificationApp(ReeFLClassificationApp):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_eval_fn(self) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
        """Return an evaluation function for centralized evaluation."""
        def evaluate(weights: fl.common.Weights, 
                    partition: str, 
                    exit_i: int, 
                    blks_to_exit: List[float]) -> Optional[Tuple[float, float]]:
            # determine device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            width_scale = self.width_scaling[exit_i]
            
            # instantiate model
            net_config = self.ckp.config.models.net
            arch_fn = get_func_from_config(net_config)
            net_args = copy.deepcopy(net_config.args)
            blk_to_exit = blks_to_exit[exit_i]
            net_args.depth = blk_to_exit + 1
            net_args.blks_to_exit = blks_to_exit[:exit_i + 1]
            net_args.width_scale = width_scale
            
            model = arch_fn(device=device, **net_args)

            assert len(model.trainable_state_dict_keys) == len(weights)

            # get subset of model
            set_partial_weights(model, model.trainable_state_dict_keys, weights)

            # instantiate dataloader
            data_config = self.ckp.config.data
            data_class = get_func_from_config(data_config)
            dataset = data_class(self.ckp, **data_config.args)

            model.to(device)

            metrics = {}
            
            testloader = dataset.get_dataloader(
                                        data_pool='server',
                                        partition=partition,
                                        batch_size=self.ckp.config.app.eval_fn.batch_size,
                                        augment=False,
                                        num_workers=0)

            # set last exit only for evaluation
            model.last_exit_only = True
             
            loss, accuracy, _ = test(model, testloader, device=device) 

            metrics[f"centralized_{partition}_exit{exit_i}_loss"] = loss
            metrics[f"centralized_{partition}_exit{exit_i}_acc"] = accuracy * 100
                    
            del model
        
            return loss, metrics

        return evaluate


