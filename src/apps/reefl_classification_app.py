import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import math
import copy
from typing import Dict, Callable, Optional, Tuple 
from flwr.server.server import Server
from flwr.server.history import History
from flwr.common import weights_to_parameters

from collections import OrderedDict, defaultdict
from src.apps import ClassificationApp
from src.apps.app_utils import cosine_decay_with_warmup
from src.models.model_utils import set_weights, set_partial_weights
from src.apps.clients import ree_early_exit_test
from src.utils import get_func_from_config

import logging
logger = logging.getLogger(__name__)

class ReeFLClassificationApp(ClassificationApp):    
    def __init__(self, *args, mode='multi_tier', save_log_file=True, width_scaling=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert mode in ['maximum', 'multi_tier']
        self.mode = mode
        self.save_log_file = save_log_file
        self.no_of_exits = self.ckp.config.models.net.args.no_of_exits
        self.no_of_clients = self.ckp.config.simulation.num_clients
        self.width_scaling = width_scaling
        if self.width_scaling:
            assert len(self.width_scaling) == self.no_of_exits, 'no. of width scales must match no. of exits.'

        if self.mode == 'multi_tier':
            self.client_max_exit_layers = {str(i): i % self.no_of_exits for i in range(self.no_of_clients)}
            if self.width_scaling:
                self.client_width_scales = {str(i): self.width_scaling[i % len(self.width_scaling)] for i in range(self.no_of_clients)}

            
    def get_client_fn(self):
        client = get_func_from_config(self.app_config.client)
        def client_fn(cid: str):  
            max_early_exit_layer = None
            width_scale = 1.0
            if self.mode == 'maximum':
                max_early_exit_layer = self.no_of_exits - 1
            else: # multi_tier
                max_early_exit_layer = self.client_max_exit_layers[cid]
                if self.width_scaling:
                    width_scale = self.client_width_scales[cid]

            return client(
                cid,
                max_early_exit_layer, 
                width_scale,
                self.ckp,
                **self.app_config.client.args
            )

        return client_fn

    def get_fit_config_fn(self):
        """Return a configuration with static batch size and (local) epochs."""
        def fit_config_fn(rnd: int) -> Dict[str, str]:
            fit_config = self.ckp.config.app.on_fit

            if fit_config.cos_lr_decay:
                current_lr = cosine_decay_with_warmup(rnd,
                                learning_rate_base=fit_config.start_lr,
                                total_steps=self.ckp.config.app.run.num_rounds,
                                minimum_learning_rate=fit_config.min_lr,
                                warmup_learning_rate=0,
                                warmup_steps=0,
                                hold_base_rate_steps=0.)
            else: 
                current_lr = fit_config.start_lr

            client_config = {
                "lr": current_lr,
                "current_round": rnd,
                }
            return client_config

        return fit_config_fn

    
    def get_evaluate_config_fn(self):
        """"Client evaluate. Evaluate on client's test set"""
        def evaluate_config_fn(rnd: int) -> Dict[str, str]:
            eval_config = self.ckp.config.app.on_evaluate

            client_config = {
                "lr": eval_config.lr,
                "current_round": rnd,
                "finetune_epochs": eval_config.finetune_epochs }
            return client_config

        return evaluate_config_fn

    def get_eval_fn(self) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
        """Return an evaluation function for centralized evaluation."""
        def evaluate(weights: fl.common.Weights, partition: str) -> Optional[Tuple[float, float]]:
            # determine device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # instantiate model
            net_config = self.ckp.config.models.net
            data_config = self.ckp.config.data
            arch_fn = get_func_from_config(net_config)
            net_args = copy.deepcopy(net_config.args)
            net_args.last_exit_only = False # evaluate all exits during evaluation
            model = arch_fn(device=device, **net_args)

            set_partial_weights(model, model.trainable_state_dict_keys, weights)

            # instantiate dataloader
            data_class = get_func_from_config(data_config)
            dataset = data_class(self.ckp, **data_config.args)

            model.to(device)

            metrics = {}
            max_exit = self.no_of_exits - 1
            
            mean_loss, mean_acc = 0.0, 0.0

            testloader = dataset.get_dataloader(
                                        data_pool='server',
                                        partition=partition,
                                        batch_size=self.ckp.config.app.eval_fn.batch_size,
                                        augment=False,
                                        num_workers=0)
            results = ree_early_exit_test(model, max_exit, testloader, device=device, ensemble=True) 

            for ee in range(self.no_of_exits):
                metrics[f"centralized_{partition}_exit{ee}_loss"] = results[ee]['loss']
                metrics[f"centralized_{partition}_exit{ee}_acc"] = results[ee]['accuracy'] * 100
                    
                mean_loss += results[ee]['loss']
                mean_acc += results[ee]['accuracy']

            mean_loss /= self.no_of_exits
            mean_acc /= self.no_of_exits

            del model

            metrics[f"centralized_{partition}_exit_all_loss"] = mean_loss
            metrics[f"centralized_{partition}_exit_all_acc"] = mean_acc * 100
            metrics[f'centralized_{partition}_exit_all_ensemble_acc'] = results['ensemble_accuracy'] * 100
        
            return mean_loss, metrics

        return evaluate

    def run(self, server: Server):
        """Run federated averaging for a number of rounds."""
        history = History()
        data_config = self.ckp.config.data
        data_class = get_func_from_config(data_config)
        dataset = data_class(self.ckp, **data_config.args)

        def _centralized_evaluate(rnd, partition, log=True):
            server_metrics = None
            # Evaluate model using strategy implementation (runs eval_fn)
            parameters = server.parameters 
            res_cen = server.strategy.evaluate(parameters=parameters, partition=partition)
            if res_cen is not None:
                loss_cen, server_metrics = res_cen
                history.add_loss_centralized(rnd=rnd, loss=loss_cen)
                history.add_metrics_centralized(rnd=rnd, metrics=server_metrics)
                if log:
                    self.ckp.log(server_metrics, step=rnd)
            return server_metrics

        # `Initial`ize parameters
        if self.load or self.start_run > 1:
            server.parameters = self.current_weights
            logger.info('[*] Global Parameters Loaded.')
        else:
            net_config = self.ckp.config.models.net
            arch_fn = get_func_from_config(net_config)  
            net = arch_fn(device='cpu', **net_config.args) 
            # set initial parameters
            server.parameters = weights_to_parameters([val.cpu().numpy() for k, val in net.state_dict().items() if k in net.all_state_dict_keys])

        # Run federated learning for num_rounds
        logger.info("FL starting")
        # start_time = timeit.default_timer()

        app_run_config = self.ckp.config.app.run

        for rnd in range(self.start_run, app_run_config.num_rounds + 1):
            # Train model and replace previous global model
            server_metrics = None
            clients_metrics = None
            res_fit = server.fit_round(rnd=rnd)
            if res_fit:
                parameters_prime, _, (results, _) = res_fit  
                clients_metrics = [res[1].metrics for res in results]

                if parameters_prime:
                    server.parameters = parameters_prime

            if rnd % app_run_config.test_every_n == 0:
                logger.debug(f"[Round {rnd}] Evaluating global model on test set.")
                server_metrics = _centralized_evaluate(rnd, 'test')
                logger.info(f"[Round {rnd}] {server_metrics}")

            # end of round saving
            self.ckp.save(f'results/round_{rnd}.pkl', 
                {'round': rnd,
                'clients_metrics': clients_metrics, 
                'server_metrics': server_metrics})
            self.ckp.save(f'models/latest_weights.pkl',
                server.parameters)
            if app_run_config.save_every_n is not None and (rnd == self.start_run or rnd % app_run_config.save_every_n == 0):
                self.ckp.save(f'models/weights_round_{rnd}.pkl',
                    server.parameters)
            self.ckp.save(f'models/last_round_saved.pkl', rnd)

        # test set evaluation and logging using wandb summary metrics
        logger.info(f"[Round {rnd}] Training done. Final test evaluation")
        server_metrics = _centralized_evaluate(rnd, 'test', log=False)
        logger.info(f"Final Test Result: {server_metrics}")
        for k, v in server_metrics.items():
            self.ckp.log_summary(k, v)
            
            if self.save_log_file:
                alpha = 0 if dataset.pre_partition else '_'.join(list(dataset.test_alpha.keys()))
                self.ckp.save_results_logfile(self.mode, alpha, k, v, ps_type=f'init_{self.ckp.config.name}', \
                                            filepath=self.save_log_file, reset=False)

        self.ckp.save(f'results/round_{rnd}_test.pkl', 
                {'server_metrics': server_metrics})
        self.ckp.save(f'models/weights_{rnd}_final.pkl',
                server.parameters)
