import flwr as fl
import os
import torch
import torch.nn.functional as F
import ray
import numpy as np
import copy
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional
from config import AttrDict
from src.utils import get_func_from_config
from src.apps.clients import ClassificationClient, epochs_to_batches
from src.models.model_utils import prune
from src.data import cycle

def compute_kl_loss(kl_criterion, source, target, softmax_temp=1.):
    return kl_criterion(F.log_softmax(source/softmax_temp, dim=1), 
                        F.log_softmax(target/softmax_temp, dim=1)) * (softmax_temp ** 2)

def train(
    net,
    max_exit_layer,
    trainloader,
    valloader,
    optimizer,
    finetune_batch,
    device: str,
    round: int,
    mu: float = 0,
    kl_loss: bool =  False,
    kl_weight = None,
    kl_softmax_temp=1.,
    aggregation='fedavg',
    prev_grads=None,
    global_params=None,
    feddyn_alpha=0.,
    clip=1,
):
    """Train the network on the training set. Returns average
    accuracy and loss. """
    if finetune_batch == 0:
        return 0

    criterion = torch.nn.CrossEntropyLoss()
    if kl_loss:
        kl_criterion = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)

    if aggregation == 'feddyn':
        assert prev_grads is not None
        assert global_params is not None
        for k in prev_grads.keys():
            global_params[k] = global_params[k].to(device)
            prev_grads[k] = prev_grads[k].to(device)

    net.train()
    if mu > 0:
        last_round_model = copy.deepcopy(net)
    avg_loss = 0.0
    total = 0

    trainloader = iter(cycle(trainloader))
    if valloader:
        valloader = iter(cycle(valloader))
    
    if kl_loss == 'dynamic_ma':
        ma_train_loss = np.array([0.] * (max_exit_layer + 1))

    for _ in range(finetune_batch):
        images, labels = next(trainloader)

        images, labels = images.to(device), labels.to(device)
        net.zero_grad()

        if images.size(1) == 1: # single-channel
            images = images.expand(-1, 3, images.shape[2], images.shape[3])

        all_outputs = net(images)

        loss = 0.
        if kl_loss:
            _kl_loss = 0.
        for i, output in enumerate(all_outputs):                
            if mu > 0:
                # FedProx: compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(net.parameters(), last_round_model.parameters()):
                    proximal_term += (w - w_t).norm(2)

                train_loss = criterion(output, labels) + (mu / 2) * proximal_term
            else:
                train_loss = criterion(output, labels)
            
            loss += train_loss

            if kl_loss == 'dynamic_ma':
                ma_train_loss[i] = 0.8 * ma_train_loss[i] + 0.2 * train_loss.item()
            
            if kl_loss == 'all':
                for j, target_output in enumerate(all_outputs):
                    if j == i:
                        continue

                    _kl_loss += compute_kl_loss(kl_criterion, all_outputs[i], target_output.detach(), softmax_temp=kl_softmax_temp)

        if max_exit_layer > 0:
            if kl_loss == 'forward': 
                # distill using max budget exit as target
                for j in range(len(all_outputs) - 1):
                    _kl_loss += compute_kl_loss(kl_criterion, all_outputs[j], all_outputs[-1].detach(), softmax_temp=kl_softmax_temp)             
            elif kl_loss == 'dynamic': 
                # distill using exit with the lowest validation score
                with torch.no_grad():
                    if valloader:
                        v_images, v_labels = next(valloader)
                    else: # simply use another batch of train data as validation
                        v_images, v_labels = next(trainloader)
                    v_images, v_labels = v_images.to(device), v_labels.to(device)

                    if v_images.size(1) == 1: # single-channel
                        v_images = v_images.expand(-1, 3, v_images.shape[2], v_images.shape[3])

                    all_val_outputs = net(v_images)
                    min_i = torch.argmin(torch.stack([criterion(v_outputs, v_labels) for v_outputs in all_val_outputs]))
                for j in range(len(all_outputs)):
                    if j != min_i:
                        _kl_loss += compute_kl_loss(kl_criterion, all_outputs[j], all_outputs[min_i].detach(), softmax_temp=kl_softmax_temp)
            elif kl_loss == 'dynamic_ma':
                # distill using exit with the lowest moving avereage training score
                min_i = np.argmin(ma_train_loss)
                for j in range(len(all_outputs)):
                    if j != min_i:
                        _kl_loss += compute_kl_loss(kl_criterion, all_outputs[j], all_outputs[min_i].detach(), softmax_temp=kl_softmax_temp)
                    
        if kl_loss and max_exit_layer > 0:
            if kl_loss in ['all']:
                _kl_loss /= max_exit_layer

            if kl_weight:
                _kl_loss = kl_weight * _kl_loss
            loss += _kl_loss

        if aggregation == 'feddyn':
            for k, param in net.named_parameters():
                if k in net.trainable_state_dict_keys:
                    curr_param = param.flatten()
                    assert prev_grads[k].size() == curr_param.size()

                    lin_penalty = torch.dot(curr_param, prev_grads[k])
                    loss -= lin_penalty

                    quad_penalty = (
                        feddyn_alpha / 2.0
                        * torch.sum(torch.square(curr_param - global_params[k]))
                    )
                    loss += quad_penalty 

        loss.backward()
        
        if clip:
            torch.nn.utils.clip_grad_value_(net.parameters(), clip)

        # apply gradients
        optimizer.step()
        
        # get statistics
        avg_loss += loss.item()        
        total += images.shape[0]
    
    return avg_loss / total


class ReeFLClassificationClient(ClassificationClient):
    def __init__(
        self,
        cid: str,
        lid: int,
        width_scale: float,
        *args,
        kl_loss: str  = '',
        kl_consistency_weight: int = 300,
        kl_weight=None,
        kl_softmax_temp=1.,
        aggregation: str = 'fedavg',
        clip=1.0,
        **kwargs
    ):
        super(ReeFLClassificationClient, self).__init__(cid, *args, **kwargs)
        self.lid = lid
        self.width_scale = width_scale

        # kl hyperparameters
        self.kl_loss = kl_loss 
        self.kl_consistency_weight = kl_consistency_weight
        self.kl_weight = kl_weight
        self.kl_softmax_temp = kl_softmax_temp
        assert self.kl_loss in ['','forward','all','dynamic','dynamic_ma']
        
        self.val_set = self.kl_loss == 'dynamic'
        self.aggregation = aggregation    
        self.clip = clip
        
        assert self.aggregation in ['fedavg', 'feddyn']

        # get subset of model
        arch_fn = get_func_from_config(self.net_config)
        net_args = copy.deepcopy(self.net_config.args)
        blk_to_exit = self.net.blks_to_exit[self.lid]
        net_args.depth = blk_to_exit + 1
        net_args.blks_to_exit = self.net.blks_to_exit[:self.lid+1]
        net_args.width_scale = self.width_scale

        self.net = arch_fn(device=self.device, **net_args)
        self.trainable_state_keys = self.net.trainable_state_dict_keys

        self.feddyn_alpha = 0.
        self.prev_grads = None
        if self.aggregation == 'feddyn':
            self.feddyn_alpha = self.ckp.config.server.strategy.args.alpha
            self.prev_grads_filepath = os.path.join(self.ckp.run_dir, f'prev_grads/{cid}')
            # print(f'client {self.cid} loading from {self.prev_grads_filepath}')
            self.prev_grads = self.ckp.offline_load(self.prev_grads_filepath)
            if self.prev_grads is None:
                self.prev_grads = {k: torch.zeros(v.numel()) for (k, v) in self.net.named_parameters() if k in self.trainable_state_keys}

    def get_parameters(self):
        return [val.cpu().numpy() for k, val in self.net.state_dict().items() if k in self.trainable_state_keys]
                
    def set_parameters(self, parameters):
        assert len(parameters) == len(self.trainable_state_keys), f'length of parameters did not match the number of trainable state keys'
        params_dict = zip(self.trainable_state_keys, parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})

        if self.width_scale is not None and self.width_scale < 1.:
            param_idx = {}
            model_state_dict = self.net.state_dict()

            for k in model_state_dict.keys():
                param_idx[k] = [
                    torch.arange(size) for size in model_state_dict[k].shape
                ]  

            state_dict = prune(state_dict, param_idx)
        
        self.net.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, round_config, num_workers=None):
        # print(f"fit() on client cid={self.cid}")
        round_config = AttrDict(round_config)
        self.set_parameters(parameters)

        global_params = None
        if self.aggregation == 'feddyn':
            global_params = {
                k: val.detach().clone().flatten() for k, val in self.net.named_parameters() if k in self.trainable_state_keys
            }
        # print(f"[DEBUG] Loaded {len(parameters)} keys successfully")

        rnd = int(round_config.current_round)
        # load data for this client and get trainloader
        if num_workers is None:
            num_workers = len(ray.worker.get_resource_ids()["CPU"])

        valloader = None
        if self.val_set:
            trainloader, valloader = self.dataloader(
                data_pool='train',
                cid=self.cid,
                partition='train',
                batch_size=int(self.batch_size),
                num_workers=num_workers,
                shuffle=True,
                augment=True,
                val_ratio=0.2,
            )
        else:
            trainloader = self.dataloader(
                data_pool='train',
                cid=self.cid,
                partition='train',
                batch_size=int(self.batch_size),
                num_workers=num_workers,
                shuffle=True,
                augment=True,
            )

        # send model to device
        self.net.to(self.device)

        # get optimizer type
        optim_func = get_func_from_config(self.net_config.optimizer)
        trainable_layers = list(map(lambda x: x[1],filter(lambda n: n[0] in self.trainable_state_keys, self.net.named_parameters())))                

        optimizer = optim_func(
            [
                {'params': trainable_layers}
            ],
            lr=float(round_config.lr),
            **self.net_config.optimizer.args,
        )

        # determine which exit to use
        max_early_exit_layer = self.lid

        # convert epochs to num of finetune_batches
        total_fb = epochs_to_batches(self.local_epochs, len(trainloader.dataset), self.batch_size)

        # consistency weight taken from depthFL: 
        # https://github.com/adap/flower/blob/main/baselines/depthfl/depthfl/client.py#L88C10-L88C65
        kl_weight = self.kl_weight
        if self.kl_loss and kl_weight is None: # kl_weight is not constant
            current = np.clip(rnd, 0.0, self.kl_consistency_weight)
            phase = 1.0 - current / self.kl_consistency_weight
            kl_weight = float(np.exp(-5.0 * phase * phase))

        # train
        loss = train(
                            self.net,
                            max_early_exit_layer,
                            trainloader,
                            valloader,
                            optimizer=optimizer,
                            kl_loss=self.kl_loss,
                            kl_weight=kl_weight,
                            kl_softmax_temp=self.kl_softmax_temp,
                            finetune_batch=int(total_fb),
                            device=self.device,
                            round=rnd,
                            mu=self.fedprox_mu,
                            aggregation=self.aggregation,
                            prev_grads=self.prev_grads,
                            global_params=global_params,
                            feddyn_alpha=self.feddyn_alpha,
                            clip=self.clip,
                        )

        if self.aggregation == 'feddyn':
            for k, param in self.net.named_parameters():
                if k in self.trainable_state_keys:
                    curr_param = param.detach().clone().flatten()
                    self.prev_grads[k] = self.prev_grads[k] - self.feddyn_alpha * (curr_param - global_params[k])
                    self.prev_grads[k] = self.prev_grads[k].to(torch.device('cpu'))

            # print(f'client {self.cid} saving to {self.prev_grads_filepath}')
            self.ckp.offline_save(self.prev_grads_filepath, self.prev_grads)

        # return local model
        return self.get_parameters(), len(trainloader.dataset), {"fed_train_loss": loss}

    def evaluate(self, parameters, round_config, num_workers=None, finetune=True, path=None):
        # Personalized FL. Evaluate on test pool
        # print(f"evaluate() on client cid={self.cid}")
        round_config = AttrDict(round_config)
        self.set_parameters(parameters)

        global_params = None
        if self.aggregation == 'feddyn':
            global_params = {
                k: val.detach().clone().flatten() for k, val in self.net.named_parameters()
            }

        rnd = int(round_config.current_round)

        if num_workers is None:
            # get num_workers based on ray assignment
            num_workers = len(ray.worker.get_resource_ids()["CPU"])

        finetune_epochs = round_config.finetune_epochs
        
        # send model to device
        self.net.to(self.device)

        if finetune:
            if type(finetune_epochs) != list:
                ft_b = [int(finetune_epochs)]
                finetune_epochs = [int(finetune_epochs)]
            else:
                ft_b = finetune_epochs
                if len(ft_b) > 1:
                    ft_b = [ft_b[0]] + [y - x for x, y in zip(ft_b, ft_b[1:])]
        else:
            ft_b = [0]
            finetune_epochs=[0]

        valloader = None
        if self.val_set:
            trainloader, valloader = self.dataloader(
                data_pool='test',
                cid=self.cid,
                partition='train',
                batch_size=int(self.batch_size),
                num_workers=num_workers,
                augment=True,
                shuffle=True,
                val_ratio=0.2,
                path=path,
            )
        else:
            trainloader = self.dataloader(
                data_pool='test',
                cid=self.cid,
                partition='train',
                batch_size=int(self.batch_size),
                num_workers=num_workers,
                augment=True,
                shuffle=True,
                path=path,
            )

        testloader = self.dataloader(
            data_pool='test',
            cid=self.cid, 
            partition='test', 
            batch_size=50,
            augment=False, 
            num_workers=num_workers,
            path=path
        )

        # converting finetune epochs to finetune batches
        ft_b = [epochs_to_batches(b, len(trainloader.dataset), self.batch_size) for b in ft_b]

        # get optimizer type
        optim_func = get_func_from_config(self.net_config.optimizer)
        trainable_layers = list(map(lambda x: x[1],filter(lambda n: n[0] in self.trainable_state_keys, self.net.named_parameters())))                

        optimizer = optim_func(
            [
                {'params': trainable_layers}
            ],
            lr=float(round_config.lr),
            **self.net_config.optimizer.args,
        )

        # determine which exit to use
        max_early_exit_layer = self.lid

        metrics = {}
        for finetune_batch, ft_epoch in zip(ft_b, finetune_epochs):
            # train
            _, _ = train(
                                self.net,
                                max_early_exit_layer,
                                trainloader,
                                valloader,
                                optimizer=optimizer,
                                kl_loss=self.kl_loss,
                                finetune_batch=finetune_batch,
                                device=self.device,
                                round=rnd,
                                mu=self.fedprox_mu,
                                aggregation=self.aggregation,
                                prev_grads=self.prev_grads,
                                feddyn_alpha=self.feddyn_alpha,
                                clip=self.clip,
                            )

            # evaluate
            results = ree_early_exit_test(self.net, max_early_exit_layer, testloader, device=self.device)
            metrics[f'ps_test_acc_{ft_epoch}_exit{max_early_exit_layer}'] = results[max_early_exit_layer]['accuracy'] * 100
        
        # return statistics
        return float(loss), len(testloader.dataset), {**metrics, "accuracy": float(results[max_early_exit_layer]['accuracy'] * 100)}