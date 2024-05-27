import numpy as np
import os
import torch
import json
from collections import defaultdict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from .fl_dataset import FederatedDataset
from .utils import VisionDataset_FL
import pickle
from typing import List
from PIL import Image

import logging
logger = logging.getLogger(__name__)


def read_femnist_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def femnistTransformation(augment):
    if augment == 'jit':
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    elif augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
            ]
        )

class FemnistDataset(FederatedDataset):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.num_clients = self.ckp.config.simulation.num_clients

        self.jit_augment = transforms.Compose([
                                            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                            transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)),
                                            ])
        self.jit_normalize = self.jit_augment

        if not os.path.exists(os.path.join(self.dataset_fl_root, 'test.pt')):
            self.create_fl_partitions()

    def create_fl_partitions(self):
        print("Creating FEMNIST Partitions..")
        assert os.path.exists(os.path.join(self.path_to_data, 'train'))
        assert os.path.exists(os.path.join(self.path_to_data, 'test'))

        train_data_dir = os.path.join(self.path_to_data, 'train')
        test_data_dir = os.path.join(self.path_to_data, 'test')
        train_clients, _, train_data = read_femnist_dir(train_data_dir)
        test_clients, _, test_data = read_femnist_dir(test_data_dir)

        assert len(train_clients) == self.num_clients, f'No. of defined clients ({self.num_clients}) do not match with actual number of clients ({len(train_clients)})'
        print(f"Partitioning FEMNIST data with {len(train_clients)} clients")

        assert train_clients == test_clients

        # saving local client data
        train_size = 0
        test_size = 0
        for cid, client in enumerate(train_clients):
            client_path = os.path.join(self.dataset_fl_root, str(cid))
            if os.path.exists(os.path.join(client_path, 'train.pt')) and os.path.exists(os.path.join(client_path, 'test.pt')):
                continue
            train_x, train_y = np.array(train_data[client]['x']), train_data[client]['y']
            test_x, test_y = np.array(test_data[client]['x']), test_data[client]['y']
            train_x = (train_x * 255).astype(np.uint8).reshape(-1,28,28,1)
            test_x = (test_x * 255).astype(np.uint8).reshape(-1,28,28,1)
            train_y = np.array(train_y, dtype=np.int64)
            test_y = np.array(test_y, dtype=np.int64)

            os.makedirs(client_path, exist_ok=True)
            torch.save((train_x, train_y), os.path.join(client_path, 'train.pt'))
            torch.save((test_x, test_y), os.path.join(client_path, 'test.pt'))
            train_size += train_x.shape[0]
            test_size += test_x.shape[0]
        
        print(f'No of train samples: {train_size}. No of test samples: {test_size}')
        
        # saving global test data
        global_test_x, global_test_y = None, None
        for v in test_data.values():
            test_x = (np.array(v['x']) * 255).astype(np.uint8).reshape(-1,28,28,1)
            test_y = np.array(v['y'], dtype=np.int64)

            if global_test_x is None:
                global_test_x = test_x
                global_test_y = test_y
            else:
                global_test_x = np.concatenate((global_test_x, test_x), axis=0)
                global_test_y = np.concatenate((global_test_y, test_y))

        torch.save((global_test_x, global_test_y), os.path.join(self.dataset_fl_root, 'test.pt'), pickle_protocol=4)
    
    def download(self):
        # please pre-partition the full non-IID dataset using LEAF
        return

    def get_available_training_clients(self):
        return list(range(self.num_clients))

    def get_dataloader(self, 
                    data_pool, 
                    partition,
                    batch_size,
                    num_workers, 
                    augment,
                    shuffle=False,
                    cid=None, 
                    path=None,
                    val_ratio=0,
                    seed=None,
                    **kwargs):
        '''
        Return the class specific dataloader from server or client
        '''
        data_pool = data_pool.lower()
        assert data_pool.lower() in ('server', 'train', 'test'), 'Data pool must be in server, train, or test pool'

        if path is not None and os.path.exists(path):
            # forced to use the path 
            # print(f'Forced to use path {path} instead of following data_pool')
            prefix_path = path if cid is None else os.path.join(path, cid)
            path = os.path.join(prefix_path, f'{partition}.pt')
        else:
            if data_pool == 'server':
                assert cid is None
                path = os.path.join(self.dataset_fl_root, f'{partition}.pt')
            else:
                # there is only one pool of clients in femnist
                assert cid is not None
                path = os.path.join(os.path.join(self.dataset_fl_root, cid), f'{partition}.pt')

        # print(f'Getting dataloader.. data_pool: {data_pool}, partition: {partition}. \n path: {path}. augment: {augment}. ')

        if val_ratio:
            assert partition.lower() == 'train'

            dataset = VisionDataset_FL(path_to_data=path, 
                transform=femnistTransformation(augment))

            val_len = int(val_ratio * len(dataset))

            datasets = torch.utils.data.random_split(dataset, [len(dataset) - val_len, val_len], 
                                        generator=torch.Generator().manual_seed(seed) if seed else torch.Generator())          
                                        
            return [DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=shuffle, **kwargs) for dataset in datasets]
        else:
            dataset = VisionDataset_FL(path_to_data=path, 
                transform=femnistTransformation(augment))
            return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=shuffle, **kwargs)