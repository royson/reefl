import torch
import numpy as np
import os
from pathlib import Path
from .fl_dataset import FederatedDataset
from .utils import VisionDataset_FL
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)

def cifar100Transformation(augment):
    if augment == 'jit':
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    elif augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                ),
            ]
        )

class Cifar100Dataset(FederatedDataset):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.data_dir = self.path_to_data
        self.path_to_data = Path(os.path.join(self.path_to_data, 'cifar-100-python'))
        self.jit_augment = torch.nn.Sequential(
                transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                ),
            )
        self.jit_normalize = torch.nn.Sequential(
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    (0.5070751592371323, 0.48654887331495095, 0.4409178433670343), \
                    (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
                ),
            )
        
    def download(self):
        # download trainset
        if not os.path.exists(self.path_to_data):
            logger.info("Generating unified CIFAR dataset")
            # test set
            test_path = self.path_to_data / "test.pt"
            test_data = datasets.CIFAR100(root=self.data_dir, train=False, download=True)
            X_test, y_test = test_data.data, np.array(test_data.targets, dtype=np.int64)
            torch.save([X_test, np.array(y_test, dtype=np.int64)], test_path)

            # train and possible validation
            train_path = self.path_to_data / "train.pt"
            train_data = datasets.CIFAR100(root=self.data_dir, train=True, download=True)
            X_train, y_train = train_data.data, np.array(train_data.targets, dtype=np.int64)

            torch.save([X_train, y_train], train_path)

    def get_dataloader(self, 
                    data_pool, 
                    partition,
                    batch_size,
                    num_workers, 
                    augment,
                    cid=None, 
                    path=None,
                    shuffle=False,
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
            elif data_pool == 'train':
                # load training pool of clients
                prefix_path = self.fed_train_dir if cid is None else os.path.join(self.fed_train_dir, cid)
                path = os.path.join(prefix_path, f'{partition}.pt')
            else:
                # load test pool of clients
                prefix_path = self.fed_test_dir if cid is None else os.path.join(self.fed_test_dir, cid)
                path = os.path.join(prefix_path, f'{partition}.pt')

        # print(f'Getting dataloader.. data_pool: {data_pool}, partition: {partition}. \n path: {path}. augment: {augment}. ')

        if val_ratio:
            assert partition.lower() == 'train'

            dataset = VisionDataset_FL(path_to_data=path, 
                transform=cifar100Transformation(augment))

            val_len = int(val_ratio * len(dataset))

            datasets = torch.utils.data.random_split(dataset, [len(dataset) - val_len, val_len], 
                                        generator=torch.Generator().manual_seed(seed) if seed else torch.Generator())            

            return [DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=shuffle, **kwargs) for dataset in datasets]

        else:
            dataset = VisionDataset_FL(path_to_data=path, 
                transform=cifar100Transformation(augment))

            return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=shuffle, **kwargs)