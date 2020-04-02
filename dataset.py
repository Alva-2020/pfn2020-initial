from __future__ import annotations
from collections import OrderedDict
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from loader import accimage_loader, pil_loader


class DomainAwareDataset(Dataset):
    r"""Dataset for domain adaptation tasks in computer vision.

    Dataset expects the following file hierarchy:
    dataset_dir / domain_dirs / class_dirs / files.

    Unlike vanilla datasets with train/test split indices, this dataset loads all examples at once.
    Use .set_active_split() to toggle between loading different data splits when your DataLoader calls
    .__getitem__(), e.g. between train/val/test splits or your specified cross-validation folds.

    dataset_dir:        path to dataset
    included_domains:   domains to include in this dataset. None=all, or list<int>. Default None
    included_classes:   classes to include in this dataset  None=all, or list<int>. Default None
    splits          :   list<str,float> for name of split e.g. 'train' or 'test' and weighting.
                        e.g. [('train', 0.8), ('test', 0.2)] specifies 80/20 train/test split.
                        Default None, which includes all data under one split..
    """

    def __init__(self, dataset_dir:str, included_domains:list=None, included_classes:list=None,
            splits:list=None, transform=None, loader=accimage_loader):

        # This constructs the dataset in-memory, since we're just manipulating strings.

        super().__init__()
        self.dir = dataset_dir
        self.transform = transform
        self.loader = loader

        self.domain2idx = {} # dict<str, int> mapping domain names to idx
        self.class2idx = {}  # dict<str, int> mapping class names to idx
        self.idx2domain = {}
        self.idx2class = {}
        self.split2idx = {}  # For indexing counts
        self.idx2split = {}
        self.filepaths = {}  # dict<int, dict<int, dict<str, list>>> mapping domain -> class -> split --> examples
                             # This isn't used for anything besides convenience and checking. Feel free to disable.
        self.counts = None   # Torch tensor of size (n_domains, n_classes, n_splits) recording num of samples
                             # Use this to compute relative class frequencies if we need to weigh them.
        self.data = OrderedDict()   # dict<str,list<(str,int,int)>> main dict holding data. tuples are (filename, class_idx, domain_idx).
                                    # Used for __getitem__.

        self.splits = splits
        if self.splits is None:
            self.data[None] = {}
            self.split2idx[None] = 0
            self.idx2split[0] = None
        else:
            for i, (split_name, _) in enumerate(self.splits):
                self.data[split_name] = {}
                self.split2idx[split_name] = i
                self.idx2split[i] = split_name
            self.active_split = 0
        
        # setup domains
        for d, domain_name in enumerate(os.listdir(self.dir)):
            self.domain2idx[domain_name] = d
            self.idx2domain[d] = domain_name
        # setup classes
        domain_dir = os.path.join(self.dir, os.listdir(self.dir)[0])
        for c, class_name in enumerate(os.listdir(domain_dir)):
            self.class2idx[class_name] = c
            self.idx2class[c] = class_name

        self.active_split = 0
        self.active_domain = 0

        self.num_splits = 1 if self.splits is None else len(self.splits)
        self.num_domains = len(self.domain2idx.keys())
        self.num_classes = len(self.class2idx.keys())
        self.counts = torch.zeros((self.num_splits, self.num_domains, self.num_classes), dtype=torch.int)
        self.included_domains = set(included_domains) if included_domains is not None else set(list(self.domain2idx.values()))
        self.included_classes = set(included_classes) if included_classes is not None else set(list(self.class2idx.values()))

        if self.splits is None: # Calculate indices to split along
            w = [1.0]
        else:
            _, w = zip(*self.splits)
            sum_w = np.sum(w)
            w = w / sum_w

        for i in range(self.num_splits):

            split_name = self.splits[i][0] if self.splits is not None else None
            if split_name not in self.data:
                self.data[split_name] = {}

            for domain_name, d in self.domain2idx.items():
                if d not in self.included_domains:
                    continue
                if d not in self.data[split_name]:
                    self.data[split_name][d] = []

                for class_name, c in self.class2idx.items():
                    if c not in self.included_classes:
                        continue

                    dirpath = os.path.join(self.dir, domain_name, class_name)
                    filepaths = list(os.path.join(dirpath, filepath) for filepath in os.listdir(dirpath))

                    start_idx = 0 if i == 0 else int(np.floor(np.sum(w[:i]) * len(filepaths)))
                    end_idx = int(np.floor(np.sum(w[:i+1]) * len(filepaths)))
                    count = end_idx - start_idx
                    self.counts[i][d][c] = count
                    # TODO: warn if count drops to zero.
                    items = list((filepath, c, d) for filepath in filepaths[start_idx:end_idx])
                    self.data[split_name][d].extend(items)

            start_idx = end_idx

        print("Dataset loaded.")
        print("num_domains  : {}".format(self.num_domains))
        print("num_classes  : {}".format(self.num_classes))
        print("num_splits   : {}".format(self.num_splits))
        self._print_active_split()
        self._print_active_domain()
        self._print_len()

    def _print_active_split(self):
        print("active_split : {} ({})".format(self.active_split, self.idx2split[self.active_split]))

    def _print_active_domain(self):
        print("active_domain: {} ({})".format(self.active_domain, self.idx2domain[self.active_domain]))

    def _print_len(self):
        print("len          : {}".format(len(self.data[self.idx2split[self.active_split]][self.active_domain])))


    def __getitem__(self, idx):
        filepath, class_idx, domain_idx = self.data[self.idx2split[self.active_split]][self.active_domain][idx]
        img = self.loader(filepath)
        if self.transform is not None:
            img = self.transform(img)
        return (img, class_idx, domain_idx)
    

    def __len__(self):
        return len(self.data[self.idx2split[self.active_split]][self.active_domain])


    def set_active_split(self, split) -> DomainAwareDataset:
        # Toggles the dataset to use the specified data split.
        # Use to toggle between train/test splits, for example.
        if isinstance(split, str):
            if split not in self.split2idx.keys():
                raise KeyError("Split {} not registered.".format(split))
            else:
                self.active_split = self.split2idx[split]
        elif isinstance(split, int):
            if split not in self.split2idx.values():
                raise KeyError("Split {} not registered.".format(split))
            else:
                self.active_split = split
        # print("Split toggled.")
        # self._print_active_split()
        # self._print_len()
        return self


    def set_active_domain(self, domain) -> DomainAwareDataset:
        if isinstance(domain, str):
            if domain not in self.domain2idx.keys():
                raise KeyError("Domain {} not registered.".format(domain))
            else:
                self.active_domain = self.domain2idx[domain]
        elif isinstance(domain, int):           
            if domain not in self.domain2idx.values():
                raise KeyError("Domain index {} not registered.".format(domain))
            else:
                self.active_domain = domain
        else:
            raise TypeError("Argument {} not recognized.".format(domain))
        print("Domain toggled.")
        self._print_active_domain()
        self._print_len()
        return self

    def set_transform(self, transform) -> DomainAwareDataset:
        # Use this to toggle between different transforms during train/test
        self.transform = transform
        return self



class OfficeHomeDataset(DomainAwareDataset):

    def __init__(self, dataset_dir='data/officehome/raw',
            included_domains:list=None, included_classes:list=None, splits:list=None,
            transform=None, loader=accimage_loader):
        super().__init__(dataset_dir, included_domains, included_classes, splits, transform, loader)

