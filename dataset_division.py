from torchvision import datasets, transforms
import numpy as np
import random
import torch
import os
import glob
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import io

def get_dataset(dir, conf):
    # Generate client data
    if conf["dataset_name"] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandAugment(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #     transforms.RandomErasing(p=0.25),
        # ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # train
        train_dataset = datasets.CIFAR10(dir, train=True, download=True,transform=transform_train)
        # test
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
    elif conf["dataset_name"] == 'cifar100':
        # transform_train = transforms.Compose([
        #     transforms.RandomCrop(32, padding=4),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        # ])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            transforms.RandomErasing(p=0.25),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        # train
        train_dataset = datasets.CIFAR100(dir, train=True, download=True,transform=transform_train)
        # test
        eval_dataset = datasets.CIFAR100(dir, train=False, transform=transform_test)
    elif conf["dataset_name"] == 'CAMELYON17-WILDS':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        # train
        train_dataset = Camelyon17WildsDataset(dir = dir, prefix="train", target_centers=[0, 3, 4], transform=transform_train)
        # test
        eval_dataset = Camelyon17WildsDataset(dir = dir, prefix="test", target_centers=[2], transform=transform_test)
        clients_distribution = train_dataset.clients_distribution
        train_clients_index = train_dataset.clients_index
        eval_clients_index = {}
        eval_clients_index[0] = eval_dataset.clients_index[0]
        eval_clients_index[1] = eval_dataset.clients_index[0]
        eval_clients_index[2] = eval_dataset.clients_index[0]
        return train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index
    else:
        print("Wrong dataset name.")

    # sample training data amongst users
    if conf["data_distribution"] == 'IID':
        clients_distribution, train_clients_index, eval_clients_index = client_iid(train_dataset, eval_dataset,
                                                                                   conf)
    elif conf["data_distribution"] == 'Dirichlet':
        clients_distribution, train_clients_index, eval_clients_index = client_noniid_Dirichlet(train_dataset,
                                                                                                eval_dataset, conf)
    elif conf["data_distribution"] == 'LongTail':
        clients_distribution, train_clients_index, eval_clients_index = client_noniid_LongTail(train_dataset,
                                                                                           eval_dataset, conf)
    else:
        print("Client data distribution error，The corrects are：IID、Dirichlet、LongTail.")
    return train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index

# The samples in the dataset are separated by label
def get_every_lable_list(dataset):
    # training and test sets, we separate the list by label and split it into lists of labels
    num_class = len(dataset.classes)
    # Holds the index values for 10 categories, and the elements in the list are lists
    every_lable_list = []
    for _ in range(num_class):
        every_lable_list.append([])
    # lable
    lable_list = list(dataset.class_to_idx.values())
    for i in range(len(dataset.targets)):
        lable_index = lable_list.index(dataset.targets[i])
        every_lable_list[lable_index].append(i)
    # shuffle
    for l in every_lable_list:
        random.shuffle(l)
    return every_lable_list

# Assign data to each client according to the data distribution
def fill_data_with_distribution(dataset, clients_distribution, conf):
    # Number of classes in the dataset
    num_class = len(dataset.classes)
    # Number of samples per client
    # When the num_small_data hyperparameter is 0, the training data is split equally among all clients;
    # When the num_small_data hyperparameter is a specific number, the data is directly the number of client samples,
    # and the small data scenario in the paper is 50
    num_client_data = int(len(dataset.targets) / conf["num_client"])
    # The samples in the dataset are separated by label
    every_lable_list = get_every_lable_list(dataset)
    clients_index = {}
    for i in range(conf["num_client"]):
        clients_index[i] = []
        distribution_data = (clients_distribution[i] * num_client_data).astype('int')
        # Deal with small problems that are not divisible
        distribution_data[np.random.randint(low=0, high=num_class)] += (num_client_data - np.sum(distribution_data))
        # Add samples to each class one by one
        for j in range(num_class):
            num_samples = distribution_data[j]
            # Avoid some special case bugs, rarely used.
            # Dirichlet distribution, there are some extreme cases where you don't have enough samples for some classes.
            if len(every_lable_list[j]) < num_samples:
                every_lable_list[j] += get_every_lable_list(dataset)[j]
            # add samples
            for k in range(num_samples):
                clients_index[i].append(every_lable_list[j].pop())
    return clients_index

def client_iid(train_dataset, eval_dataset, conf):
    num_class = len(train_dataset.classes)
    clients_distribution = {}
    # iid is the label uniform distribution
    for i in range(conf["num_client"]):
        clients_distribution[i] = np.ones(num_class) / num_class
    train_clients_index = fill_data_with_distribution(train_dataset, clients_distribution, conf)
    eval_clients_index = fill_data_with_distribution(eval_dataset, clients_distribution, conf)
    return clients_distribution, train_clients_index, eval_clients_index

def client_noniid_Dirichlet(train_dataset, eval_dataset, conf):
    num_class = len(train_dataset.classes)
    clients_distribution = {}
    # Dirichlet distribution alpha parameter, uniformly distributed over num_clients [dirichlet_alpha_min,dirichlet_alpha_max]
    alpha_list = []
    for i in range(conf["num_client"]):
        alpha_list.append(
            ((conf["dirichlet_alpha_max"] - conf["dirichlet_alpha_min"]) / conf["num_client"]) * (i + 1) + conf[
                "dirichlet_alpha_min"])
    # The distribution of each client is sampled following the alpha parameter of the Dirichlet distribution
    for i in range(conf["num_client"]):
        clients_distribution[i] = np.random.dirichlet((np.ones(num_class) * alpha_list[i]), 1)[0]
    train_clients_index = fill_data_with_distribution(train_dataset, clients_distribution, conf)
    eval_clients_index = fill_data_with_distribution(eval_dataset, clients_distribution, conf)
    return clients_distribution, train_clients_index, eval_clients_index

def client_noniid_LongTail(train_dataset, eval_dataset, conf):
    # IID is used for validation data
    num_class = len(train_dataset.classes)
    clients_distribution = {}
    for i in range(conf["num_client"]):
        clients_distribution[i] = np.ones(num_class) / num_class
    eval_clients_index = fill_data_with_distribution(eval_dataset, clients_distribution, conf)
    # The training data has a long tail, so the amount of data is not always the same
    # clients_distribution  should be long-tailed distribution
    for i in range(conf["num_client"]):
        clients_distribution[i] = np.ones(num_class)
        sum = 0.0
        for j in range(num_class):
            clients_distribution[i][j] *= (conf["longtail_exp"] ** j)
            sum += (conf["longtail_exp"] ** j)
        clients_distribution[i] /= sum
    # The training data simply removes some from the total data to become a long-tailed distribution
    # This list temporarily stores all the index data of all categories
    every_lable_list = get_every_lable_list(train_dataset)
    # In the "every_lable_list", the number of data for each class is determined one by one according to the exponential distribution,
    # and then all of them are placed in the "all_data_list" list.
    all_data_list = []
    for x in range(len(every_lable_list)):
        lable_x_num = len(every_lable_list[x]) * (conf["longtail_exp"] ** x)
        while (len(every_lable_list[x]) > lable_x_num):
            every_lable_list[x].pop()
        all_data_list += every_lable_list[x]
    # Shuffle the all_data_list list so that all the data is evenly placed in each client
    random.shuffle(all_data_list)
    train_clients_index = {i: [] for i in range(conf["num_client"])}
    while (len(all_data_list) > 0):
        for indexs in train_clients_index.values():
            if len(all_data_list) > 0:
                indexs.append(all_data_list.pop())
    return clients_distribution, train_clients_index, eval_clients_index

class Camelyon17WildsDataset(Dataset):
    def __init__(self, dir, prefix, target_centers=None, transform=None):
        # dir: Directory where the file is located; prefix: File prefix, such as "train", "val", "test"; transform: Image transformation
        self.transform = transform
        dir = dir+'CAMELYON17-WILDS'
        pattern = os.path.join(dir, f"{prefix}-*.parquet")
        files = sorted(glob.glob(pattern))
        if len(files) == 0:
            raise FileNotFoundError(f"No matching Parquet file was found: {pattern}")
        print(f"loading {len(files)} {prefix} parquet file...")
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        # Perform data filtering based on the input list of target_centers.
        if target_centers is not None:
            if isinstance(target_centers, (int, str)):
                target_centers = [target_centers]
            df = df[df["center"].isin(target_centers)]
            print(f"-> Filtering the data of Centers {target_centers}: There are {len(df)} remaining samples.")
        else:
            print(f"-> Total {prefix} sample count: {len(df)}")
        # Reset the index. Make sure that the index of the Dataset starts from 0 and is consecutive.
        self.df = df.reset_index(drop=True)
        # Build the central index dictionary and the label proportion dictionary
        self.clients_index = {}
        self.clients_distribution = {}
        key = 0
        for center_val in self.df["center"].unique():
            idx_list = self.df[self.df["center"] == center_val].index.tolist()
            self.clients_index[key] = idx_list
            center_df = self.df[self.df["center"] == center_val]
            total_samples = len(center_df)
            count_0 = (center_df["label"] == 0).sum()
            count_1 = (center_df["label"] == 1).sum()
            ratio_0 = count_0 / total_samples if total_samples > 0 else 0.0
            ratio_1 = count_1 / total_samples if total_samples > 0 else 0.0
            self.clients_distribution[key] = np.array([ratio_0, ratio_1])
            key += 1
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_data = row["image"]
        if isinstance(img_data, dict):
            img_bytes = img_data["bytes"]
        elif isinstance(img_data, bytes):
            img_bytes = img_data
        else:
            raise ValueError(f"Unrecognizable image format: {type(img_data)}")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label