from torchvision import datasets, transforms
import numpy as np
import random
import torch

def get_dataset(dir, conf):
    # Generate client data
    if conf["dataset_name"] == 'mnist':
        # train
        train_dataset = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())
        # test
        eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())
    elif conf["dataset_name"] == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # train
        train_dataset = datasets.CIFAR10(dir, train=True, download=True,transform=transform_train)
        # test
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
    elif conf["dataset_name"] == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        # train
        train_dataset = datasets.CIFAR100(dir, train=True, download=True,transform=transform_train)
        # test
        eval_dataset = datasets.CIFAR100(dir, train=False, transform=transform_test)
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