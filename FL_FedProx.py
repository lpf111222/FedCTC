import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy
import random
import argparse
import json
import os
import model, server_base, client_base, util, dataset_division


class FedProx_Client(client_base.ClientBase):
    def __init__(self, global_model, conf, train_dataset, eval_dataset, data_distribution, train_index, eval_index, id):
        super().__init__(conf, train_dataset, eval_dataset, data_distribution, train_index, eval_index, id)
        # Create client local model
        self.local_model = copy.deepcopy(global_model)

    def local_training(self, old_model_parameters):
        # The global model parameter Settings issued by the server are entered into the local model
        self.local_model.load_state_dict(old_model_parameters)
        # Set the model in training mode
        self.local_model.train()
        # optimizer
        if self.conf["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf["lr"])
        elif self.conf["optimizer"] == "Adagrad":
            optimizer = torch.optim.Adagrad(self.local_model.parameters(), lr=self.conf["lr"])
        else:
            print('This experiment does not include any other optimizer.')
        # Local training epochs
        train_error_loss = 0.0
        for e in range(self.conf["local_epochs"]):
            # The data for each epoch training is shuffled
            random.shuffle(self.train_data_indices_lists)
            for batch_id, batch in enumerate(self.train_data_loader):
                data = batch[0].to(self.device)
                target = batch[1].to(self.device)
                optimizer.zero_grad()
                logits = self.local_model(data)
                loss = F.cross_entropy(logits, target, reduction='mean')
                ############ FedProx参数的变化量做为loss
                proximal_term = 0.0
                for name, param in self.local_model.named_parameters():
                    proximal_term += ((param - old_model_parameters[name]).norm(2) ** 2) * 0.5 * 0.001
                loss += proximal_term
                ############
                loss.backward()
                optimizer.step()
                train_error_loss += loss.item() * len(target)
        train_error_loss /= (self.conf["local_epochs"] * self.train_data_quantity)
        train_complexity_loss = 0.0
        # Read the locally trained parameters
        trained_model_parameters = self.local_model.state_dict()
        # print
        print('Client %d completes local training, error loss: %f' % (self.client_id, train_error_loss))
        # Return: trained parameters, number of client data samples, loss
        return trained_model_parameters, self.train_data_quantity, train_error_loss, train_complexity_loss


class FedProx_Server(server_base.ServerBase):
    def __init__(self, conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index):
        super().__init__(conf, eval_dataset)
        self.server_name = 'FedProx'
        # Create a server model
        self.global_model = model.get_model(conf)
        # The client is  a member of the server
        self.clients = []
        for i in range(conf["num_client"]):
            self.clients.append(
                FedProx_Client(self.global_model, conf, train_dataset, eval_dataset, clients_distribution[i],
                             train_clients_index[i],
                             eval_clients_index[i], i))
        return None

if __name__ == '__main__':
    # Read the hyperparameters stored in conf.json
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        conf = json.load(f)
    print('hyperparameters are: ',conf)

    # Set the random seed
    util.set_ramdom_seed(conf["ramdom_seed"])

    # Use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # read dateset
    train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index = dataset_division.get_dataset('./data/', conf)
    print('Dateset is OK.')

    # Create FL server, train, and evaluate
    FedProx = FedProx_Server(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index)
    FedProx.global_train()
    FedProx.global_model_get_all_metrics()