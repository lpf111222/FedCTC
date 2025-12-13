import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import numpy as np
import model, server_base, client_base


class FedCL_MixUp_Client(client_base.ClientBase):
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
                batch_size = data.shape[0]
                # lam is a mixing parameter, and the parameter mixup_alpha is formed by sampling from the beta distribution.
                lam = torch.from_numpy(np.random.beta(self.conf["mixup_alpha"], self.conf["mixup_alpha"], batch_size)).to(
                    torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                for i in range(len(data.shape) - 1):
                    lam.unsqueeze_(i + 1)
                shuffled_index = torch.randperm(batch_size).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                # mix data
                mixed_data = lam * data + (1 - lam) * data[shuffled_index, :]
                # forward
                logits = self.local_model(mixed_data)
                # log_softmax
                log_softmax_outputs = F.log_softmax(logits, dim=1)
                # onehot
                target_onehot = torch.zeros_like(log_softmax_outputs).fill_(0).scatter(1, target.unsqueeze(1), 1).detach()
                # mix lable
                mixed_targets = lam.view(batch_size, -1) * target_onehot + (1 - lam).view(batch_size, -1) * target_onehot[shuffled_index]
                # loss
                loss = -torch.sum(mixed_targets * log_softmax_outputs, dim=1).mean()
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


class FedCL_MixUp_Server(server_base.ServerBase):
    def __init__(self, conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index, MonteCarlo_times=1):
        super().__init__(conf, eval_dataset)
        self.server_name = 'FedCL_MixUp'
        # Create a server model
        self.global_model = model.get_model(conf)
        # The client is  a member of the server
        self.clients = []
        for i in range(conf["num_client"]):
            self.clients.append(
                FedCL_MixUp_Client(self.global_model, conf, train_dataset, eval_dataset, clients_distribution[i],
                             train_clients_index[i],
                             eval_clients_index[i], i))
        return None