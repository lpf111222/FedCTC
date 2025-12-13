import argparse
import json
import os
import dataset_division
import util
import FL_FedAvg
import FL_FedCL_MixUp
import FL_FedCL_LabelSmooth
import FL_FedCL_LogitNorm
import FL_FedCL_3In1


if __name__ == '__main__':

    # Read the hyperparameters stored in conf.json
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        conf = json.load(f)
    print('hyperparameters are: ',conf)

    # Set the random seed
    util.set_ramdom_seed()

    # Use GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # read dateset
    train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index = dataset_division.get_dataset('data/', conf)
    print('Dateset is OK.')

    # Create FL server, train, and evaluate
    fedavg = FL_FedAvg.FedAvgServer(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index)
    fedavg.global_train()
    fedavg.global_model_get_all_metrics()

    fedcl_mixup = FL_FedCL_MixUp.FedCL_MixUp_Server(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index)
    fedcl_mixup.global_train()
    fedcl_mixup.global_model_get_all_metrics()

    fedcl_labelsmooth = FL_FedCL_LabelSmooth.FedCL_LabelSmooth_Server(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index)
    fedcl_labelsmooth.global_train()
    fedcl_labelsmooth.global_model_get_all_metrics()

    fedcl_logitnorm = FL_FedCL_LogitNorm.FedCL_LogitNorm_Server(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index)
    fedcl_logitnorm.global_train()
    fedcl_logitnorm.global_model_get_all_metrics()

    fedcl_3in1 = FL_FedCL_3In1.FedCL_3In1_Server(conf, train_dataset, eval_dataset, clients_distribution, train_clients_index, eval_clients_index)
    fedcl_3in1.global_train()
    fedcl_3in1.global_model_get_all_metrics()
