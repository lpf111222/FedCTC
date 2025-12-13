import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import random
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.metrics import auc
from scipy.interpolate import interp1d
from torchvision import datasets, transforms
from torchvision.datasets import SVHN
import util


class ServerBase(object):

    def __init__(self, conf, eval_dataset):
        self.conf = conf
        self.global_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # eval dataset
        self.eval_dataset = eval_dataset
        self.eval_data_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=False)
        self.eval_data_quantity = len(eval_dataset.targets)
        # The training and testing information was recorded and saved as excel after completing the training
        self.train_eval_records = pd.DataFrame(index=list(range(self.conf['global_epochs'])),
                                               columns=['train_error_loss',
                                                        'train_complexity_loss',
                                                        'global_eval_acc'], dtype=float)
        # prepare_for_calibration
        self.prepare_for_all_metrics()
        return None

    def global_train(self):
        for e in range(self.conf['global_epochs']):
            # set ramdom seed
            util.set_ramdom_seed(seed=e)
            # broadcast, train, upload
            clients_trained_parameters, train_data_quantity_sum = self.broadcast_train_upload()
            # Server model weight aggregation
            new_model_parameters = self.aggregate_parameters(clients_trained_parameters, train_data_quantity_sum)
            # Save the aggregated server model parameters
            self.global_model.load_state_dict(new_model_parameters)
            # Testing of the server global model and the client personalized model.
            # It is computationally expensive and takes place only once in every N epochs. The figure in the paper is N=10
            if e%10==0:  # if e%10==0:
                self.global_model_eval()
            # Finish
            self.global_epoch += 1
            # Save model parameters
            model_parameters = self.global_model.state_dict()
            torch.save(model_parameters, 'result/' + self.server_name + '_params.pt')
            print("Save model！")
        return None

    def broadcast_train_upload(self):
        # The current model parameters, a copy, next sent to the client
        old_model_parameters = copy.deepcopy(self.global_model.state_dict())
        # k clients are randomly selected to participate in the training
        training_clients_list = random.sample(self.clients, self.conf['k'])
        clients_trained_parameters = []
        train_data_quantity_sum = 0
        train_error_loss_sum = 0.0
        train_complexity_loss_sum = 0.0
        # Training one by one
        for c in training_clients_list:
            trained_model_parameters, train_data_quantity, train_error_loss, train_complexity_loss = c.local_training(old_model_parameters=old_model_parameters)
            # The model parameters uploaded by the client are stored in a list
            clients_trained_parameters.append({'trained_model_parameters': trained_model_parameters, 'train_data_quantity': train_data_quantity})
            train_data_quantity_sum += train_data_quantity
            train_error_loss_sum += (train_error_loss * train_data_quantity)
            train_complexity_loss_sum += (train_complexity_loss * train_data_quantity)
        # Recording the training loss
        train_error_loss = train_error_loss_sum / train_data_quantity_sum
        train_complexity_loss = train_complexity_loss_sum / train_data_quantity_sum
        self.train_eval_records.loc[self.global_epoch, 'train_error_loss']= train_error_loss
        self.train_eval_records.loc[self.global_epoch, 'train_complexity_loss'] = train_complexity_loss
        # print
        print('The server completes round %d of global training，NLL loss：%f，KL loss：%f，' % (self.global_epoch, train_error_loss, train_complexity_loss))
        return clients_trained_parameters, train_data_quantity_sum

    def aggregate_parameters(self, clients_trained_parameters, train_data_quantity_sum):
        # create an empty dictionary ready for our new arguments
        new_model_parameters = dict()
        for name, params in clients_trained_parameters[0]['trained_model_parameters'].items():
            new_model_parameters[name] = torch.zeros_like(params)
        # The weighted summation proposed by FedAvg
        for c in clients_trained_parameters:
            # The weight coefficient is calculated according to the amount of client data
            client_weight = c['train_data_quantity'] / train_data_quantity_sum
            for name, params in c['trained_model_parameters'].items():
                if params.type() == 'torch.cuda.FloatTensor' or params.type() == 'torch.FloatTensor':
                    new_model_parameters[name].add_(params, alpha=client_weight)
        return new_model_parameters

    def global_model_eval(self):    # Server global model test
        with torch.no_grad():
            self.global_model.eval()
            global_correct_num = 0
            global_eval_loss_num = 0.0
            for test_batch_id, test_batch in enumerate(self.eval_data_loader):
                test_X = test_batch[0].to(self.device)
                test_Y = test_batch[1].to(self.device)
                logits = self.global_model(test_X)
                pred_Y = logits.max(1)[1]
                # Count the correct number
                global_correct_num += pred_Y.eq(test_Y.view_as(pred_Y)).sum().item()
            # acc , loss
            global_eval_acc = global_correct_num / self.eval_data_quantity * 100.0
            self.train_eval_records.loc[self.global_epoch, 'global_eval_acc'] = global_eval_acc
            # print
            print("Server %s completes round %d of global model test，eval_acc：%f"
                  % (self.server_name, self.global_epoch, global_eval_acc))
            # The training and testing information was saved as excel
            excel_writer = pd.ExcelWriter('result/' + self.server_name + '_records.xlsx')
            self.train_eval_records.to_excel(excel_writer, index_label=False)
            excel_writer.close()
        return None

    def global_model_get_all_metrics(self):    # Server global model test
        with torch.no_grad():
            self.global_model.eval()
            num_classes = self.global_model.num_classes
            max_entropy = math.log(num_classes)
            # conf_P_list = []
            # lable_list = []
            index = 0
            # InD
            for test_batch_id, test_batch in enumerate(self.ind_loader):
                test_X = test_batch[0].to(self.device)
                test_Y = test_batch[1].to(self.device)
                # logits
                logits = self.global_model(test_X)
                self.ind_records.loc[index, 'LogitsSum'] = logits.abs().sum().item()
                self.ood_records.loc[index, 'LogitsSum'] = logits.abs().sum().item()
                # P
                p_softmax = F.softmax(logits, dim=1)
                # conf_P_list.append(p_softmax[0].tolist())
                # lable_list.append(test_Y[0].tolist())
                # NLL
                NLL = F.nll_loss(torch.log(p_softmax + 0.000000001), test_Y).item()
                self.ind_records.loc[index, 'NLL'] = NLL
                # Normalized_Entropy
                Entropy = -torch.sum(p_softmax * torch.log(p_softmax + 0.000000001), dim=1).item()
                Normalized_Entropy = Entropy / max_entropy
                self.ind_records.loc[index, 'Entropy'] = Normalized_Entropy
                self.ood_records.loc[index, 'Entropy'] = Normalized_Entropy
                # Predicting labels
                pred_Y = p_softmax.max(dim=1)[1]
                # conf_P
                self.ind_records.loc[index, 'lable_P'] = p_softmax[0][test_Y].item()
                self.ind_records.loc[index, 'pred_P'] = p_softmax[0][pred_Y].item()
                self.ood_records.loc[index, 'pred_P'] = p_softmax[0][pred_Y].item()
                # Brier
                Brier = 0.0
                for c in range(num_classes):
                    if c == test_Y.item():
                        Brier += (1 - p_softmax[0][c].item()) ** 2
                    else:
                        Brier += (p_softmax[0][c]).item() ** 2
                self.ind_records.loc[index, 'Brier'] = Brier
                # OOD
                self.ood_records.loc[index, 'InD'] = 1
                self.ood_records.loc[index, 'OoD'] = 0
                # True or False
                if pred_Y == test_Y:
                    self.ind_records.loc[index, 'Top1'] = 1
                    self.ind_records.loc[index, 'Wrong'] = 0
                    self.ind_records.loc[index, 'Top2'] = 0
                    self.ind_records.loc[index, 'Top3'] = 0
                    self.ind_records.loc[index, 'UCE'] = Normalized_Entropy
                else:
                    self.ind_records.loc[index, 'Top1'] = 0
                    self.ind_records.loc[index, 'Wrong'] = 1
                    self.ind_records.loc[index, 'UCE'] = 1 - Normalized_Entropy
                    # top2
                    top2_pred_Y = (-p_softmax).kthvalue(k=2, dim=1)[1]
                    if top2_pred_Y == test_Y:
                        self.ind_records.loc[index, 'Top2'] = 1
                        self.ind_records.loc[index, 'Top3'] = 0
                    else:
                        # top3
                        self.ind_records.loc[index, 'Top2'] = 0
                        top3_pred_Y = (-p_softmax).kthvalue(k=3, dim=1)[1]
                        if top3_pred_Y == test_Y:
                            self.ind_records.loc[index, 'Top3'] = 1
                        else:
                            self.ind_records.loc[index, 'Top3'] = 0
                # next one
                index += 1
            # OOD
            for test_batch_id, test_batch in enumerate(self.ood_loader):
                test_X = test_batch[0].to(self.device)
                # test_Y = test_batch[1].to(self.device)
                # logits
                logits = self.global_model(test_X)
                self.ood_records.loc[index, 'LogitsSum'] = logits.abs().sum().item()
                # P
                p_softmax = F.softmax(logits, dim=1)
                pred_Y = p_softmax.max(dim=1)[1]
                self.ood_records.loc[index, 'pred_P'] = p_softmax[0][pred_Y].item()
                # Normalized_Entropy
                Entropy = -torch.sum(p_softmax * torch.log(p_softmax + 0.00000001), dim=1).item()
                self.ood_records.loc[index, 'Entropy'] = Entropy / max_entropy
                # OOD
                self.ood_records.loc[index, 'InD'] = 0
                self.ood_records.loc[index, 'OoD'] = 1
                # next one
                index += 1
            ##############################################################################
            ##############################################################################
            ##############################################################################
            # detailed_metrics
            # predP
            self.detailed_metrics.loc[self.global_epoch, 'pred_P'] = self.ind_records['pred_P'].mean()
            self.detailed_metrics.loc[self.global_epoch, 'R_pred_P'] = (self.ind_records['pred_P'] * self.ind_records['Top1']).sum() / self.ind_records['Top1'].sum()
            self.detailed_metrics.loc[self.global_epoch, 'E_pred_P'] = (self.ind_records['pred_P'] * self.ind_records['Wrong']).sum() / self.ind_records['Wrong'].sum()
            # Logits
            self.detailed_metrics.loc[self.global_epoch, 'LogitsSum'] = self.ind_records['LogitsSum'].mean()
            self.detailed_metrics.loc[self.global_epoch, 'R_LogitsSum'] = (self.ind_records['LogitsSum'] * self.ind_records['Top1']).sum() / self.ind_records['Top1'].sum()
            self.detailed_metrics.loc[self.global_epoch, 'E_LogitsSum'] = (self.ind_records['LogitsSum'] * self.ind_records['Wrong']).sum() / self.ind_records['Wrong'].sum()
            # NLL
            self.detailed_metrics.loc[self.global_epoch, 'NLL'] = self.ind_records['NLL'].mean()
            self.detailed_metrics.loc[self.global_epoch, 'R_NLL'] = (self.ind_records['NLL'] * self.ind_records['Top1']).sum() / self.ind_records['Top1'].sum()
            self.detailed_metrics.loc[self.global_epoch, 'E_NLL'] = (self.ind_records['NLL'] * self.ind_records['Wrong']).sum() / self.ind_records['Wrong'].sum()
            # Entropy
            self.detailed_metrics.loc[self.global_epoch, 'Entropy'] = self.ind_records['Entropy'].mean()
            self.detailed_metrics.loc[self.global_epoch, 'R_Entropy'] = (self.ind_records['Entropy'] * self.ind_records['Top1']).sum() / self.ind_records['Top1'].sum()
            self.detailed_metrics.loc[self.global_epoch, 'E_Entropy'] = (self.ind_records['Entropy'] * self.ind_records['Wrong']).sum() / self.ind_records['Wrong'].sum()
            # Brier
            self.detailed_metrics.loc[self.global_epoch, 'Brier'] = self.ind_records['Brier'].mean()
            self.detailed_metrics.loc[self.global_epoch, 'R_Brier'] = (self.ind_records['Brier'] * self.ind_records['Top1']).sum() / self.ind_records['Top1'].sum()
            self.detailed_metrics.loc[self.global_epoch, 'E_Brier'] = (self.ind_records['Brier'] * self.ind_records['Wrong']).sum() / self.ind_records['Wrong'].sum()
            ##############################################################################
            ##############################################################################
            ##############################################################################
            # key_metrics
            # acc
            top1_acc = self.ind_records['Top1'].sum() / self.ind_data_quantity * 100
            self.key_metrics.loc[self.global_epoch, 'Top1_ACC'] = top1_acc
            top3_acc = (self.ind_records['Top1'].sum() + self.ind_records['Top2'].sum() + self.ind_records['Top3'].sum()) / self.ind_data_quantity * 100
            self.key_metrics.loc[self.global_epoch, 'Top3_ACC'] = top3_acc
            # ECE
            ECE_all = []
            ECE_B_00_01 = []
            ECE_B_01_02 = []
            ECE_B_02_03 = []
            ECE_B_03_04 = []
            ECE_B_04_05 = []
            ECE_B_05_06 = []
            ECE_B_06_07 = []
            ECE_B_07_08 = []
            ECE_B_08_09 = []
            ECE_B_09_10 = []
            for _, row in self.ind_records.iterrows():
                acc = row['Top1']
                conf = row['pred_P']
                if conf < 0.1:
                    ECE_B_00_01.append(acc - conf)
                elif conf >= 0.1 and conf < 0.2:
                    ECE_B_01_02.append(acc - conf)
                elif conf >= 0.2 and conf < 0.3:
                    ECE_B_02_03.append(acc - conf)
                elif conf >= 0.3 and conf < 0.4:
                    ECE_B_03_04.append(acc - conf)
                elif conf >= 0.4 and conf < 0.5:
                    ECE_B_04_05.append(acc - conf)
                elif conf >= 0.5 and conf < 0.6:
                    ECE_B_05_06.append(acc - conf)
                elif conf >= 0.6 and conf < 0.7:
                    ECE_B_06_07.append(acc - conf)
                elif conf >= 0.7 and conf < 0.8:
                    ECE_B_07_08.append(acc - conf)
                elif conf >= 0.8 and conf < 0.9:
                    ECE_B_08_09.append(acc - conf)
                elif conf >= 0.9 and conf < 1.01:
                    ECE_B_09_10.append(acc - conf)
                else:
                    print('P is wrong!')
            ECE_all.append(abs(sum(ECE_B_00_01)))
            ECE_all.append(abs(sum(ECE_B_01_02)))
            ECE_all.append(abs(sum(ECE_B_02_03)))
            ECE_all.append(abs(sum(ECE_B_03_04)))
            ECE_all.append(abs(sum(ECE_B_04_05)))
            ECE_all.append(abs(sum(ECE_B_05_06)))
            ECE_all.append(abs(sum(ECE_B_06_07)))
            ECE_all.append(abs(sum(ECE_B_07_08)))
            ECE_all.append(abs(sum(ECE_B_08_09)))
            ECE_all.append(abs(sum(ECE_B_09_10)))
            ECE = sum(ECE_all) / self.ind_data_quantity
            self.key_metrics.loc[self.global_epoch, 'ECE'] = ECE
            # UCE
            self.key_metrics.loc[self.global_epoch, 'UCE'] = self.ind_records['UCE'].mean()
            # Brier
            self.key_metrics.loc[self.global_epoch, 'Brier'] = self.ind_records['Brier'].mean()
            #############################################################
            # Failure Prediction task: Assign a score of pred_P and consider Succ as Positive.
            #############################################################
            # Failure Prediction task: Assign a score of pred_P and consider Succ as Positive : AUROC
            Failure_Prediction_predP_Succ_y_true = self.ind_records['Top1']
            Failure_Prediction_predP_Succ_y_scores = self.ind_records['pred_P']
            Failure_Prediction_predP_Succ_fpr, Failure_Prediction_predP_Succ_tpr, Failure_Prediction_predP_Succ_thresholds = util.roc_curve(
                Failure_Prediction_predP_Succ_y_true, Failure_Prediction_predP_Succ_y_scores)
            Failure_Prediction_predP_Succ_AUROC = auc(Failure_Prediction_predP_Succ_fpr, Failure_Prediction_predP_Succ_tpr)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_predP_Succ_AUROC'] = Failure_Prediction_predP_Succ_AUROC
            # Failure Prediction task: Assign a score of pred_P and consider Succ as Positive : FPR95.
            Failure_Prediction_predP_Succ_interp_func = interp1d(Failure_Prediction_predP_Succ_tpr,
                                                                 Failure_Prediction_predP_Succ_fpr, kind='linear',
                                                                 bounds_error=True)
            Failure_Prediction_predP_Succ_FPR95 = Failure_Prediction_predP_Succ_interp_func(0.95)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_predP_Succ_FPR95'] = Failure_Prediction_predP_Succ_FPR95
            # Failure Prediction task: Assign a score of pred_P and consider Succ as Positive : AUPR
            Failure_Prediction_predP_Succ_precision, Failure_Prediction_predP_Succ_recall, Failure_Prediction_predP_Succ_thresholds = util.precision_recall_curve(
                Failure_Prediction_predP_Succ_y_true,
                Failure_Prediction_predP_Succ_y_scores)
            Failure_Prediction_predP_Succ_AUPR = auc(Failure_Prediction_predP_Succ_recall,
                                                     Failure_Prediction_predP_Succ_precision)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_predP_Succ_AUPR'] = Failure_Prediction_predP_Succ_AUPR
            #############################################################
            # Failure Prediction task: Assign a score of 1-pred_P and consider Err as Positive.
            #############################################################
            # Failure Prediction task: Assign a score of 1-pred_P and consider Err as Positive : AUROC
            Failure_Prediction_predP_Err_y_true = self.ind_records['Wrong']
            Failure_Prediction_predP_Err_y_scores = 1 - self.ind_records['pred_P']
            Failure_Prediction_predP_Err_fpr, Failure_Prediction_predP_Err_tpr, Failure_Prediction_predP_Err_thresholds = util.roc_curve(
                Failure_Prediction_predP_Err_y_true, Failure_Prediction_predP_Err_y_scores)
            Failure_Prediction_predP_Err_AUROC = auc(Failure_Prediction_predP_Err_fpr, Failure_Prediction_predP_Err_tpr)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_predP_Err_AUROC'] = Failure_Prediction_predP_Err_AUROC
            # Failure Prediction task: Assign a score of 1-pred_P and consider Err as Positive : FPR95
            Failure_Prediction_predP_Err_interp_func = interp1d(Failure_Prediction_predP_Err_tpr,
                                                                 Failure_Prediction_predP_Err_fpr, kind='linear',
                                                                 bounds_error=True)
            Failure_Prediction_predP_Err_FPR95 = Failure_Prediction_predP_Err_interp_func(0.95)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_predP_Err_FPR95'] = Failure_Prediction_predP_Err_FPR95
            # Failure Prediction task: Assign a score of 1-pred_P and consider Err as Positive : AUPR
            Failure_Prediction_predP_Err_precision, Failure_Prediction_predP_Err_recall, Failure_Prediction_predP_Err_thresholds = util.precision_recall_curve(Failure_Prediction_predP_Err_y_true,
                                                                                              Failure_Prediction_predP_Err_y_scores)
            Failure_Prediction_predP_Err_AUPR = auc(Failure_Prediction_predP_Err_recall, Failure_Prediction_predP_Err_precision)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_predP_Err_AUPR'] = Failure_Prediction_predP_Err_AUPR
            #############################################################
            # The Failure Prediction task: assigns a score of 1 - Entropy and classifies Success as Positive
            #############################################################
            # The Failure Prediction task: assigns a score of 1 - Entropy and classifies Success as Positive : AUROC
            Failure_Prediction_Entropy_Succ_y_true = self.ind_records['Top1']
            Failure_Prediction_Entropy_Succ_y_scores = 1 - self.ind_records['Entropy']
            Failure_Prediction_Entropy_Succ_fpr, Failure_Prediction_Entropy_Succ_tpr, Failure_Prediction_Entropy_Succ_thresholds = util.roc_curve(
                Failure_Prediction_Entropy_Succ_y_true, Failure_Prediction_Entropy_Succ_y_scores)
            Failure_Prediction_Entropy_Succ_AUROC = auc(Failure_Prediction_Entropy_Succ_fpr,
                                                      Failure_Prediction_Entropy_Succ_tpr)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_Entropy_Succ_AUROC'] = Failure_Prediction_Entropy_Succ_AUROC
            # The Failure Prediction task: assigns a score of 1 - Entropy and classifies Success as Positive : FPR95
            Failure_Prediction_Entropy_Succ_interp_func = interp1d(Failure_Prediction_Entropy_Succ_tpr,
                                                                 Failure_Prediction_Entropy_Succ_fpr, kind='linear',
                                                                 bounds_error=True)
            Failure_Prediction_Entropy_Succ_FPR95 = Failure_Prediction_Entropy_Succ_interp_func(0.95)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_Entropy_Succ_FPR95'] = Failure_Prediction_Entropy_Succ_FPR95
            # The Failure Prediction task: assigns a score of 1 - Entropy and classifies Success as Positive : AUPR
            Failure_Prediction_Entropy_Succ_precision, Failure_Prediction_Entropy_Succ_recall, Failure_Prediction_Entropy_Succ_thresholds = util.precision_recall_curve(
                Failure_Prediction_Entropy_Succ_y_true,
                Failure_Prediction_Entropy_Succ_y_scores)
            Failure_Prediction_Entropy_Succ_AUPR = auc(Failure_Prediction_Entropy_Succ_recall,
                                                     Failure_Prediction_Entropy_Succ_precision)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_Entropy_Succ_AUPR'] = Failure_Prediction_Entropy_Succ_AUPR
            #############################################################
            # Failure Prediction task: assigns a score of Entropy and classifies Err as Positive
            #############################################################
            # Failure Prediction task: assigns a score of Entropy and classifies Err as Positive : AUROC
            Failure_Prediction_Entropy_Err_y_true = self.ind_records['Wrong']
            Failure_Prediction_Entropy_Err_y_scores = self.ind_records['Entropy']
            Failure_Prediction_Entropy_Err_fpr, Failure_Prediction_Entropy_Err_tpr, Failure_Prediction_Entropy_Err_thresholds = util.roc_curve(
                Failure_Prediction_Entropy_Err_y_true, Failure_Prediction_Entropy_Err_y_scores)
            Failure_Prediction_Entropy_Err_AUROC = auc(Failure_Prediction_Entropy_Err_fpr, Failure_Prediction_Entropy_Err_tpr)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_Entropy_Err_AUROC'] = Failure_Prediction_Entropy_Err_AUROC
            # Failure Prediction task: assigns a score of Entropy and classifies Err as Positive : FPR95
            Failure_Prediction_Entropy_Err_interp_func = interp1d(Failure_Prediction_Entropy_Err_tpr,
                                                                Failure_Prediction_Entropy_Err_fpr, kind='linear',
                                                                bounds_error=True)
            Failure_Prediction_Entropy_Err_FPR95 = Failure_Prediction_Entropy_Err_interp_func(0.95)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_Entropy_Err_FPR95'] = Failure_Prediction_Entropy_Err_FPR95
            # Failure Prediction task: assigns a score of Entropy and classifies Err as Positive : AUPR
            Failure_Prediction_Entropy_Err_precision, Failure_Prediction_Entropy_Err_recall, Failure_Prediction_Entropy_Err_thresholds = util.precision_recall_curve(
                Failure_Prediction_Entropy_Err_y_true,
                Failure_Prediction_Entropy_Err_y_scores)
            Failure_Prediction_Entropy_Err_AUPR = auc(Failure_Prediction_Entropy_Err_recall,
                                                       Failure_Prediction_Entropy_Err_precision)
            self.key_metrics.loc[self.global_epoch, 'Failure_Prediction_Entropy_Err_AUPR'] = Failure_Prediction_Entropy_Err_AUPR
            #############################################################
            # AURC
            risk = 1 - self.ind_records['lable_P']
            risk_ascending = risk.sort_values(ascending=True)
            cumulative_risk = np.cumsum(risk_ascending) / risk_ascending.sum()
            aurc = np.trapz(cumulative_risk, np.linspace(0, 1, len(risk_ascending)))
            self.key_metrics.loc[self.global_epoch, 'AURC'] = aurc
            # E-AURC
            empirical_risk = np.mean(risk)
            aurc_ideal = empirical_risk + (1 - empirical_risk) * np.log(1 - empirical_risk)
            excess_AURC = aurc - aurc_ideal
            self.key_metrics.loc[self.global_epoch, 'E_AURC'] = excess_AURC
            #############################################################
            # OOD detection task: Use pred_P as the score and consider In as the Positive
            #############################################################
            # OOD detection task: Use pred_P as the score and consider In as the Positive : AUROC
            OOD_detection_predP_In_y_true = self.ood_records['InD']
            OOD_detection_predP_In_y_scores = self.ood_records['pred_P']
            OOD_detection_predP_In_fpr, OOD_detection_predP_In_tpr, OOD_detection_predP_In_thresholds = util.roc_curve(
                OOD_detection_predP_In_y_true, OOD_detection_predP_In_y_scores)
            OOD_detection_predP_In_AUROC = auc(OOD_detection_predP_In_fpr, OOD_detection_predP_In_tpr)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_predP_In_AUROC'] = OOD_detection_predP_In_AUROC
            # OOD detection task: Use pred_P as the score and consider In as the Positive : FPR95
            OOD_detection_predP_In_interp_func = interp1d(OOD_detection_predP_In_tpr,
                                                                 OOD_detection_predP_In_fpr, kind='linear',
                                                                 bounds_error=True)
            OOD_detection_predP_In_FPR95 = OOD_detection_predP_In_interp_func(0.95)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_predP_In_FPR95'] = OOD_detection_predP_In_FPR95
            # OOD detection task: Use pred_P as the score and consider In as the Positive : AUPR
            OOD_detection_predP_In_precision, OOD_detection_predP_In_recall, OOD_detection_predP_In_thresholds = util.precision_recall_curve(
                OOD_detection_predP_In_y_true,
                OOD_detection_predP_In_y_scores)
            OOD_detection_predP_In_AUPR = auc(OOD_detection_predP_In_recall, OOD_detection_predP_In_precision)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_predP_In_AUPR'] = OOD_detection_predP_In_AUPR
            #############################################################
            # OOD detection task: Assign a score of 1-pred_P to Out and consider it as Positive
            #############################################################
            # OOD detection task: Assign a score of 1-pred_P to Out and consider it as Positive : AUROC
            OOD_detection_predP_Out_y_true = self.ood_records['OoD']
            OOD_detection_predP_Out_y_scores = 1 - self.ood_records['pred_P']
            OOD_detection_predP_Out_fpr, OOD_detection_predP_Out_tpr, OOD_detection_predP_Out_thresholds = util.roc_curve(
                OOD_detection_predP_Out_y_true, OOD_detection_predP_Out_y_scores)
            OOD_detection_predP_Out_AUROC = auc(OOD_detection_predP_Out_fpr, OOD_detection_predP_Out_tpr)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_predP_Out_AUROC'] = OOD_detection_predP_Out_AUROC
            # OOD detection task: Assign a score of 1-pred_P to Out and consider it as Positive : FPR95
            OOD_detection_predP_Out_interp_func = interp1d(OOD_detection_predP_Out_tpr,
                                                          OOD_detection_predP_Out_fpr, kind='linear',
                                                          bounds_error=True)
            OOD_detection_predP_Out_FPR95 = OOD_detection_predP_Out_interp_func(0.95)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_predP_Out_FPR95'] = OOD_detection_predP_Out_FPR95
            # OOD detection task: Assign a score of 1-pred_P to Out and consider it as Positive : AUPR
            OOD_detection_predP_Out_precision, OOD_detection_predP_Out_recall, OOD_detection_predP_Out_thresholds = util.precision_recall_curve(
                OOD_detection_predP_Out_y_true,
                OOD_detection_predP_Out_y_scores)
            OOD_detection_predP_Out_AUPR = auc(OOD_detection_predP_Out_recall, OOD_detection_predP_Out_precision)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_predP_Out_AUPR'] = OOD_detection_predP_Out_AUPR
            #############################################################
            # OOD detection task: Assign a score of 1-Entropy to In and consider it as Positive.
            #############################################################
            # OOD detection task: Assign a score of 1-Entropy to In and consider it as Positive : AUROC
            OOD_detection_Entropy_In_y_true = self.ood_records['InD']
            OOD_detection_Entropy_In_y_scores = 1 - self.ood_records['Entropy']
            OOD_detection_Entropy_In_fpr, OOD_detection_Entropy_In_tpr, OOD_detection_Entropy_In_thresholds = util.roc_curve(
                OOD_detection_Entropy_In_y_true, OOD_detection_Entropy_In_y_scores)
            OOD_detection_Entropy_In_AUROC = auc(OOD_detection_Entropy_In_fpr, OOD_detection_Entropy_In_tpr)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_Entropy_In_AUROC'] = OOD_detection_Entropy_In_AUROC
            # OOD detection task: Assign a score of 1-Entropy to In and consider it as Positive : FPR95
            OOD_detection_Entropy_In_interp_func = interp1d(OOD_detection_Entropy_In_tpr,
                                                          OOD_detection_Entropy_In_fpr, kind='linear',
                                                          bounds_error=True)
            OOD_detection_Entropy_In_FPR95 = OOD_detection_Entropy_In_interp_func(0.95)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_Entropy_In_FPR95'] = OOD_detection_Entropy_In_FPR95
            # OOD detection task: Assign a score of 1-Entropy to In and consider it as Positive : AUPR
            OOD_detection_Entropy_In_precision, OOD_detection_Entropy_In_recall, OOD_detection_Entropy_In_thresholds = util.precision_recall_curve(
                OOD_detection_Entropy_In_y_true,
                OOD_detection_Entropy_In_y_scores)
            OOD_detection_Entropy_In_AUPR = auc(OOD_detection_Entropy_In_recall, OOD_detection_Entropy_In_precision)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_Entropy_In_AUPR'] = OOD_detection_Entropy_In_AUPR
            #############################################################
            # OOD detection task: Use Entropy as the score and consider Out as Positive.
            #############################################################
            # OOD detection task: Use Entropy as the score and consider Out as Positive : AUROC
            OOD_detection_Entropy_Out_y_true = self.ood_records['OoD']
            OOD_detection_Entropy_Out_y_scores = self.ood_records['Entropy']
            OOD_detection_Entropy_Out_fpr, OOD_detection_Entropy_Out_tpr, OOD_detection_Entropy_Out_thresholds = util.roc_curve(
                OOD_detection_Entropy_Out_y_true, OOD_detection_Entropy_Out_y_scores)
            OOD_detection_Entropy_Out_AUROC = auc(OOD_detection_Entropy_Out_fpr, OOD_detection_Entropy_Out_tpr)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_Entropy_Out_AUROC'] = OOD_detection_Entropy_Out_AUROC
            # OOD detection task: Use Entropy as the score and consider Out as Positive : FPR95
            OOD_detection_Entropy_Out_interp_func = interp1d(OOD_detection_Entropy_Out_tpr,
                                                            OOD_detection_Entropy_Out_fpr, kind='linear',
                                                            bounds_error=True)
            OOD_detection_Entropy_Out_FPR95 = OOD_detection_Entropy_Out_interp_func(0.95)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_Entropy_Out_FPR95'] = OOD_detection_Entropy_Out_FPR95
            # OOD detection task: Use Entropy as the score and consider Out as Positive : AUPR
            OOD_detection_Entropy_Out_precision, OOD_detection_Entropy_Out_recall, OOD_detection_Entropy_Out_thresholds = util.precision_recall_curve(
                OOD_detection_Entropy_Out_y_true,
                OOD_detection_Entropy_Out_y_scores)
            OOD_detection_Entropy_Out_AUPR = auc(OOD_detection_Entropy_Out_recall, OOD_detection_Entropy_Out_precision)
            self.key_metrics.loc[self.global_epoch, 'OOD_detection_Entropy_Out_AUPR'] = OOD_detection_Entropy_Out_AUPR
            # saved as excel
            excel_writer = pd.ExcelWriter('result/' + self.server_name + '_InD_OOD_records_' + str(self.global_epoch) + '.xlsx')
            self.ind_records.to_excel(excel_writer, sheet_name='InD')
            self.ood_records.to_excel(excel_writer, sheet_name='OoD')
            excel_writer.close()
            excel_writer = pd.ExcelWriter('result/' + self.server_name + '_all_metrics.xlsx')
            self.key_metrics.to_excel(excel_writer, sheet_name='key_metrics')
            self.detailed_metrics.to_excel(excel_writer, sheet_name='all_metrics')
            excel_writer.close()
            print("Save all metrics！")
        return None

    def prepare_for_all_metrics(self):
        # InD dataset
        self.ind_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=1, shuffle=False)
        self.ind_data_quantity = len(self.ind_loader)
        # OOD dataset
        # transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970])
        ])
        # test set as OOD
        self.ood_dataset = SVHN(root='data/SVHN', split='test', download=True, transform=transform)
        # OOD DataLoader
        self.ood_loader = torch.utils.data.DataLoader(self.ood_dataset, batch_size=1, shuffle=False)
        self.ood_data_quantity = len(self.ood_loader)
        # The data that the global_model_get_all_metrics() needs to store
        self.ind_records = pd.DataFrame(index=list(range(self.ind_data_quantity)),
                                                     columns=['lable_P',
                                                              'pred_P',
                                                              'LogitsSum',
                                                              'Top1',
                                                              'Wrong',
                                                              'Top2',
                                                              'Top3',
                                                              'NLL',
                                                              'Entropy',
                                                              'UCE',
                                                              'Brier'],
                                                     dtype=float)
        self.ood_records = pd.DataFrame(index=list(range(self.ind_data_quantity + self.ood_data_quantity)),
                                        columns=['pred_P',
                                                 'LogitsSum',
                                                 'Entropy',
                                                 'InD',
                                                 'OoD'],
                                        dtype=float)
        self.detailed_metrics = pd.DataFrame(columns=['pred_P',
                                                      'R_pred_P',
                                                      'E_pred_P',
                                                      'LogitsSum',
                                                      'R_LogitsSum',
                                                      'E_LogitsSum',
                                                      'NLL',
                                                      'R_NLL',
                                                      'E_NLL',
                                                      'Entropy',
                                                      'R_Entropy',
                                                      'E_Entropy',
                                                      'Brier',
                                                      'R_Brier',
                                                      'E_Brier'],
                                            dtype=float)
        self.key_metrics = pd.DataFrame(columns=['Top1_ACC',
                                                 'Top3_ACC',
                                                 'ECE',
                                                 'UCE',
                                                 'Brier',
                                                 'Failure_Prediction_predP_Succ_AUROC',
                                                 'Failure_Prediction_predP_Succ_FPR95',
                                                 'Failure_Prediction_predP_Succ_AUPR',
                                                 'Failure_Prediction_predP_Err_AUROC',
                                                 'Failure_Prediction_predP_Err_FPR95',
                                                 'Failure_Prediction_predP_Err_AUPR',
                                                 'Failure_Prediction_Entropy_Succ_AUROC',
                                                 'Failure_Prediction_Entropy_Succ_FPR95',
                                                 'Failure_Prediction_Entropy_Succ_AUPR',
                                                 'Failure_Prediction_Entropy_Err_AUROC',
                                                 'Failure_Prediction_Entropy_Err_FPR95',
                                                 'Failure_Prediction_Entropy_Err_AUPR',
                                                 'AURC',
                                                 'E_AURC',
                                                 'OOD_detection_predP_In_AUROC',
                                                 'OOD_detection_predP_In_FPR95',
                                                 'OOD_detection_predP_In_AUPR',
                                                 'OOD_detection_predP_Out_AUROC',
                                                 'OOD_detection_predP_Out_FPR95',
                                                 'OOD_detection_predP_Out_AUPR',
                                                 'OOD_detection_Entropy_In_AUROC',
                                                 'OOD_detection_Entropy_In_FPR95',
                                                 'OOD_detection_Entropy_In_AUPR',
                                                 'OOD_detection_Entropy_Out_AUROC',
                                                 'OOD_detection_Entropy_Out_FPR95',
                                                 'OOD_detection_Entropy_Out_AUPR'],
                                        dtype=float)