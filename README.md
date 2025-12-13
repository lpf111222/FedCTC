# FedCTC

This is the source code of the paper "Client-side Training Calibration in Federated Learning: Improving Global Model under Label Skew". The paper is currently under review.

This paper addresses the issue of excessive confidence and model drift in the client-side models of federated learning under label skew from the perspective of confidence calibration. Confidence calibration is performed during the clients training phase of federated learning.  $L_p$ norm, label smoothing, and mixup are used for calibration to introduce controllable uncertainty and suppress clients drift in federated learning.
It is found that the aggregated global model can achieve higher accuracy, lower calibration error, and stronger failure prediction and OOD detection.



Each calibration method, along with vanilla FedAvg, is implemented in its own file (e.g., FL_FedCL_MixUp.py) as a class that inherits from ServerBase and ClientBase defined in server_base.py and client_base.py. The new classes only override the parts that differ from standard FL, so the framework can be extended with a few lines of code.

All methods are instantiated and evaluated in main.py.



To run the code, run the command:   python main.py -c ./conf.json

All hyperparameters are written to file conf.json, which can be rewritten to run different Settings.These hyperparameters are all detailed in the paper, Let's briefly explain what each hyperparameter means: 

(1) model_name："resnet18" represents the convolutional neural network of ResNet-18 that was used in the experiments of this paper.

(2) dataset_name："cifar10" for the cifar-10 dataset, "cifar100" for the cifar-100 dataset.

(3) num_client：the total number of FL clients. In the experiments of this paper, this parameter was set to 100, it can be modified to other positive integers.

(4) k：the number of participating clients in each round of FL training. In the experiments of this paper, this parameter was set to 20, it can be modified to an integer less than or equal to num_client.

(5) global_epochs：total number of FL global training rounds. In the experiments of this paper, this parameter was set to 2000.

(6) batch_size：the batch_size of the client training. In the experiments of this paper, this parameter was set to 50.

(7) local_epochs：the total number of clients local data training epochs for each client training. In the experiments of this paper, this parameter was set to 5.

(8) optimizer:  the optimizer used for training the model on the clients,  "SGD" is original SGD optimizer, "Adagrad"  is the adaptive AdaGrad optimizer.

(9) lr : the learning rate for clients training. In the experiments of this paper, this parameter was set to 0.01.

(10) data_distribution："LongTail" means that the distribution of data classes follows a long-tail distribution; "Dirichlet" means that the client data classes are Dirichlet distributed to sample; "IID" stands for all client data being IID(independently and identically distributed).

(11) longtail_exp：this parameter is active only when the previous argument data_distribution is set to “LongTail”; it gives the multiplicative factor between successive class sizes under the long-tailed setup. In the experiments of this paper, it is set to 0.7 for CIFAR-10 and 0.99 for CIFAR-100.

(12) dirichlet_alpha_min，dirichlet_alpha_max：hyperparameters for Dirichlet distribution of client data that adjust the degree of NonIID. See Experimental Setting for details.  It is 0-0.5 in our paper.

(13) lable_smooth_p: Probability assigned to the ground-truth class when generating the soft label in label smoothing; we use P = 0.99.

(14) mixup_alpha: the mixup hyper-parameter that sets the mixed ratio (detailed in the paper). It is fixed at 0.2 in this paper.

(15) lp: weight of the explicit Lp-norm regulariser (detailed in the paper).  we set it to 0.0001.
