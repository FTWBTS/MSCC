
import pandas as pd
import torch
import random, argparse
from .evaluation.metrics import get_metrics
from .utils.slidingWindows import find_length_rank
from .model_wrapper import *
from .HP_list import Optimal_Multi_algo_HP_dict

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())


if __name__ == '__main__':

    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running Experiment')
    parser.add_argument('--AD_Name', type=str, default='MSCC')
    args = parser.parse_args()
    
    # data_train = np.load("Datasets/SWAT/swat_train.npy")
    # data_test = np.load("Datasets/SWAT/swat_test.npy")
    # label = np.load("Datasets/SWAT/swat_label.npy")
    
    
    data_train = np.load("Datasets/PSM/psm_train.npy")
    data_test = np.load("Datasets/PSM/psm_test.npy")
    label = np.load("Datasets/PSM/psm_label.npy")
    
    # data_train = np.load("Datasets/MSL/MSL_train.npy")
    # data_test = np.load("Datasets/MSL/MSL_test.npy")
    # label = np.load("Datasets/MSL/MSL_test_label.npy")
    
    # data_train = np.load("Datasets/SMD/SMD_train.npy")
    # data_test = np.load("Datasets/SMD/SMD_test.npy")
    # label = np.load("Datasets/SMD/SMD_test_label.npy")
    
    
    slidingWindow = find_length_rank(data_test, rank=1)
    Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]

    if args.AD_Name in Semisupervise_AD_Pool:
        output = run_Semisupervise_AD(args.AD_Name, data_train, data_test, **Optimal_Det_HP)
    elif args.AD_Name in Unsupervise_AD_Pool:
        output = run_Unsupervise_AD(args.AD_Name, data_test, **Optimal_Det_HP)

    if isinstance(output, np.ndarray):
        evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=output > (np.mean(output)+0.1*np.std(output)))
        print('AUC-PR: ', evaluation_result['AUC-PR'], 'AUC-ROC: ',evaluation_result['AUC-ROC'])
    else:
        print('Evaluation Result: ',output)

