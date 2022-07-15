# coding=utf-8
# Copy from lxmert with modifications

import argparse
import random

import numpy as np
import torch


def get_optimizer(optim):
    # Bind the optimizer
    if optim == 'rms':
        print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamax':
        print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        print("Optimizer: sgd")
        optimizer = torch.optim.SGD
    elif 'bert' in optim:
        optimizer = 'bert'      # The bert optimizer will be bind later.
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer

def parse_args():
    parser = argparse.ArgumentParser()
    #  visualbert
    #  uniter
    #  lxmert
    parser.add_argument("--model", default='visualbert')   #原本是uniter   #--adjust
    parser.add_argument("--mode",default='title')#模式是title还是attr
    parser.add_argument("--save_model_name",default=r'../data/model_data/best_model_visualbert_title_1fold.pth')#训练模型保存的路径
    parser.add_argument("--load_model_name",default=r'../data/best_model_visualbert_title_1fold.pth')#测试加载的模型
    parser.add_argument("--output",default=r"../data/tmp_data/testA_title_visualbert_1fold.txt")#测试结果的输出路径
    parser.add_argument("--pretrain_model",default=None)#测试结果的输出路径
    

    # Data Splits
    parser.add_argument("--train", default=True)
    parser.add_argument("--valid", default=False)
    parser.add_argument("--test", default=None)

    #data path
    parser.add_argument('--attr_file_path', type=str, default=r"./code/attr_to_attrvals.json")  #属性文件路径                                 
    parser.add_argument('--train_pth_path', type=str, default=r"./code/title_train_1.pth")  #训练集测试集划分文件路径                                         
    parser.add_argument('--dataset_path', type=str, default=r"./code/title_dataset/")  #数据集路径    attr_dataset_2是使用的               
    parser.add_argument('--testdata_path', type=str, default=r"../data/contest_data/preliminary_testA.txt")  #测试集路径    

    # Training Hyper-parameters
    parser.add_argument('--batchSize', dest='batch_size', type=int, default=128) #attr 800                      
    parser.add_argument('--optim', default='bert')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)                                       
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=9595, help='random seed')#属性换随机数种子，title固定随机数种子为9595
    parser.add_argument('--max_seq_length', type=int, default=31, help='max sequence_length')

    # Debugging
    #parser.add_argument('--output', type=str, default='models/trained/')
    parser.add_argument("--fast", action='store_const', default=False, const=True)
    parser.add_argument("--tiny", action='store_const', default=False, const=True)
    parser.add_argument("--tqdm", action='store_const', default=True, const=True)

    # Model Loading
    parser.add_argument('--basebert', type=str, default=r"./code/chinese_roberta_wwm_ext_pytorch",
                        help='Load the model basebert.')#填绝对路径                             --adjust

    parser.add_argument('--load_trained', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--load_pretrained', dest='load_pretrained', type=str, default=None,
                        help='Load the pre-trained LXMERT/VisualBERT/UNITER model.')
    parser.add_argument("--fromScratch", dest='from_scratch', action='store_const', default=False, const=True,
                        help='If none of the --load_trained, --load_pretrained, is set, '
                             'the model would be trained from scratch. If --fromScratch is'
                             ' not specified, the model would load BERT-pre-trained weights by'
                             ' default. ')

    # Optimization
    parser.add_argument("--mceLoss", dest='mce_loss', action='store_const', default=False, const=True)



    # Training configuration
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument("--numWorkers", dest='num_workers', default=0)

    # Parse the arguments.
    args = parser.parse_args()

    # Bind optimizer class.
    args.optimizer = get_optimizer(args.optim)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


args = parse_args()
