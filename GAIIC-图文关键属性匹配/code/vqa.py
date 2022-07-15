import os
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from param import args
from src.modeling import BertLayerNorm, GeLU
from vqa_model import VQAModel
from vqa_data import JsonDataset
from param import args
import json
import itertools
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')
def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r',encoding="utf-8") as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict

def match_attrval(title, attr, attr_dict):
    # 在title中匹配属性值
    attrvals = "|".join(attr_dict[attr])
    ret = re.findall(attrvals, title)
    return ret
    #return "{}{}".format(attr, ''.join(ret))

def get_data_tuple(splits, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = JsonDataset(splits)
    data_loader = DataLoader(
        dset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=None)
def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = torch.sigmoid(output) >= 0.5
    truth = target >= 0.5
    acc = pred.eq(truth).sum() / target.numel()
    return acc
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
class VQA:
    def __init__(self):
        # Datasets

        self.valid_tuple = get_data_tuple(args.valid, bs=args.batch_size, shuffle=False, drop_last=False)
        self.train_tuple = get_data_tuple(args.train, bs=args.batch_size, shuffle=True, drop_last=True)
        
        # Model
        self.model = VQAModel(1, args.model)
        # Load pre-trained weights
        if args.load_pretrained is not None:
            self.model.encoder.load(args.load_pretrained)
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = BCEFocalLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            #print("BertAdam Total Iters: %d" % t_total)
            from src.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, val_tuple):
        dset, loader, evaluator = train_tuple
        dset_val, loader_val, evaluator_val = val_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        iter_wrapper_val = (lambda x: tqdm(x, total=len(loader_val))) if args.tqdm else (lambda x: x)
        best_valid = 0.
        best_acc = 0
        if args.pretrain_model != None:
            checkpoint = torch.load(args.pretrain_model)#attr的预训练模型使用c5
            self.model.load_state_dict(checkpoint)
        for epoch in range(args.epochs):
            for i, (feats, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, target = feats.cuda(), target.cuda()
                logit = self.model(feats,sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                #loss = self.focal_loss(logit,target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                batch_score = accuracy(logit, target)
                if i % 100 == 0:
                    print('epoch {}, Step {}/{}, Loss: {},acc:{}'.format(epoch, i, len(loader), loss.item(),batch_score))
                #
            classes = [0,1]
            correct_pred = {classname: 0 for classname in classes}
            total_pred = {classname: 0 for classname in classes}
            target_list = []
            with torch.no_grad():
                for i, (feats,sent,target) in iter_wrapper_val(enumerate(loader_val)):
                    self.model.eval()
                    feats, target = feats.cuda(),target.cuda()
                    logit = self.model(feats,sent)
                    pred = torch.sigmoid(logit) >= 0.5
                    pred = pred.squeeze(1)

                    #for item in target:
                    #     target_list.append(int(item[0].detach().cpu().numpy()))
                    #print(Counter(target_list))

                    target = target.detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()
                    for label, prediction in zip(target, pred):
                        label, prediction = int(label), int(prediction)
                        if label == prediction:
                            correct_pred[classes[label]] += 1
                        total_pred[classes[label]] += 1
            average_acc = 0
            for classname, correct_count in correct_pred.items():
                acc = 100 * float(correct_count) / total_pred[classname]
                average_acc = average_acc + acc
                print("Accuracy for class {} is: {:.1f} %".format(classname,acc))
            print("各类别平均正确率:",average_acc / 2)
            if average_acc / 2 > best_acc:
                best_acc = average_acc / 2
                torch.save(self.model.state_dict(),args.save_model_name)

    def pred(self):
        model_pred = args.model
        checkpoint = torch.load("./best_" + model_pred + "_title_10fold.pth")
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        test_data = args.testdata_path
        attr_dict_file = args.attr_file_path
        attr_dict = load_attr_dict(attr_dict_file)
        rets = []

        flag = False#True预测属性False预测图文
        if flag:
            with open(test_data, 'r',encoding="utf-8") as f:
                for i, data in enumerate(tqdm(f)):
                    data = json.loads(data)
                    texts = [data['title'] if a=='图文' else match_attrval(data['title'], a, attr_dict) for a in data['query']]
                    assert len(data["query"]) == len(texts)
                    match = {}
                    item = {}
                    if len(data["query"]) == 1:
                        pass
                    else:
                        for i in range(1,len(texts)):
                            feats = torch.tensor(data["feature"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            feats = feats.cuda()

                            logit = self.model(feats,texts[i])
                            result = torch.sigmoid(logit) >= 0.5
                            result = int(result.detach().cpu().numpy())
                            match[data["query"][i]] = result
                    item["match"] = match
                    item["img_name"] = data["img_name"]
                    print(item)
                    rets.append(json.dumps(item, ensure_ascii=False)+'\n')

            with open("./testB_pred_attr_" + model_pred + ".txt", 'w') as f:
                f.writelines(rets)
        else:
            with open(test_data, 'r',encoding="utf-8") as f:
                for i, data in enumerate(tqdm(f)):
                    data = json.loads(data)
                    texts = [data['title'] if a=='图文' else match_attrval(data['title'], a, attr_dict) for a in data['query']]

                    sent = texts[0]
                    sent = re.sub("\d+年","",sent)
                    
                    #sent = convert_title(sent)
                    if 1:
                        #print(sent)
                        feats = torch.tensor(data["feature"], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        feats = feats.cuda()

                        logit = self.model(feats,[sent])
                        result = torch.sigmoid(logit) >= 0.5
                        result = int(result.detach().cpu().numpy())

                        match = {}
                        match["图文"] = result
                        item = {}
                        item["img_name"] = data["img_name"]
                        item["match"] = match
                        print(item)
                        rets.append(json.dumps(item, ensure_ascii=False)+'\n')

            with open("./testB_pred_title_" + model_pred + "_10fold.txt", 'w') as f:
                f.writelines(rets)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)
        
        
if __name__ == "__main__":
    # Build Class
    vqa = VQA()
    if args.mode == "title":
        args.max_seq_length = 31
    else:
        args.max_seq_length = 4
    print(args.max_seq_length)
    # train
    vqa.train(vqa.train_tuple, vqa.valid_tuple)
    #test
    #vqa.pred()

