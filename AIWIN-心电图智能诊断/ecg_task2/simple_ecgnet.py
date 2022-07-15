
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

import torch

def conv_2d(in_planes, out_planes, stride=(1,1), size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,size), stride=stride,
                     padding=(0,(size-1)//2), bias=False)

def conv_1d(in_planes, out_planes, stride=1, size=3):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=size, stride=stride,
                     padding=(size-1)//2, bias=False)
class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, size=3, res=True):
        super(BasicBlock1d, self).__init__()
        self.conv1 = conv_1d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_1d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv_1d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        #out = self.relu(out)

        return out

class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1,1), downsample=None, size=3, res=True):
        super(BasicBlock2d, self).__init__()
        self.conv1 = conv_2d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_2d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_2d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        #out = self.relu(out)

        return out

class ECGNet_1(nn.Module):
    def __init__(self, input_channel=1, num_classes=12):#, layers=[2, 2, 2, 2, 2, 2]
        sizes = [
            [3,3,3,3,3,3],
            [5,5,5,5,3,3],
            [7,7,7,7,3,3],
                ]
        self.sizes = sizes
        layers = [
            [3,3,2,2,2,2],
            [3,2,2,2,2,2],
            [2,2,2,2,2,2]
            ]
        super(ECGNet_1, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(1,50), stride=(1,2), padding=(0,0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
        #                       bias=False)
        #print(self.conv2)
        #self.bn3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(.4)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,0))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.layers1_list = nn.ModuleList()
        self.layers2_list = nn.ModuleList()
        for i,size in enumerate(sizes):
            self.inplanes = 32
            self.layers1 = nn.Sequential()
            self.layers2 = nn.Sequential()
            self.layers1.add_module('layer{}_1_1'.format(size), self._make_layer2d(BasicBlock2d, 32, layers[i][0], stride=(1,1), size=sizes[i][0]))
            self.layers1.add_module('layer{}_1_2'.format(size), self._make_layer2d(BasicBlock2d, 32, layers[i][1], stride=(1,1), size=sizes[i][1]))
            self.inplanes *= 12
            self.layers2.add_module('layer{}_2_1'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][2], stride=2, size=sizes[i][2]))
            self.layers2.add_module('layer{}_2_2'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][3], stride=2, size=sizes[i][3]))
            self.layers2.add_module('layer{}_2_3'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][4], stride=2, size=sizes[i][4]))
            self.layers2.add_module('layer{}_2_4'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][5], stride=2, size=sizes[i][5]))
            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)
        # self.drop = nn.Dropout(p=0.2)
        # self.fc0 = nn.Linear(256*len(sizes),256)
        self.fc = nn.Linear(256, num_classes)#之前这里加了年龄等特征
    def _make_layer1d(self, block, planes, blocks, stride=2, size=3, res=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)

    def _make_layer2d(self, block, planes, blocks, stride=(1,2), size=3, res=True):
        downsample = None
        if stride != (1,1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=(1,1), padding=(0,0), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))
        return nn.Sequential(*layers)
    def forward(self, x_):
        x0 = x_.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x0 = self.conv2(x0)
        #x0 = self.bn2(x0)
        #x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x0 = self.dropout(x0)
        xs = []
        #print(self.layers1_list[i])
        x = self.layers1_list[0](x0)
        x = torch.flatten(x, start_dim=1,end_dim=2)
        x = self.layers2_list[0](x)
        x = self.avgpool(x)
        xs.append(x)
        out = torch.cat(xs, dim=2)
        out0 = out.view(out.size(0), -1)
######################################################
        out = torch.cat([out0], dim=1)#之前这里融合了年龄特征
        # out = self.dropout(out)
        # out = self.fc0(out)
        # out = self.dropout(out)
        out = self.fc(out)
        return out
class ECGNet_2(nn.Module):
    def __init__(self, input_channel=1, num_classes=12):#, layers=[2, 2, 2, 2, 2, 2]
        sizes = [
            [3,3,3,3,3,3],
            [5,5,5,5,3,3],
            [7,7,7,7,3,3],
                ]
        self.sizes = sizes
        layers = [
            [3,3,2,2,2,2],
            [3,2,2,2,2,2],
            [2,2,2,2,2,2]
            ]
        super(ECGNet_2, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(1,50), stride=(1,2), padding=(0,0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
        #                       bias=False)
        #print(self.conv2)
        #self.bn3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(.4)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,0))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.layers1_list = nn.ModuleList()
        self.layers2_list = nn.ModuleList()
        for i,size in enumerate(sizes):
            self.inplanes = 32
            self.layers1 = nn.Sequential()
            self.layers2 = nn.Sequential()
            self.layers1.add_module('layer{}_1_1'.format(size), self._make_layer2d(BasicBlock2d, 32, layers[i][0], stride=(1,1), size=sizes[i][0]))
            self.layers1.add_module('layer{}_1_2'.format(size), self._make_layer2d(BasicBlock2d, 32, layers[i][1], stride=(1,1), size=sizes[i][1]))
            self.inplanes *= 12
            self.layers2.add_module('layer{}_2_1'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][2], stride=2, size=sizes[i][2]))
            self.layers2.add_module('layer{}_2_2'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][3], stride=2, size=sizes[i][3]))
            self.layers2.add_module('layer{}_2_3'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][4], stride=2, size=sizes[i][4]))
            self.layers2.add_module('layer{}_2_4'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][5], stride=2, size=sizes[i][5]))
            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)
        # self.drop = nn.Dropout(p=0.2)
        # self.fc0 = nn.Linear(256*len(sizes),256)
        self.fc = nn.Linear(256, num_classes)#之前这里加了年龄等特征
    def _make_layer1d(self, block, planes, blocks, stride=2, size=3, res=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)

    def _make_layer2d(self, block, planes, blocks, stride=(1,2), size=3, res=True):
        downsample = None
        if stride != (1,1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=(1,1), padding=(0,0), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))
        return nn.Sequential(*layers)
    def forward(self, x_):
        x0 = x_.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x0 = self.conv2(x0)
        #x0 = self.bn2(x0)
        #x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x0 = self.dropout(x0)
        xs = []
        #print(self.layers1_list[i])
        x = self.layers1_list[1](x0)
        x = torch.flatten(x, start_dim=1,end_dim=2)
        x = self.layers2_list[1](x)
        x = self.avgpool(x)
        xs.append(x)
        out = torch.cat(xs, dim=2)
        out0 = out.view(out.size(0), -1)
######################################################
        out = torch.cat([out0], dim=1)#之前这里融合了年龄特征
        # out = self.dropout(out)
        # out = self.fc0(out)
        # out = self.dropout(out)
        out = self.fc(out)
        return out
class ECGNet_3(nn.Module):
    def __init__(self, input_channel=1, num_classes=12):#, layers=[2, 2, 2, 2, 2, 2]
        sizes = [
            [3,3,3,3,3,3],
            [5,5,5,5,3,3],
            [7,7,7,7,3,3],
                ]
        self.sizes = sizes
        layers = [
            [3,3,2,2,2,2],
            [3,2,2,2,2,2],
            [2,2,2,2,2,2]
            ]
        super(ECGNet_3, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(1,50), stride=(1,2), padding=(0,0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
        #                       bias=False)
        #print(self.conv2)
        #self.bn3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(.4)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,0))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.layers1_list = nn.ModuleList()
        self.layers2_list = nn.ModuleList()
        for i,size in enumerate(sizes):
            self.inplanes = 32
            self.layers1 = nn.Sequential()
            self.layers2 = nn.Sequential()
            self.layers1.add_module('layer{}_1_1'.format(size), self._make_layer2d(BasicBlock2d, 32, layers[i][0], stride=(1,1), size=sizes[i][0]))
            self.layers1.add_module('layer{}_1_2'.format(size), self._make_layer2d(BasicBlock2d, 32, layers[i][1], stride=(1,1), size=sizes[i][1]))
            self.inplanes *= 12
            self.layers2.add_module('layer{}_2_1'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][2], stride=2, size=sizes[i][2]))
            self.layers2.add_module('layer{}_2_2'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][3], stride=2, size=sizes[i][3]))
            self.layers2.add_module('layer{}_2_3'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][4], stride=2, size=sizes[i][4]))
            self.layers2.add_module('layer{}_2_4'.format(size), self._make_layer1d(BasicBlock1d, 256, layers[i][5], stride=2, size=sizes[i][5]))
            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)
        # self.drop = nn.Dropout(p=0.2)
        # self.fc0 = nn.Linear(256*len(sizes),256)
        self.fc = nn.Linear(256, num_classes)#之前这里加了年龄等特征
    def _make_layer1d(self, block, planes, blocks, stride=2, size=3, res=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))

        return nn.Sequential(*layers)

    def _make_layer2d(self, block, planes, blocks, stride=(1,2), size=3, res=True):
        downsample = None
        if stride != (1,1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=(1,1), padding=(0,0), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res))
        return nn.Sequential(*layers)
    def forward(self, x_):
        x0 = x_.unsqueeze(1)
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x0 = self.conv2(x0)
        #x0 = self.bn2(x0)
        #x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x0 = self.dropout(x0)
        xs = []
        #print(self.layers1_list[i])
        x = self.layers1_list[2](x0)
        x = torch.flatten(x, start_dim=1,end_dim=2)
        x = self.layers2_list[2](x)
        x = self.avgpool(x)
        xs.append(x)
        out = torch.cat(xs, dim=2)
        out0 = out.view(out.size(0), -1)
######################################################
        out = torch.cat([out0], dim=1)#之前这里融合了年龄特征
        # out = self.dropout(out)
        # out = self.fc0(out)
        # out = self.dropout(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    sentence_len = 12
    word_len = 4000
    n_class = 2
    x_ = torch.ones((32,sentence_len, word_len)).to(torch.float32)
    y_ = torch.ones((32,)).to(torch.int64).random_(n_class)
    # model = TextCNN(n_class)
    model = ECGNet_3()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    x_ = x_.to(device)
    y_ = y_.to(device)
    outputs = model(x_)
    optim.zero_grad()
    loss_value = loss(outputs, y_)
    loss_value.backward()
    optim.step()
    print(outputs.shape)