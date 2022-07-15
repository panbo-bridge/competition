import torch
dd = torch.load("title_train.pth")
print(len(dd["train"]))
print(len(dd["val"]))
print(len(dd["file2idx"]))
