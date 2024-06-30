import os
import torch
import torch.nn as nn
from dataloader import Tiny_ImageNet, CIFAR100
from model import load_model, NTXentLoss, self_supervised_train, supervised_train

def main():
    data_dir = '/FinalTerm/task01/data/tiny-imagenet-200'
    batch_size = 64
    num_epochs = 70
    learning_rate = 0.001
    momentum = 0.9
    pthpath = None  # 如果有保存的模型路径，填写路径
    base_dir = '/FinalTerm/task01/'
    decay = 1e-3
    milestones = []  # 按需填写学习率下降的epoch列表
    gamma = 0.1
    strategy = ['self-sup', "sup", 'self1', "self2", "sup1", "sup2"]
    optimizer = 'SGD'

    if len(milestones) == 0:
        milestones = [int(num_epochs * 0.5), int(num_epochs * 0.75)]
    else:
        milestones = [int(x) for x in "".join(milestones[1:-1]).split(',')]

    if len(milestones) > 0:
        for milestone in milestones:
            if milestone > num_epochs:
                raise ValueError("Milestone epoch cannot be greater than the total number of epochs.")
    
    # 如果是自监督学习
    if strategy == "self-sup":       
        train_loader = Tiny_ImageNet(batch_size=batch_size, data_dir=data_dir)
        model = load_model(self_supervised=True)
        parameters = [{"params": model.parameters(), "lr": learning_rate}]
        criterion = NTXentLoss()

    # 如果是自监督学习的评估
    elif strategy in ["self1", "self2"]:
        train_loader, test_loader = CIFAR100(batch_size=batch_size, data_dir=data_dir)
        model = load_model(self_supervised=True, linear_protocal=True, pthpath=pthpath) 
        if strategy == "sl1":
            # 冻结模型除分类层以外的所有层的参数
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False 
            parameters = [{'params': model.fc.parameters(), 'lr': learning_rate}]  
        else:
            parameters = [
            {"params": model.fc.parameters(), "lr": learning_rate},
            {"params": [param for name, param in model.named_parameters() if "fc" not in name], "lr": learning_rate*0.1}
            ]
        criterion = nn.CrossEntropyLoss()   

    # 如果是从零开始的监督学习
    elif strategy == "sup":
        train_loader, test_loader = CIFAR100(batch_size=batch_size, data_dir=data_dir)
        model = load_model(supervised=True)
        parameters = [{"params": model.parameters(), "lr": learning_rate}]
        criterion = nn.CrossEntropyLoss()
    elif strategy in ["sup1", "sup2"]:
        train_loader, test_loader = CIFAR100(batch_size=batch_size, data_dir=data_dir)
        model = load_model(pretrained=True)
        if strategy == "pl1":
            # 冻结模型除分类层以外的所有层的参数
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False 
            parameters = [{"params": model.fc.parameters(), "lr": learning_rate}]
        else:
            parameters = [
            {"params": model.fc.parameters(), "lr": learning_rate},
            {"params": [param for name, param in model.named_parameters() if "fc" not in name], "lr": learning_rate*0.1}
            ]
        criterion = nn.CrossEntropyLoss()

    # 构造目录名称
    modelpth = os.path.basename(pthpath) if pthpath else None
    directory_name = f"{strategy}_{modelpth}_{decay}_{learning_rate}_{num_epochs}_{batch_size}_{milestones}_{gamma}"
    
    # 设置 save_dir 和 logdir
    save_dir = os.path.join(base_dir, "modelpth", directory_name)
    logdir = os.path.join(base_dir, "tensorboard", directory_name)

    optimizer = torch.optim.SGD(parameters, momentum=momentum, weight_decay=decay)

    if strategy == "self-sup":
        self_supervised_train(model, train_loader, optimizer, criterion, num_epochs, gamma, logdir, save_dir)
    else:
        supervised_train(model, train_loader, test_loader, optimizer, criterion, num_epochs, logdir, save_dir, milestones, gamma)
    
if __name__ == '__main__':
    main()