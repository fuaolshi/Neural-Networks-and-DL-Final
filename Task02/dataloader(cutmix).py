import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(batch_size: int = 64):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 加载数据集
    trainset = datasets.CIFAR100(root='/FinalTerm/task02/data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='/FinalTerm/task02/data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def cutmix(data, targets, alpha=1.0):
    # 随机打乱数据和目标
    indices = torch.randperm(data.size(0)).to(data.device)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    new_data = data.clone()
    # 应用 CutMix 操作
    new_data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    # 更新 lambda 为剪切区域的相对比例
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))

    return new_data, targets, shuffled_targets, lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = (1. - lam).sqrt()
    cut_w = (W * cut_rat).int()
    cut_h = (H * cut_rat).int()

    # 保证中心点在图片范围内
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)

    return bbx1, bby1, bbx2, bby2