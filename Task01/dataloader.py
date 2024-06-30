import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class SimCLRAugmentation:
    """
    定义SimCLR数据增强类，包含一系列图像增强操作。
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

def load_tiny_imagenet(batch_size: int = 64, data_dir: str = '/FinalTerm/task01/data/tiny-imagenet-200'):
    """
    加载Tiny ImageNet数据集并应用SimCLR数据增强。
    :param batch_size: 批量大小
    :param data_dir: 数据集目录
    :return: DataLoader对象
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")

    transform = SimCLRAugmentation()
    dataset = ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return dataloader

def load_cifar100(batch_size: int = 64, data_dir: str = '/FinalTerm/task01/data'):
    """
    加载CIFAR-100数据集并应用数据增强。
    :param batch_size: 批量大小
    :param data_dir: 数据集目录
    :return: 训练集和测试集的DataLoader对象
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录 {data_dir} 不存在")

    # 数据预处理
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
    trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader
