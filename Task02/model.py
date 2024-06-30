import os
import time
import torch
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from collections import deque
from dataloader import cutmix
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import vit_b_16, vgg11, ViT_B_16_Weights, VGG11_Weights

class ViTB16_new(nn.Module):
    def __init__(self, pthpath: str = None):
        super(ViTB16_new, self).__init__()

        if pthpath:
            self.vit = vit_b_16(weights=None)
        else:
            # 加载预训练模型
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 100)
        
        existing_layers = self.vit.encoder.layers
        new_layers = [deepcopy(layer) for layer in existing_layers[-6:]]
        self.vit.encoder.layers.extend(new_layers)
        self.vit.encoder.num_layers = len(self.vit.encoder.layers)

        if pthpath:
            checkpoint = torch.load(pthpath)
            self.vit.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.vit(x)
        return x

class VGG11(nn.Module):
    def __init__(self, pthpath: str = None):
        super(VGG11, self).__init__() 
        if pthpath:
            self.vgg11 = vgg11(weights=None)
        else:
            self.vgg11 = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        
        in_features = self.vgg11.classifier[-1].in_features
        self.vgg11.classifier[-1] = nn.Linear(in_features, 100)  

        if pthpath:
            checkpoint = torch.load(pthpath)
            self.vgg11.load_state_dict(checkpoint)

    def forward(self, x):
        x = self.vgg11(x) 
        return x

def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, 
                criterion: nn.Module, optimizer: Optimizer, num_epochs: int = 70, 
                logdir: str ='/FinalTerm/task02/Tensorboard/1',
                save_dir: str ='/FinalTerm/task02/pth/1',
                step_size: int = 10, chioce: str = "vgg11"):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    writer = SummaryWriter(log_dir=logdir)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    best_test_acc = 0.0
    best_test_files = deque()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 开始训练计时
        train_start_time = time.time()
        data_transform_time = 0.0
        for inputs, labels in train_loader:
            data_transform = time.time()
            inputs, labels1, labels2, lam = cutmix(inputs, labels, alpha=1.0)  # 应用 CutMix
            data_transform_time += time.time() - data_transform
            inputs, labels1, labels2 = inputs.to(device), labels1.to(device), labels2.to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels1) * lam + criterion(outputs, labels2) * (1. - lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # 结束训练计时
        train_end_time = time.time()
        train_elapsed_time = train_end_time - train_start_time
        print(f'Epoch {epoch+1}/{num_epochs}, \nTrain Loss: {epoch_loss:.4f}, Data Transform Time: {data_transform_time: .2f}s, Training Time: {train_elapsed_time:.2f}s')

        # 将训练loss写入TensorBoard
        writer.add_scalar('Loss/Train Loss', epoch_loss, epoch)
        writer.add_scalar('Data Transform Time', data_transform_time, epoch)
        writer.add_scalar('Time/Train', train_elapsed_time, epoch)

        # 将当前学习率写入TensorBoard
        lr_i = 1
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            writer.add_scalar(f'Learning Rate/{lr_i}', current_lr, epoch)
            lr_i += 1

        # 验证步骤
        model.eval()
        test_loss = 0.0
        corrects = 0

        # 开始验证计时
        test_start_time = time.time()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        test_loss = test_loss / len(test_loader.dataset)
        test_acc = corrects.double() / len(test_loader.dataset)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Time: {test_elapsed_time:.2f}s')

        # 将验证loss和accuracy写入TensorBoard
        writer.add_scalar('Loss/Test Loss', test_loss, epoch)
        writer.add_scalar('Accuracy/Test Accuracy', test_acc, epoch)
        writer.add_scalar('Time/Test', test_elapsed_time, epoch)

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            file_path = f"{epoch+1}_{best_test_acc}.pth"
            sub_model = getattr(model, chioce)
            torch.save(sub_model.state_dict(), os.path.join(save_dir, file_path))
            best_test_files.append(file_path)
            if len(best_test_files) > 10:
                file_to_remove = best_test_files.popleft()
                os.remove(os.path.join(save_dir, file_to_remove))
    
    writer.flush()
    writer.close()