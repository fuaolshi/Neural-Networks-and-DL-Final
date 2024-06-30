import os
import time
import torch
from torch import nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
from torch.optim import Optimizer
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter

def load_model(self_supervised=False, projection_dim=128, pretrained=False, 
               linear_protocal=False, supervised=False, test=False, pthpath=None):
    # 使用Tiny ImageNet数据集进行SimCLR自监督学习
    if self_supervised:
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        # 加入SimCLR的projection head
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        # Linear Protocol
        if linear_protocal:
            if not pthpath:
                raise ValueError('Please provide the path to the checkpoint.')
            model.load_state_dict(torch.load(pthpath))
            model.fc = nn.Linear(in_features, 100)
    # 使用ImageNet预训练模型进行相同的Linear Protocol
    elif pretrained:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 100)
    # 使用CIFAR-100从零进行监督学习
    elif supervised:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 100)
    # 完成后进行模型测试
    elif test:
        if not pthpath:
            raise ValueError('Please provide the path to the checkpoint.')
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 100)
        model.load_state_dict(torch.load(pthpath))
    else:
        raise ValueError('Invalid model type. Please specify one of self_supervised, pretrained, supervised.') 
    return model

# 定义损失函数
class NTXentLoss(nn.Module):
    def __init__(self):
        super(NTXentLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, out1, out2):
        batch_size = out1.shape[0]
        out1 = F.normalize(out1, p=2, dim=-1)
        out2 = F.normalize(out2, p=2, dim=-1)

        # 计算相似度矩阵
        logits_aa = torch.matmul(out1, out1.T)
        logits_bb = torch.matmul(out2, out2.T)
        logits_ab = torch.matmul(out1, out2.T)
        logits_ba = torch.matmul(out2, out1.T)    

        def calculate_loss(logits_main, logits_other, logits_self):
            diag_main = logits_main.diag().unsqueeze(1)
            off_diag_main = logits_main.masked_fill(torch.eye(batch_size, device=logits_main.device).bool(), float('-inf'))
            off_diag_self = logits_self.masked_fill(torch.eye(batch_size, device=logits_self.device).bool(), float('-inf'))

            combined_logits = torch.cat([diag_main, off_diag_main, off_diag_self], dim=1)
            softmax_results = F.softmax(combined_logits, dim=1)
            softmax_diag = softmax_results[:, 0]
            log_softmax_diag = -torch.log(softmax_diag)

            return log_softmax_diag

        log_softmax_diag1 = calculate_loss(logits_ab, logits_ab, logits_aa)
        log_softmax_diag2 = calculate_loss(logits_ba, logits_ba, logits_bb)

        loss = (log_softmax_diag1.sum() + log_softmax_diag2.sum()) / (2 * batch_size)

        return loss


# 自监督学习训练函数
def self_supervised_train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, 
                          criterion: nn.Module, epochs: int = 70, gamma: float = 0.1,
                          logdir: str ='/FinalTerm/task01/Tensorboard/1',
                          save_dir: str ='/FinalTerm/task01/pth/1'):
    
    divided = epochs // 10
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    writer = SummaryWriter(log_dir=logdir)

    # 添加模型图
    init_img = torch.zeros((1, 3, 224, 224)).to(device) 
    writer.add_graph(model, init_img)

    history_loss = deque() 
    lowest_loss = float("inf")
    lowest_TrainLoss_files = deque() # 记录历史最低训练loss对应的模型文件，用于删除模型文件
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        train_start_time = time.time()
        for (images1, images2), _ in data_loader:
            images1 = images1.to(device)
            images2 = images2.to(device)
            optimizer.zero_grad()

            features1 = model(images1)
            features2 = model(images2)

            loss = criterion(features1, features2) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader)
        if isinstance(optimizer, optim.SGD):
            history_loss.append(epoch_loss)

        # 结束训练计时
        train_end_time = time.time()
        train_elapsed_time = train_end_time - train_start_time
        print(f'Epoch {epoch+1}/{epochs}, \nTrain Loss: {epoch_loss:.4f}, Training Time: {train_elapsed_time:.2f}s')

        # 将训练loss写入TensorBoard
        writer.add_scalar('Loss/Train Loss', epoch_loss, epoch)
        writer.add_scalar('Time/Train', train_elapsed_time, epoch)

        # 将当前学习率写入TensorBoard
        lr_i = 1
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            writer.add_scalar(f'Learning Rate/{lr_i}', current_lr, epoch)
            lr_i += 1

        # 保存loss最低的模型
        if epoch_loss < lowest_loss:
            # 更新最小的loss
            lowest_loss = epoch_loss
            # 与按epoch等分进行区分开
            if (epoch+1) % divided != 0:
                file_path = f"{epoch+1}_{epoch_loss}.pth"
                torch.save(model.state_dict(), os.path.join(save_dir, file_path))
                lowest_TrainLoss_files.append(file_path)
                if len(lowest_TrainLoss_files) > 10:
                    file_to_remove = lowest_TrainLoss_files.popleft()
                    os.remove(os.path.join(save_dir, file_to_remove))
        # 把epoch分成十等分，按照epoch进行保存模型，提供更多的模型选择
        if (epoch+1) % divided == 0:
            file_path = f"{epoch+1}_{epoch_loss}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, file_path))

        if isinstance(optimizer, optim.SGD):
            if len(history_loss) == 150:
                max_acc = max(history_loss)
                min_acc = min(history_loss)
                
                # 检测准确率变化是否小于阈值
                if max_acc - min_acc < 0.005:
                    # 减少学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= gamma
                    # 清空历史记录以重新开始收集数据
                    history_loss.clear()
                else:
                    # 移除最旧的记录，继续收集直到deque满
                    history_loss.popleft()

    writer.flush()
    writer.close()

# 监督学习训练函数
def supervised_train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                     optimizer: Optimizer, criterion: nn.Module, epochs: int = 70, 
                     logdir: str ='/FinalTerm/task01/Tensorboard/1',
                     save_dir: str ='/FinalTerm/task01/pth/1',
                     milestones: list = [], gamma: float = 0.1):
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    writer = SummaryWriter(log_dir=logdir)

    # 添加模型图
    init_img = torch.zeros((1, 3, 224, 224)).to(device)
    writer.add_graph(model, init_img)

    history_accuracy = deque() 
    best_test_acc = 0.0
    best_test_files = deque()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        if epoch+1 in milestones:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma

        train_start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects.double() / len(train_loader.dataset)
        history_accuracy.append(epoch_acc)

        # 结束训练计时
        train_end_time = time.time()
        train_elapsed_time = train_end_time - train_start_time
        print(f'Epoch {epoch+1}/{epochs}, \nTrain Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}, Training Time: {train_elapsed_time:.2f}s')

        # 将训练loss写入TensorBoard
        writer.add_scalar('Loss/Train Loss', epoch_loss, epoch)
        writer.add_scalar('Time/Train', train_elapsed_time, epoch)
        writer.add_scalar('Accuracy/Train Accuracy', epoch_acc, epoch)

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
        writer.add_scalar('Time/Test', test_elapsed_time, epoch)
        writer.add_scalar('Accuracy/Test Accuracy', test_acc, epoch)

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            file_path = f"{epoch+1}_{best_test_acc}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, file_path))
            best_test_files.append(file_path)
            if len(best_test_files) > 10:
                file_to_remove = best_test_files.popleft()
                os.remove(os.path.join(save_dir, file_to_remove))
    
    writer.flush()
    writer.close()