源文件：https://github.com/fuaolshi/Neural-Networks-and-DL-Final

权重文件：链接：https://pan.baidu.com/s/1RSyFYYjvHge6zEzqYNUq0w?pwd=m773 
提取码：m773 

# Task01

对比监督学习和自监督学习在图像分类任务上的性能表现

基本要求：
(1) 实现任一自监督学习算法并使用该算法在自选的数据集上训练ResNet-18，随后在CIFAR-100数据集中使用Linear Classification Protocol对其性能进行评测；
(2) 将上述结果与在ImageNet数据集上采用监督学习训练得到的表征在相同的协议下进行对比，并比较二者相对于在CIFAR-100数据集上从零开始以监督学习方式进行训练所带来的提升；
(3) 尝试不同的超参数组合，探索自监督预训练数据集规模对性能的影响；

## 模型介绍

### 自监督学习阶段

- 使用 Tiny ImageNet 中的训练集作为预训练的数据集。
- 使用SimCLR自监督算法。

- 自监督预训练，模型文件为夹名为`self-sup_None_0.001_0.001_2000_64_[1000, 1500]_0.1`
  - `weight_decay=0.001`
  - `Learning_Rate = 0.001`
  - `batch_size = 64`
  - `epoch = 2000`
  - `milestone = [1000,1500]`
  - `gamma = 0.1`
- 自监督预训练结果基础上，应用下游任务
  - 微调分类层，模型文件夹名为`self1_974_4.035845334642828.pth_0.001_0.001_300_64_[150, 225]_0.1`
    - `weight_decay=0.001`
    - `Learning_Rate = 0.001`
    - `batch_size = 64`
    - `epoch = 300`
    - `milestone = [150,225]`
    - `gamma = 0.1`
  - 全局微调，模型文件夹为`self2_974_4.035845334642828.pth_0.001_0.001_300_64_[150, 225]_0.1`，参数同上

### 监督学习阶段

- 无预训练监督学习，模型文件夹名为`sup_None_0.001_0.001_300_64_[150, 225]_0.1`
  - `weight_decay=0.001`
  - `Learning_Rate = 0.001`
  - `batch_size = 64`
  - `epoch = 300`
  - `milestone = [150,225]`
  - `gamma = 0.1`
- 预训练监督学习（读取Pytorch中ResNet-18的预训练权重）
  - 微调分类层，模型文件夹名为`sup1_None_0.001_0.001_300_64_[150, 225]_0.1`，参数同上
  - 全局微调，模型文件夹为`sup2_None_0.001_0.001_300_64_[150, 225]_0.1`，参数同上

## 模型训练结果

### 自监督预训练

1. 学习率曲线

   <img src="Task01\Image\自监督预训练\Learning Rate.svg" style="zoom:25%;" />

2. Loss曲线

   <img src="Task01\Image\自监督预训练\Loss.svg" style="zoom:25%;" />

结果分析：

- **收敛性**：损失曲线应随着训练过程逐渐下降，表明模型在不断优化和学习。但是其**最后的loss仍然高达4及以上**。

- **稳定性**：如果损失曲线在训练后期趋于平稳并且波动较小，说明模型训练良好并已基本收敛。**每次学习率下降，loss便会下降，这说明模型找到了正确的训练方向。**

### 五种训练结果

1. 学习率曲线

   <img src="Task01\Image\Learning Rate.svg" style="zoom:25%;" />

2. 训练集准确率：

   <img src="Task01\Image\Train Accuracy.svg" style="zoom:25%;" />

   - 粉色曲线：自监督预训练结果基础 分类层微调
   - 绿色曲线：自监督预训练结果基础 全局微调
   - 黄色曲线：预训练监督学习（读取Pytorch中ResNet-18的预训练权重）分类层微调
   - 红色曲线：预训练监督学习（读取Pytorch中ResNet-18的预训练权重）全局微调
   - 蓝色曲线：无预训练监督学习

3. 测试集准确率：

   <img src="Task01\Image\Test Accuracy.svg" style="zoom:25%;" />

4. 训练集Loss：

   <img src="Task01\Image\Train Loss.svg" style="zoom:25%;" />

5. 测试集Loss：

   <img src="Task01\Image\Test Loss.svg" style="zoom:25%;" />

### 结果分析

1. 自监督预训练结果基础上分类层微调（准确率最低）
   - 训练和测试表现
     - 自监督预训练虽然提供了一个初始特征表示，但由于只微调分类层，其效果受到限制。
     - 模型的大部分参数保持不变，无法充分调整以适应新任务，导致收敛速度慢，最终准确率较低。
     - 这种方法在分类任务上可能不如监督预训练效果好，因为自监督预训练的特征并不是专门为分类任务优化的。
   - 可能的原因：仅微调分类层限制了模型在新任务上的适应能力。
2. 自监督预训练结果基础上全局微调
   - 训练和测试表现
     - 自监督预训练提供了一个初始特征表示，全局微调允许模型所有参数更新，使其能够更好地适应新任务。
     - 收敛速度较快，训练损失迅速下降。
     - 全局微调提升了模型在测试集上的泛化能力，测试准确率较高，测试损失较低。
   - 可能原因：全局微调可以充分利用自监督预训练的潜力，优化所有层的参数，适应新的分类任务。
3. 预训练监督学习（读取Pytorch中ResNet-18的预训练权重）分类层微调
   - 训练和测试表现
     - 监督学习的预训练权重已经在大量有标签的数据上训练过，提供了有效的初始权重。
     - 分类层微调虽然限制了模型的大部分参数更新，但由于初始权重好，收敛速度较快，但准确率并不太高。
     - 测试集上表现一般，测试准确率一般，属于是第二差。

4. 预训练监督学习（读取Pytorch中ResNet-18的预训练权重）全局微调
   - 训练和测试表现
     - 监督学习的预训练权重在有标签数据上训练过，初始权重非常好。
     - 全局微调允许所有参数更新，使模型能更快地适应新任务，训练损失迅速下降。
     - 测试集上通常能达到最高的准确率和最低的损失，因为全局微调能充分利用预训练的优势。

- 可能的原因：全局微调使模型能够更好地适应新任务，利用监督学习预训练的特征进行优化。

5. 无预训练监督学习
   - 训练和测试表现
     - 无预训练的模型从头开始学习，因此收敛速度最慢，训练损失下降最慢。
     - 初期测试准确率较低，测试损失较高。
     - 需要更多数据和训练时间才能达到较好的效果。

- 原因：从零开始训练没有任何初始特征，导致学习效率低，模型需要更多的训练时间和数据。

### 总结

- **粉色曲线**（自监督预训练 + 分类层微调）表现最差，因为仅微调分类层无法充分利用自监督预训练的特征，导致模型的适应能力和泛化能力较差。
- **蓝色曲线**（无预训练监督学习）表现次差，因为模型从头开始学习，需要更多的训练时间和数据。
- **黄色曲线**（监督学习预训练 + 分类层微调）由于监督学习预训练的特征直接适用于分类任务，微调分类层也能取得较好的效果。
- **绿色曲线**（自监督预训练 + 全局微调）通过全局微调，充分利用了自监督预训练的潜力，表现较好。
- **红色曲线**（监督学习预训练 + 全局微调）表现最佳，既利用了监督学习预训练的优势，又通过全局微调充分适应新任务。

**但事实上，自监督学习预训练的模型，不应该在提取特征方面与监督学习预训练的模型差距这么大。**分析原因主要在于，Pytorch 中的监督学习预训练模型，是用千万级别的 ImageNet 数据集进行的预训练，而此处自监督学习预训练时，采用的仅为小数据集，数据集的规模对于模型的预训练效果产生了很大的影响。

# Task02

在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型

基本要求：
（1） 分别基于CNN和Transformer架构实现具有相近参数量的图像分类网络；
（2） 在CIFAR-100数据集上采用相同的训练策略对二者进行训练，其中数据增强策略中应包含CutMix；
（3） 尝试不同的超参数组合，尽可能提升各架构在CIFAR-100上的性能以进行合理的比较。

## 模型选择及介绍

1. CNN：选择经典的VGG网络架构，尝试了使用**VGG19、16、11**三个模型分别计算参数量，最终**选定VGG11**

2. 基于Transformer：选择经典的ViT模型架构，尝试了使用**ViT-B/16**、**ViT-L/16**、**ViT-H/14**，最终选定ViT-B/16在其基础上进行拓展。

3. 以上模型均使用`test_params.py`进行测试

   ```bash
   The number of ViT-B/16 Params: 85875556
   The number of New Extend ViT-B/16 Params: 128402788
   The number of VGG11 Params: 129176036
   ```

## 训练设置

数据增强使用了CutMix的方法

```python
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
```

参数选择：batch size为64，learning rate为0.001，weight decay为0.001，optimize选择为SGD，momentum选择为0.9，150 epochs 降低学习率，总epoch为300

## 训练结果

1. 学习率曲线：两者完全一致，排除变量干扰

   !(D:\研究生课程\课程相关\研一下\神经网络\期末作业\Task02\image\VGG_Learning Rate.svg)

   <img src="Task02\image\VIT_Learning Rate.svg" alt="VIT_Learning Rate" style="zoom: 25%;" />

<img src="Task02\image\VIT_Learning Rate.svg" style="zoom:25%;" />

2. 测试集准确率（上面为VGG，下面为VIT）

   <img src="Task02\image\VGG_Test Accuracy.svg" style="zoom:25%;" />

   <img src="Task02\image\VIT_Test Accuracy.svg" alt="VIT_Test Accuracy" style="zoom:25%;" />

   总结：

   - VGG和VIT模型都展示了有效的学习能力，从它们的测试集准确率曲线上可以看出这一点。

   - 与VIT模型相比，VGG模型的测试集准确率最终稳定**远没有VIT模型高**，但是它的提升**更加稳定，波动较少**。

   - VIT模型虽然达到了**较高的准确率**，但其表现**更为不稳定**，这可能是由于Transformer架构的特性，容易对不同的数据分布或超参数变化敏感。

   - 两个模型相同的学习率曲线表明，**性能和稳定性上的差异主要是由于架构本身**，而不是训练策略的差异。

3. 训练集LOSS（上面为VGG，下面为VIT）

   <img src="Task02\image\VGG_Train Loss.svg" style="zoom:25%;" />

   <img src="Task02\image\VIT_Train Loss.svg" alt="VIT_Train Loss" style="zoom:25%;" />

   总结：

   - VGG和VIT模型的训练集损失曲线都表明它们在训练过程中**有效地减少了损失**，显示出良好的学习能力。

   - VGG模型的损失曲线**较为平滑，表明其收敛过程较为稳定**。

   - VIT模型虽然损失曲线**有更多波动**，但整体趋势仍然是下降的，并且**Loss更低**，显示出有效的学习过程。

   - 这些波动可能是由于Transformer模型的特点，导致其对训练过程中**某些变化更为敏感**

4. 测试集LOSS（上面为VGG，下面为VIT）

   <img src="Task02\image\VGG_Test Loss.svg" style="zoom:25%;" />

   <img src="Task02\image\VIT_Test Loss.svg" alt="VIT_Test Loss" style="zoom:25%;" />

## 总结

**稳定性**：

- VGG模型在训练和测试过程中表现出更稳定的损失下降，损失曲线较为**平滑**，波动较小。
- VIT模型虽然损失曲线有更多波动，但整体损失值的**下降趋势明确**，显示出**更有效的学习和优化过程**。

**敏感性**：

- VIT模型对**训练和测试数据的敏感性较高**，可能导致损失曲线的波动。这种敏感性可能与Transformer架构的特性有关。
- VGG模型在处理数据时表现出**更大的稳定性，损失曲线较少波动**。

**性能**：

- 两个模型在测试集上的准确率都显示了良好的性能，但VGG模型的**表现更加稳定**，VIT模型**最终效果更好**。
