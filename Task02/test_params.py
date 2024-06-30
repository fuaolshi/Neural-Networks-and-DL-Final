import torch
import torch.nn as nn
from copy import deepcopy
from torchvision.models import vit_b_16, vgg11
from torchvision.models import VGG11_Weights, ViT_B_16_Weights

# 加载预训练的ViT模型
class ViTB16(nn.Module):
    def __init__(self):
        super(ViTB16, self).__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 100) 

    def forward(self, x):
        x = self.vit(x)
        return x

class ViTB16_new(nn.Module):
    def __init__(self):
        super(ViTB16_new, self).__init__()
        # 加载预训练模型
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, 100)  
        
        existing_layers = self.vit.encoder.layers
        new_layers = [deepcopy(layer) for layer in existing_layers[-6:]]
        self.vit.encoder.layers.extend(new_layers)
        self.vit.encoder.num_layers = len(self.vit.encoder.layers)

    def forward(self, x):
        x = self.vit(x)
        return x

# 加载预训练的VGG11模型
class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.vgg11 = vgg11(weights=VGG11_Weights.IMAGENET1K_V1) 
        self.vgg11.classifier[-1] = nn.Linear(4096, 100)

    def forward(self, x):
        x = self.vgg11(x)
        return x
       
vitb16_new = ViTB16_new()
vitb16 = ViTB16()
vgg11 = VGG11()

init_img = torch.ones((1, 3, 224, 224))
output_vitb16 = vitb16(init_img)
output_vitb16_new = vitb16_new(init_img)

# 检查参数数量
vitb16_params = sum(p.numel() for p in vitb16.parameters() if p.requires_grad)
vgg11_params = sum(p.numel() for p in vgg11.parameters() if p.requires_grad)
vitb16_new_params = sum(p.numel() for p in vitb16_new.parameters() if p.requires_grad)

# 计算参数数量
print(f"The number of ViT-B/16 Params: {vitb16_params}")
print(f"The number of New Extend ViT-B/16 Params: {vitb16_new_params}")
print(f"The number of VGG11 Params: {vgg11_params}")