import os
import torch
import argparse
import torch.nn as nn
from dataloader import get_loaders
from model import VGG11, ViTB16_new, train_model

def main():
    num_epochs = 300
    learning_rate = 0.001
    momentum = 0.9
    base_dir = '/FinalTerm/task02/'
    decay = 1e-3
    step_size = 150
    model_choice = ['vgg', 'vit']
            
    train_loader, test_loader = get_loaders(batch_size=64, data_dir='/FinalTerm/task02/data')

    criterion = nn.CrossEntropyLoss()
    
    model = None
    if model_choice == "vgg":
        model = VGG11()
        parameters = [{"params": model.vgg11.parameters(), "lr": learning_rate}]    
    elif model_choice == "vit":
        model = ViTB16_new()
        parameters = [{"params": model.vit.parameters(), "lr": learning_rate}]

    # 设置 save_dir 和 logdir
    save_dir = os.path.join(base_dir, "pth", f"{model_choice}")
    logdir = os.path.join(base_dir, "Tensorboard", f"{model_choice}")

    optimizer = torch.optim.SGD(parameters, momentum=momentum, weight_decay=decay)
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, logdir, save_dir, step_size, model_choice)

if __name__ == '__main__':
    main()