
import torch
from torch import optim, nn
from torchvision import models

import os
import sys
import csv
import time
import argparse

from utils.trainer import *
from utils.dataset import *


parser = argparse.ArgumentParser(description='Train and Evaluate MosquitoDL')
parser.add_argument('--net_type', default='resnet50', type=str,
                    help='networktype: resnet')
parser.add_argument('--data_type', default='cub200', type=str,
                    help='cifar10, cifar100, cub200, imagenet, mosquitoDL')   
parser.add_argument('--data_root', default='/home/ryan/datasets', type=str,
                    help='Path to datasets')                    
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--gpus', type=str, default='0')

parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--crop_size', type=int, default=448)

parser.add_argument('--pth_path', type=str)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset_root = args.data_root

crop_size = args.crop_size # Default
net_type = args.net_type.lower()
data_type = args.data_type.lower()
batch_size = (args.batch_size, args.batch_size)
num_workers = args.num_workers

print(f"Building Dataloaders: {data_type}")

if data_type == 'mosquitodl':
    train_loader, valid_loader, num_classes = MosquitoDL_loaders(dataset_root, crop_size, batch_size, num_workers)
elif data_type == 'ip102':
    train_loader, valid_loader, test_loader, num_classes = IP102(dataset_root, crop_size=args.crop_size, batch_size=args.batch_size, num_workers=args.num_workers)
elif 'cifar' in data_type:
    if data_type == 'cifar10':
        train_loader, valid_loader, num_classes = CIFAR_loaders(dataset_root, '10', batch_size, num_workers)
    elif data_type == 'cifar100':
        train_loader, valid_loader, num_classes = CIFAR_loaders(dataset_root, '100', batch_size, num_workers)
    else:
        assert f'Unrecognized \'{data_type}\' for CIFAR dataset.'
elif data_type == 'imagenet':
    train_loader, valid_loader, num_classes = ImageNet_loaders(dataset_root, batch_size, num_workers)
elif data_type == 'cub200':
    train_loader, valid_loader, num_classes = CUB200_loaders(dataset_root, crop_size, batch_size, num_workers)
else:
    assert f'Unsupported Dataset Type \'{data_type}\'.'

print(" - Done !")


print(f"Buliding \'{net_type}\' network...")

if data_type == 'cifar': # Using 32x32 version of resnet. (ClovaAI Implementation)
    pass
else: # Using ImageNet version of resnet.
    if net_type == 'resnet50':
        model = models.resnet50(pretrained=args.pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = nn.DataParallel(model)
    elif net_type == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=args.pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model = nn.DataParallel(model)
    elif net_type == 'vgg16':
        model = models.vgg16(pretrained=args.pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        model = nn.DataParallel(model)
    else:
        assert "Invalid 'net_type' !"


criterion = nn.CrossEntropyLoss()
model = model.to(device)
print(f"\t - Done !")

model, epoch_train_loss, epoch_train_acc = test(model, train_loader, criterion, device, None, 0)
print(f"Training Loss: {epoch_train_loss:.4f}")
print(f"Training Accuracy : {epoch_train_acc*100:.4f}%")

model, epoch_valid_loss, epoch_valid_acc = test(model, valid_loader, criterion, device, None, 0)
print(f"Validation Loss: {epoch_valid_loss:.4f}")
print(f"Validation Accuracy : {epoch_valid_acc*100:.4f}%")