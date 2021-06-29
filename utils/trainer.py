import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

import os
import numpy as np
import matplotlib.pyplot as plt

from utils.trainer import *
from utils.misc import generate_attentive_box, rand_bbox

def train_HybridPartSwapping(model, train_loader, optimizer, scheduler, criterion, cur_epoch, device, **kwargs):
    """
        Author: Junyoung Park (jy_park@inu.ac.kr)

        train_HybridPartSwapping: Regularization and Augmentation 

        model(torch.nn.Module): Target model to train.
        train_loader(list(DataLoader)): Should be a list with splitted dataset.
        vervose(bool): Print detailed train/val status.
    """
    epoch_train_loss = 0
    epoch_train_acc = 0

    cut_prob = kwargs['cut_prob']
    save_path = kwargs['save_path']
    radius = kwargs['radius']
    num_proposals = kwargs['num_proposals']
    model.train()

    train_loss = 0
    train_n_corrects = 0
    train_n_samples = 0

    for idx, data in enumerate(train_loader):

        batch, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()

        pred = model(batch)
        pred_max = torch.argmax(pred, 1)

        target_stage_name = 'None'

        r = np.random.rand(1)[0]

        param_info = ''
        if r < cut_prob:
            radius = torch.randint(low=0, high=radius+1,size=[1])[0]
            target_stage_index = model.num_stages - 1

            target_stage_name = model.stage_names[target_stage_index]

            loss = criterion(pred, labels)
            
            loss.backward()

            target_fmap = model.dict_activation[target_stage_name]
            target_gradients = model.dict_gradients[target_stage_name][0]
            optimizer.zero_grad()
            model.clear_dict()

            N, C, W_f, H_f = target_fmap.shape

            importance_weights = F.adaptive_avg_pool2d(target_gradients, 1) # [N x C x 1 x 1]

            class_activation_map = torch.mul(target_fmap, importance_weights).sum(dim=1, keepdim=True) # [N x 1 x W_f x H_f]
            class_activation_map = F.relu(class_activation_map).squeeze(dim=1) # [N x W_f x H_f]

            # Get Image A mask
            # Get Image B mask
            # Overwrite target region on A to target of B

            attention_box = generate_attentive_box(class_activation_map, radius=radius, num_proposals=num_proposals, allow_boundary=False) # [N, W, H]
            attention_box = (attention_box/W_f)*batch.shape[2] # Scaling attention box from the feature size to the original image size.
            # print(attention_box)
            # exit()

            param_info = f', Radius: {radius}, Num. Proposals: {num_proposals}'

            rand_index = torch.randperm(batch.size()[0]).cuda()
            batch_original = batch.clone().detach()
            
            for batch_idx in range(batch.shape[0]):
                target_idx = rand_index[batch_idx]
                # print(target_idx)
                x_min_a, x_max_a, y_min_a, y_max_a = attention_box[batch_idx].int()
                x_min_b, x_max_b, y_min_b, y_max_b = attention_box[target_idx].int()
                # print(f'Image A({batch_idx}): ({x_min_a},{y_min_a}), ({x_max_a},{y_max_a})')
                # print(f'Image B({target_idx}): ({x_min_b},{y_min_b}), ({x_max_b},{y_max_b})\n')

                
                batch[batch_idx, :, x_min_a:x_max_a, y_min_a:y_max_a] = batch_original[target_idx, :, x_min_b:x_max_b, y_min_b:y_max_b]
                
            n_mix = (radius + 1) ** 2 # Number of zeros in attention_mask
            mix_ratio = n_mix/(W_f*H_f) # Ratio of image_b
        
            target_a = labels
            target_b = labels[rand_index]

            pred = model(batch)
            pred_max = torch.argmax(pred, 1)
            # print(1-mix_ratio)
            # print(mix_ratio)

            if target_stage_index < 3:          # Fine grained features
                loss = criterion(pred, labels)  # - No label mixing
            else:
                loss = criterion(pred, target_a) * (1 - mix_ratio) + criterion(pred, target_b) * (mix_ratio)

        else:
            loss = criterion(pred, labels)

        if idx%150 == 0 and cur_epoch % 10 == 0:
            input_ex = make_grid(batch.detach().cpu(), normalize=True, nrow=8, padding=2).permute([1,2,0])
            fig, ax = plt.subplots(1,1,figsize=(8*2,2*(batch.size(0)//8)+1))
            ax.imshow(input_ex)
            ax.set_title(f"Train Original Batch Examples\nCut_Prob:{cut_prob}, Cur_Target: {target_stage_name}, {param_info}")
            ax.axis('off')
            fig.savefig(os.path.join(save_path, f"Train_Orig_BatchSample_E{cur_epoch}_I{idx}.png"))
            plt.draw()
            plt.clf()
            plt.close("all")
            
            
        train_loss += loss.detach().cpu().numpy()
        train_n_samples += labels.size(0)
        train_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()

    epoch_train_loss = train_loss / len(train_loader)
    epoch_train_acc = train_n_corrects/train_n_samples

    return model, epoch_train_loss, epoch_train_acc

def train(model, train_loader, optimizer, scheduler, criterion, cur_epoch, device, **kwargs):
    """
        train - Training code with vanilla method

        model(torch.nn.Module): Target model to train.
        train_loader(list(DataLoader)): Should be a list with splitted dataset.
        vervose(bool): Print detailed train/val status.
    """

    save_path = kwargs['save_path']

    model.train()

    train_loss = 0
    train_n_corrects = 0
    train_n_samples = 0

    for idx, data in enumerate(train_loader):

        batch, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()

        pred = model(batch)
        pred_max = torch.argmax(pred, 1)

        loss = criterion(pred, labels)

        if idx%100 == 0 and cur_epoch % 20 == 0:
            input_ex = make_grid(batch.detach().cpu(), normalize=True, nrow=8, padding=2).permute([1,2,0])
            fig, ax = plt.subplots(1,1,figsize=(8,(batch.size(0)//8)+1))
            ax.imshow(input_ex)
            ax.set_title(f"Train Batch Examples")
            ax.axis('off')
            fig.savefig(os.path.join(save_path, f"Train_BatchSample_E{cur_epoch}_I{idx}.png"))
            plt.draw()
            plt.clf()
            plt.close("all")
            
        train_loss += loss.detach().cpu().numpy()
        train_n_samples += labels.size(0)
        train_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()

        loss.backward()
        optimizer.step()
   
    scheduler.step()

    epoch_train_loss = train_loss / len(train_loader)
    epoch_train_acc = train_n_corrects/train_n_samples

    return model, epoch_train_loss, epoch_train_acc

def train_CutMix(model, train_loader, optimizer, scheduler, criterion, cur_epoch, device, **kwargs):
    """
        train - Training code with CutMix method (Original: ClovaAI)

        model(torch.nn.Module): Target model to train.
        train_loader(list(DataLoader)): Should be a list with splitted dataset.
        vervose(bool): Print detailed train/val status.
    """
    epoch_train_loss = 0
    epoch_train_acc = 0

    save_path = kwargs['save_path']
    cut_prob = kwargs['cut_prob']
    beta = 1 # In CutMix, they use Uniform(0, 1) distribution where Beta(1, 1).

    model.train()

    train_loss = 0
    train_n_corrects = 0
    train_n_samples = 0

    for idx, data in enumerate(train_loader):

        batch, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()

        r = np.random.rand(1)

        if r < cut_prob:
            # generate mixed sample
            lam = np.random.beta(1, 1)
            rand_index = torch.randperm(batch.size()[0]).cuda()
            labels_a = labels
            labels_b = labels[rand_index]
            
            bbx1, bby1, bbx2, bby2 = rand_bbox(batch.size(), lam)
            batch[:, :, bbx1:bbx2, bby1:bby2] = batch[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2]))
            
            # compute output
            pred = model(batch)
            loss = criterion(pred, labels_a) * lam + criterion(pred, labels_b) * (1. - lam)
        else:
            # compute output
            pred = model(batch)
            loss = criterion(pred, labels)

        pred_max = torch.argmax(pred, 1)
        
        if idx%100 == 0 and cur_epoch % 20 == 0:
            input_ex = make_grid(batch.detach().cpu(), normalize=True, nrow=8, padding=2).permute([1,2,0])
            fig, ax = plt.subplots(1,1,figsize=(8,(batch.size(0)//8)+1))
            ax.imshow(input_ex)
            ax.set_title(f"TrainVal CutMix Batch Examples")
            ax.axis('off')
            fig.savefig(os.path.join(save_path, f"TrainVal_BatchSample_E{cur_epoch}_I{idx}.png"))
            plt.draw()
            plt.clf()
            plt.close("all")

        loss.backward()
        optimizer.step()
            
        train_loss += loss.detach().cpu().numpy()
        train_n_samples += labels.size(0)
        train_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()
    
    scheduler.step()

    epoch_train_loss = train_loss / len(train_loader)
    epoch_train_acc = train_n_corrects/train_n_samples

    return model, epoch_train_loss, epoch_train_acc

def test(model, test_loader, criterion, device, save_path, cur_epoch):

    test_loss = 0
    test_acc = 0

    test_n_samples = 0
    test_n_corrects = 0

    model.eval()

    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            batch, labels = data[0].to(device), data[1].to(device)

            pred = model(batch)
            pred_max = torch.argmax(pred, 1)

            loss = criterion(pred, labels)

            test_loss += loss.detach().cpu().numpy()
            test_n_samples += labels.size(0)
            test_n_corrects += torch.sum(pred_max == labels).detach().cpu().numpy()

            if save_path != None:
                if idx%300 == 0 and cur_epoch % 20 == 0:
                    input_ex = make_grid(batch.detach().cpu(), normalize=True, nrow=8, padding=2).permute([1,2,0])
                    fig, ax = plt.subplots(1,1,figsize=(8,(batch.size(0)//8)+1))
                    ax.imshow(input_ex)
                    ax.set_title(f"Testing Batch Examples")
                    ax.axis('off')
                
                    fig.savefig(os.path.join(save_path, f"Test_BatchSample_E{cur_epoch}_I{idx}.png"))
                    plt.draw()
                    plt.clf()
                    plt.close("all")

    test_loss /= len(test_loader)
    test_acc = test_n_corrects/test_n_samples

    return model, test_loss, test_acc