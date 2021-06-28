import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
'''

Given attention map F, num. of proposals N_proposals, radius r

1. Get N_proposals proposals from F
2. Randomly sample N_proposals to get the central point (top_x, top_y)
3.

'''
def generate_attentive_mask(attention_map, radius=1, num_proposals=3, allow_boundary=False):
    """
        Author: Junyoung Park (jy_park@inu.ac.kr)
        Input:
            attention_map   (Tensor) : NxWxH tensor after GAP.
            radius           (Tensor)
        Output:
            mask            (Tensor) : NxWxH tensor for masking attentive regions
            coords          (Tensor) : Normalized coordinates(cx, cy) for masked regions
    """
    N, W, H = attention_map.shape

    x = attention_map.reshape([N, W *H])

    _, indices = torch.sort(x, descending=True, dim=1)

    if num_proposals == 1:
        targets = indices[:, 0] # [N, Most Intensive]
    else:
        targets = indices[:, 0:num_proposals] # [N, Most Intensive]
        idx_proposal = torch.randint(low=0, high=num_proposals, size=(1,))[0]
        targets = targets[:, idx_proposal]

    mask = torch.ones_like(attention_map)

    for i in range(N):
        (top_x, top_y) = (targets[i]//W , targets[i]%W)

        # Get x, y constraint
        x_min, x_max = (top_x - radius, top_x + radius + 1)
        y_min, y_max = (top_y - radius, top_y + radius + 1)

        # To keep the region in square shape
        if allow_boundary == False:
            if x_min < 0:
                x_min = 0
                x_max = radius*2 + 1
            if x_max > W - 1:
                x_min = W - 1 - radius*2
                x_max = W
            if y_min < 0:
                y_min = 0
                y_max = radius*2 + 1
            if y_max > H - 1:
                y_min = H - 1 - radius*2
                y_max = H

        # For debugging
        # print(f"x: ({x_min},{x_max})")
        # print(f"y: ({y_min},{y_max})")
        
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if (x < 0 or x >= W) or (y < 0 or y >= H):
                    continue
                mask[i, x, y] = 0
        
    return mask

class Wrapper(nn.Module):
    '''
        Author: Junyoung Park (jy_park@inu.ac.kr)
    '''
    def __init__(self, model, stage_names):
        super(Wrapper, self).__init__()

        self.dict_activation = {}
        self.dict_gradients = {}
        self.forward_hook_handles = []
        self.backward_hook_handles = []

        self.net = model
        self.stage_names = stage_names
        self.num_stages = len(self.stage_names)

        def forward_hook_function(name): # Hook function for the forward pass.
            def get_class_activation(module, input, output):
                self.dict_activation[name] = output.data
            return get_class_activation

        def backward_hook_function(name): # Hook function for the backward pass. ver=1.7.1
            def get_class_gradient(module, input, output):
                self.dict_gradients[name] = output
            return get_class_gradient

        for L in self.stage_names:
            for k, v in self.net.named_modules():
                if L in k:
                    self.forward_hook_handles.append(v.register_forward_hook(forward_hook_function(L)))
                    self.backward_hook_handles.append(v.register_full_backward_hook(backward_hook_function(L)))
                    print(f"Registered forward/backward hook on \'{k}\'")
                    break

    def forward(self, x):
        self.clear_dict()
        return self.net(x)
            
    def print_current_dicts(self):
        for k, v in self.dict_activation.items():
            print("[FW] Layer:", k)
            print("[FW] Shape:", v.shape)
        for k, v in self.dict_gradients.items():
            print("[BW] Layer:", k)      
            print("[BW] Shape:", v[0].shape)

    def clear_dict(self):
        for k, v in self.dict_activation.items():
            v = None
        for k, v in self.dict_gradients.items():
            v = None

def rand_bbox(size, lam): 
    '''
        From ClovaAi
    '''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2