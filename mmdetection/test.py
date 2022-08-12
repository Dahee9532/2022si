import torch

checkpoint = torch.load('../moco_v2_800ep_pretrain.pth.tar')

print(checkpoint['state_dict'].key)