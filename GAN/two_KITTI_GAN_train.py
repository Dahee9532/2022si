from cv2 import transform
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
import matplotlib.pylab as plt
from torchvision import transforms

import os
import numpy as np
import time
import torchvision
from torch import optim

from networks import GAN as gan
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(torch.cuda.current_device()))



import torchvision.transforms as transforms
data_transformer = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

from dataloader.dataset import KITTIDataset
data_dir = '/dahee/KITTI'
train_dataset = KITTIDataset(data_dir, data_transformer, 'training')

print("len: ", len(train_dataset))


#데이터 로더 생성

train_loader = DataLoader(train_dataset, batch_size=16, shuffle =True)

#img(x.shape), label(y.shape) 확인
for x in train_loader:
    print("img : ", x.shape)
    break

# check
from networks.GAN import Generator, Discriminator

params = {'nz': 100, 'img_size':(3,50,150)} #375, 1242
model_gen = Generator(params).to(device)
model_dis = Discriminator(params).to(device)

# 가중치 초기화 적용
model_gen.apply(gan.initialize_weights);
model_dis.apply(gan.initialize_weights);

loss_func = nn.BCELoss()

# 최적화 파라미터
lr = 2e-4
beta1 = 0.5

opt_dis = optim.Adam(model_dis.parameters(),lr=lr,betas=(beta1,0.999))
opt_gen = optim.Adam(model_gen.parameters(),lr=lr,betas=(beta1,0.999))

real_label = 1.
fake_label = 0.
nz = params['nz']
num_epochs = 500

loss_history={'gen':[], 'dis':[]}

batch_count = 0
start_time = time.time()
model_dis.train()
model_gen.train()

for epoch in range(num_epochs):
    for xb in train_loader:
        ba_si = xb.size(0)

        xb = xb.to(device)
        yb_real = torch.Tensor(ba_si,1).fill_(1.0).to(device)
        yb_fake = torch.Tensor(ba_si,1).fill_(0.0).to(device)

        # Generator
        model_gen.zero_grad()
        noise = torch.randn(ba_si,nz, device=device) # 노이즈 생성
        out_gen = model_gen(noise) # 가짜 이미지 생성
        out_dis = model_dis(out_gen) # 가짜 이미지 판별

        loss_gen = loss_func(out_dis, yb_real)
        loss_gen.backward()
        opt_gen.step()

        # Discriminator
        model_dis.zero_grad()

        out_real = model_dis(xb) # 진짜 이미지 판별
        out_fake = model_dis(out_gen.detach()) # 가짜 이미지 판별
        loss_real = loss_func(out_real, yb_real)
        loss_fake = loss_func(out_fake, yb_fake)
        loss_dis = (loss_real + loss_fake) / 2

        loss_dis.backward()
        opt_dis.step()

        loss_history['gen'].append(loss_gen.item())
        loss_history['dis'].append(loss_dis.item())

        batch_count += 1
        if batch_count % 100 == 0:
            print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, loss_gen.item(), loss_dis.item(), (time.time()-start_time)/60))
  
    #print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, loss_gen.item(), loss_dis.item(), (time.time()-start_time)/60))

loss_save = './loss_16_epo_500.png'
plt.figure(figsize=(10,5))
plt.title('Loss Progress')
plt.plot(loss_history['gen'], label='Gen. Loss')
plt.plot(loss_history['dis'], label='Dis. Loss')
plt.xlabel('batch count')
plt.ylabel('Loss')
plt.legend()
plt.savefig(loss_save)


# 가중치 저장
path2models = './models_16_epo_500/'
os.makedirs(path2models, exist_ok=True)
path2weights_gen = os.path.join(path2models, 'weights_gen.pt')
path2weights_dis = os.path.join(path2models, 'weights_dis.pt')

torch.save(model_gen.state_dict(), path2weights_gen)
torch.save(model_dis.state_dict(), path2weights_dis)

# 가중치 불러오기
weights = torch.load(path2weights_gen)
model_gen.load_state_dict(weights)

# evaluation mode
model_gen.eval()

# 가짜 이미지 생성
with torch.no_grad():
    fixed_noise = torch.randn(16, 100, device=device)
    img_fake = model_gen(fixed_noise).detach().cpu()
print(img_fake.shape)



# 가짜 이미지 시각화

plt.figure(figsize=(10,10))
for ii in range(16):
    plt.subplot(4,4,ii+1)
    plt.imshow(to_pil_image(0.5*img_fake[ii]+0.5),cmap='gray')
    plt.axis('off')
    plt.savefig('training_16_epo_500.jpg')



