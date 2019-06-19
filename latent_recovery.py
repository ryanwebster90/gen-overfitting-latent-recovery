from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from scipy import stats # for computing pval only

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--splitAt', type=int,required=True, help='where to have validation split')
parser.add_argument('--imageSize', type=int, required=True, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, required=True, help='size of the latent z vector')
parser.add_argument('--maxImages', type=int, required=True, help='max images')
parser.add_argument('--netFolder',default='',help='path to folder containing G and D')
parser.add_argument('--hasEmbedding',type=int,default=0, help='0,1 for False, True')
parser.add_argument('--G', default='', help="path to netG")

opt = parser.parse_args()
print(opt)

#input('opts')

device = torch.device("cuda")

train_set = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
test_set = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    
    

N_images_max = opt.maxImages
batch_size = 1 #1 to optimize for each image individually
N_images = min(opt.splitAt, N_images_max)

train_set.samples = train_set.samples[:N_images]
train_set.imgs = train_set.imgs[:N_images]
test_set.samples = test_set.samples[opt.splitAt:opt.splitAt + N_images]
test_set.imgs = test_set.imgs[opt.splitAt:opt.splitAt + N_images]

dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=int(4))
dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=False, num_workers=int(4))

netG = torch.load(opt.netFolder + '/' + opt.G).eval().cuda()
#netG = nn.Sequential(netG,nn.AdaptiveAvgPool2d(opt.imageSize))
#netD = torch.load(opt.netFolder + '/' + opt.D).eval().cuda()

print(netG)
#print(netD)

base_folder = 'analysis'
has_embedding=opt.hasEmbedding
out_folder = base_folder + '/' + opt.G[:-4]
if os.path.exists(out_folder)==False:
    os.mkdir(out_folder)

z_sample = torch.randn(16,opt.nz,1,1).cuda()
x_in = [z_sample]
if has_embedding:
    x_in[0] = x_in[0].view(16,-1)
    b = torch.zeros(16,dtype=torch.long,device=device)
    x_in.append(b)
y = netG(*x_in).detach()
vutils.save_image(y,out_folder + '/random_sample.jpg',nrow=4,normalize=True)


recov_errors_test = torch.zeros(N_images)
recov_errors_train = torch.zeros(N_images)
save_images_test = torch.zeros(16,3,opt.imageSize,opt.imageSize)
save_images_train = torch.zeros(16,3,opt.imageSize,opt.imageSize)
save_targets_test = torch.zeros(16,3,opt.imageSize,opt.imageSize)
save_targets_train = torch.zeros(16,3,opt.imageSize,opt.imageSize)

N_optim_iter= 50
loss_layer = nn.MSELoss()

gen_pool_layer = nn.AdaptiveAvgPool2d(opt.imageSize)
netG = nn.Sequential(netG,gen_pool_layer)

dataloaders = [dataloader_train,dataloader_test]

for dl_ind,dataloader in enumerate(dataloaders):
    recov_errors = torch.zeros(N_images)
    save_images = torch.zeros(16,3,opt.imageSize,opt.imageSize)
    save_targets = torch.zeros(16,3,opt.imageSize,opt.imageSize)
    for batch_ind,data in enumerate(dataloader):
        
        x = torch.randn(1,opt.nz,1,1).cuda()
        optimizer = optim.LBFGS([x.requires_grad_()],max_iter=1)
            
        x_in = [x]
        # handle networks with embedding layers
        if has_embedding:
            x_in[0] = x_in[0].view(1,-1)
            b = torch.zeros(1,dtype=torch.long,device=device)
            x_in.append(b)
            
        target = data[0].cuda()
        for i in range(N_optim_iter):
            def closure():
        
                optimizer.zero_grad()
                y = netG(*x_in)
                
                loss = loss_layer(y,target)
                loss.backward()
                return loss
        
            optimizer.step(closure)
        
        y = netG(*x_in).detach()
        if batch_ind < 16:
            save_targets[batch_ind,:,:,:] = target.cpu()
            save_images[batch_ind,:,:,:] = y.cpu()
        loss = loss_layer(y,target).detach().cpu()
        recov_errors[batch_ind] = loss
        print(f'Recovered image {batch_ind}/{N_images}, loss = {loss:.04f}')
        
        
    if dl_ind==0:
        save_images_train = save_images
        save_targets_train = save_targets
        recov_errors_train = recov_errors
    else:
        save_images_test = save_images
        save_targets_test = save_targets
        recov_errors_test = recov_errors
        
vutils.save_image(save_images_train,out_folder + '/recov_train.jpg',normalize=True,nrow=4)
vutils.save_image(save_images_test,out_folder + '/recov_test.jpg',normalize=True,nrow=4)
vutils.save_image(save_targets_train,out_folder + '/targets_train.jpg',normalize=True,nrow=4)
vutils.save_image(save_targets_test,out_folder + '/targets_test.jpg',normalize=True,nrow=4)
   
torch.save(recov_errors_train,out_folder + '/recov_errors_train.pth')
torch.save(recov_errors_test,out_folder + '/recov_errors_test.pth')

all_vals = torch.cat([recov_errors_train,recov_errors_test],dim=0)

import numpy as np
import matplotlib.pyplot as plt
import torch
colors = ['green','red']
labels = ['train','test']

mi = 0
# NOTE: you may want to change this value, depending on what is considered an outlier
ma = .25

N_bins = 32
bins = np.linspace(mi,ma,N_bins)
fig, ax = plt.subplots()

ax.hist(recov_errors_train,alpha=.33,bins=bins, color=colors[0],label=labels[0])
ax.hist(recov_errors_test,alpha=.33,bins=bins, color=colors[1],label=labels[1])

ax.set_title('Recover errors for test/train') 
ax.set_ylabel('Frequency')
ax.set_xlabel('Recovery error')
ax.legend()
print('saving plot')

fig.savefig(out_folder + '/G_recov_hist.jpg',dpi=200)

# compute accuracy and save in text file

all_vals = np.array(all_vals)
ids = np.argsort(all_vals)
#take last half of array for most confident image is "real"

# take smallest recovery values to detect overfitting
ids_train = ids[:N_images]
acc = float(np.sum(ids_train<=N_images))/N_images

mre_test = recov_errors_test.median()
mre_train = recov_errors_train.median()
MRE_gap = torch.abs(mre_test - mre_train)/ mre_test
s = stats.ks_2samp(recov_errors_train,recov_errors_test)
p_val = s[1]
    
stats_file = open(out_folder + '/overfitting_statistics.txt','w')

stats_file.write(f'MRE_gap = {MRE_gap:.06f}\n')
stats_file.write(f'p_val = {p_val:.06f}\n')
stats_file.write(f'mre_test = {mre_test:.06f}\n')
stats_file.write(f'mre_train = {mre_train:.06f}\n')


#stats_file.write(f'accuracy = {acc:.06f}\n')
stats_file.close()