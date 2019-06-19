import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu,nz,N_scales):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.prog = nn.ModuleList()
        self.prog.append(nn.ConvTranspose2d(nz, nz , 4, 1, 0, bias=False))
        self.prog.append(nn.BatchNorm2d(nz))
        self.prog.append(nn.ReLU(True))
        nf = nz
        nc = 3
        min_f = 16
        for scale in range(N_scales):
            nfo = max(min_f,int(nf/2))
            self.prog.append(nn.ConvTranspose2d(nf,nfo, 4, 2, 1, bias=False))
            self.prog.append(nn.BatchNorm2d(nfo))
            self.prog.append(nn.ReLU(True))
            nf = nfo
        
        self.to_rgb = nn.Sequential(
            nn.ConvTranspose2d(    nf,      nc, 3,padding=1, bias=False),
            nn.Tanh())
        
    def forward(self, input):
        for layer in self.prog:
            input = layer(input)
        output = self.to_rgb(input)
        return output
    

class Generator_input_gain(nn.Module):
    def __init__(self, ngpu,nz,N_scales):
        super(Generator_input_gain, self).__init__()
        self.ngpu = ngpu
        self.prog = nn.ModuleList()
        self.prog.append(nn.ConvTranspose2d(nz, nz , 4, 1, 0, bias=False))
#        self.prog.append(nn.BatchNorm2d(nz))
        self.prog.append(nn.ReLU(True))
        nf = nz
        nc = 3
        min_f = 16
        for scale in range(N_scales):
            nfo = max(min_f,int(nf/2))
            self.prog.append(nn.ConvTranspose2d(nf,nfo, 4, 2, 1, bias=False))
#            self.prog.append(nn.BatchNorm2d(nfo))
            self.prog.append(nn.ReLU(True))
            nf = nfo
        
        self.prog.append(nn.ConvTranspose2d(    nf,      nc, 3,padding=1, bias=False))
        self.prog.append(nn.Tanh())
        
    def forward(self, x,gains,biases):
        
        for layer_ind in range(int(len(self.prog)/2)):
            conv = self.prog[2*layer_ind]
            x = gains[layer_ind]*conv(x) + biases[layer_ind]
            nl = self.prog[2*layer_ind+1]  
            x = nl(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, ngpu,nz,N_scales):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        nc = 3
        min_f = 16
        self.from_rgb = nn.Sequential(
                nn.Conv2d(nc, min_f, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                )
        
        self.prog = nn.ModuleList()
        nf = min_f
        for scale in range(N_scales-1):
            nfo = min(nz,nf*2)
            self.prog.append(nn.Conv2d(nf, nfo, 4, 2, 1, bias=False))
            self.prog.append(nn.BatchNorm2d(nfo))
            self.prog.append(nn.LeakyReLU(0.2, inplace=True))
            nf = nfo
            
        # fully connected layer on 4x4 output    
        self.fc_layer = nn.Sequential(
                nn.Conv2d(nf, 1, 4, 1, 0, bias=False),
                nn.Sigmoid())
        
    def forward(self, input):
        output = self.from_rgb(input)
        for layer in self.prog:
            output = layer(output)
        output = self.fc_layer(output)
        
        return output.view(-1, 1).squeeze(1)