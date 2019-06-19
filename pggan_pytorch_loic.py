#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This work is based on the Theano/Lasagne implementation of
Progressive Growing of GANs paper from tkarras:
https://github.com/tkarras/progressive_growing_of_gans

PyTorch Model definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpscaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias):
        super(UpscaleConvBlock, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 2, padding=padding, bias=bias)
        self.opad = nn.ZeroPad2d(padding=(1,0,1,0))# same side as tf asymetric output padding
        if (kernel_size%2) is not 0:
            self._forward = self._forwardOdd
        else:
            self._forward = self._forwardEven

    def forward(self, x):
        return self._forward(x)

    def _forwardOdd(self, x):
        x = self.opad(x)
        x = self.conv(x)
        return x[:,:,1:,1:]

    def _forwardEven(self, x):
        x = self.conv(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.conv = nn.Linear(in_channels, out_channels*16, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        x = x.view(x.size(0), x.size(1))
        x = self.conv(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = x.add(self.bias)
        return x


class YYYNormActBlock(nn.Module):
    def __init__(self, mainModule):
        super(YYYNormActBlock, self).__init__()
        self.mainModule = mainModule
        self.norm = PixelNormLayer()

    def forward(self, x):
        x = self.mainModule(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.norm(x)
        return x

class ToRGB(nn.Module):
    def __init__(self, alpha, nfPred, nfCurr):
        super(ToRGB, self).__init__()
        self.alpha = alpha
        self.convPred = nn.Conv2d(nfPred,
                3,
                kernel_size=1,
                padding=0,
                bias=True)
        self.convCurr = nn.Conv2d(nfCurr,
                3,
                kernel_size=1,
                padding=0,
                bias=True)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        y = self.convCurr(x[1])*(1-self.alpha)
        y += self.convPred(self.upscale(x[0]))*self.alpha
        return y



class Generator(nn.Module):
    def __init__(self, lod_in=0):
        super(Generator, self).__init__()

        self.featNF = [512, 512, 512, 512, 256, 128, 64, 32, 16]
        self.norm = PixelNormLayer()
        self.features = nn.ModuleList([
            YYYNormActBlock(DenseBlock(512, 512)),
            YYYNormActBlock(ConvBlock(512, 512, kernel_size=3, padding=1, bias=True)),
            YYYNormActBlock(UpscaleConvBlock(512, 512, kernel_size=4, padding=1, bias=True)),
            YYYNormActBlock(ConvBlock(512, 512, kernel_size=3, padding=1, bias=True)),
            YYYNormActBlock(UpscaleConvBlock(512, 512, kernel_size=4, padding=1, bias=True)),
            YYYNormActBlock(ConvBlock(512, 512, kernel_size=3, padding=1, bias=True)),
            YYYNormActBlock(UpscaleConvBlock(512, 512, kernel_size=4, padding=1, bias=True)),
            YYYNormActBlock(ConvBlock(512, 512, kernel_size=3, padding=1, bias=True)),
            YYYNormActBlock(UpscaleConvBlock(512, 256, kernel_size=4, padding=1, bias=True)),
            YYYNormActBlock(ConvBlock(256, 256, kernel_size=3, padding=1, bias=True)),
            YYYNormActBlock(UpscaleConvBlock(256, 128, kernel_size=4, padding=1, bias=True)),
            YYYNormActBlock(ConvBlock(128, 128, kernel_size=3, padding=1, bias=True)),
            YYYNormActBlock(UpscaleConvBlock(128, 64, kernel_size=4, padding=1, bias=True)),
            YYYNormActBlock(ConvBlock(64, 64, kernel_size=3, padding=1, bias=True)),
            YYYNormActBlock(UpscaleConvBlock(64, 32, kernel_size=4, padding=1, bias=True)),
            YYYNormActBlock(ConvBlock(32, 32, kernel_size=3, padding=1, bias=True)),
            YYYNormActBlock(UpscaleConvBlock(32, 16, kernel_size=4, padding=1, bias=True)),
            YYYNormActBlock(ConvBlock(16, 16, kernel_size=3, padding=1, bias=True))
            ])
        self.setLOD(1.)

    def setLOD(self, lod_in):
        lod = int(lod_in) 
        res = 8 - lod
        self.res = res
        self.output = ToRGB(lod_in - lod, self.featNF[res-1], self.featNF[res])
        self.upscale = nn.Upsample(scale_factor=2**lod)

    def forward(self, x):
        self.fmaps = []
        x = self.norm(x)
        k = 0
        for k in range(self.res + 1):
            f, f2 = self.features[k*2:k*2+2]
            oldx = x
            x = f2(f(oldx))
            self.fmaps.append(x)
            k += 1
        
        x = self.output([oldx,x])
#        x = self.upscale(x)
        return x
