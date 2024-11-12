
import math
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.io import read_image

from GenVanillaNN import *
from Skeleton import Skeleton
from VideoReader import VideoReader
from VideoSkeleton import VideoSkeleton


class Discriminator(nn.Module):
    def __init__(self, num_gpu=0):
        super().__init__()
        self.num_gpu = num_gpu
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.ConvTranspose2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        """Forward pass for the Discriminator."""
        print("Discriminator: x=", x.shape)
        return self.model(x)
    



class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename)

    def train(self, n_epochs=200):
        criterion = torch.nn.MSELoss()
        
        # Optimizers for Generator and Discriminator
        optimizerG = torch.optim.SGD(self.netG.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=1e-5)

        for epoch in range(n_epochs):
            running_loss_G = 0.0
            running_loss_D = 0.0
            
            for i, data in enumerate(self.dataloader, 0):
                ske, image = data  # Assuming labels aren't necessary
                noise = torch.randn(ske.size(0), 26, 1, 1)
                generate_image = self.netG(noise)

                # Train Discriminator
                real_output = self.netD(image)
                fake_output = self.netD(generate_image.detach())
                
                real_loss = criterion(real_output, torch.ones_like(real_output))
                fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
                discrimatr_loss = real_loss + fake_loss
                
                optimizerD.zero_grad()
                discrimatr_loss.backward()
                optimizerD.step()
                
                noise = torch.randn(ske.size(0), 26, 1, 1)
                generate_image = self.netG(noise)
                fake_output = self.netD(generate_image.detach())
                
                # Train Generator
                optimizerG.zero_grad()
                generator_loss = criterion(fake_output, torch.ones_like(fake_output))
                
                generator_loss.backward()
                optimizerG.step()

                # Accumulate running loss
                running_loss_D += discrimatr_loss.item()
                running_loss_G += generator_loss.item()
                
                if i % 2000 == 1999:  # Print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] '
                        f'D Loss: {running_loss_D / 2000:.3f} | G Loss: {running_loss_G / 2000:.3f}')
                    running_loss_D = 0.0
                    running_loss_G = 0.0


            print('Finished Training')




    def generate(self, ske):           # TP-TODO
        """ generator of image from skeleton """
        ske_t = torch.from_numpy( ske.__array__(reduced=True).flatten() )
        ske_t = ske_t.to(torch.float32)
        ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1) # ske.reshape(1,Skeleton.full_dim,1,1)
        normalized_output = self.netG(ske_t)
        res = self.dataset.tensor2image(normalized_output[0])
        return res




if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(4) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

