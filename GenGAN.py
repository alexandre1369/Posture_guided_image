
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
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass for the Discriminator."""
        #print("Discriminator: x=", x.shape)
        x = self.model(x)
        return x.view(-1, 1).squeeze(1)




class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        self.optSkeOrImage = optSkeOrImage
        image_size = 64
        if optSkeOrImage == 1:
            self.netG = GenNNSkeToImage()
            src_transform = None
        else:
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([SkeToImageTransform(image_size),
                                                transforms.ToTensor()])
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)       
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=64, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG.load_state_dict(torch.load(self.filename, weights_only=True))
        
        self.netG.to(device)

    def train(self, n_epochs=200):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Déplacer les modèles sur le bon device
        self.netG = self.netG.to(device)
        self.netD = self.netD.to(device)

        criterion = torch.nn.BCELoss().to(device)  # Déplacer aussi les critères de perte

        # Optimizers for Generator and Discriminator
        real_label = 1.
        fake_label = 0.

        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.001,betas=(0.5, 0.999))
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.001, betas=(0.5, 0.999))

        for epoch in range(n_epochs):
            running_loss_G = 0.0
            running_loss_D = 0.0

            for i, data in enumerate(self.dataloader, 0):
                # Déplacer les données sur le bon device
                ske, image = data
                ske = ske.to(device)
                image = image.to(device)

                self.netD.zero_grad()
                label = torch.full((image.size(0),), real_label, dtype=torch.float, device=device)
                real_output = self.netD(image)
                real_loss = criterion(real_output, label)
                real_loss.backward()

                if self.optSkeOrImage == 1 :
                    noise = torch.randn(image.size(0), 26, 1, 1, device=device)
                    # noise if using GenNNSkeToImage 
                    generate_image = self.netG(noise)
                else:
                    generate_image = self.netG(ske)

                # Train Discriminator

                label.fill_(fake_label)
                fake_output = self.netD(generate_image.detach())
                fake_loss = criterion(fake_output, label)
                fake_loss.backward()

                # print("Fake output ", fake_output)
                # print("real_output ones_like ", torch.ones_like(real_output, device=device))
                discrimatr_loss = real_loss + fake_loss
                optimizerD.step()

                self.netG.zero_grad()
                label.fill_(real_label)

                fake_output = self.netD(generate_image)

                generator_loss = criterion(fake_output, label)
                # print("real loss", real_loss)
                # print("fake loss", fake_loss)
                # print("discrimatr_loss", discrimatr_loss.item())

                # Train Generator
                generator_loss.backward()

                optimizerG.step()

                # Accumulate running loss
                running_loss_D += discrimatr_loss.item()
                running_loss_G += generator_loss.item()

                print(f'[{epoch + 1}, {i + 1:5d}] '
                    f'D Loss: {running_loss_D} | G Loss: {running_loss_G}')
                running_loss_D = 0.0
                running_loss_G = 0.0

                # if(epoch % 10 == 0):
                #     # nouvelle_taille = (256, 256)
                #     # img = cv2.resize(generate_image, nouvelle_taille)
                #     # cv2.imshow(f'Generated image {epoch}', generate_image)
                #     torch.save(self.netG.state_dict(), f'data/Dance/DanceGenGAN{epoch}.pth')

            print(f'Epoch {epoch + 1}/{n_epochs} finished')

        print('Finished Training')
        torch.save(self.netG.state_dict(), 'data/Dance/DanceGenGAN.pth')


    def generate(self, ske):
        """ Generator of image from skeleton """
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.optSkeOrImage == 1:
            ske_t = torch.from_numpy(ske.__array__(reduced=True).flatten()).to(device)
            ske_t = ske_t.to(torch.float32)
            ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1)
            
            normalized_output = self.netG(ske_t)
            
            res = self.dataset.tensor2image(normalized_output[0].cpu()) 
            return res
        else:
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t_batch = ske_t.unsqueeze(0).to(device)  # Déplacer sur le GPU
            normalized_output = self.netG(ske_t_batch)
            print(normalized_output.shape)
            res = self.dataset.tensor2image(normalized_output[0].cpu())  # Déplacer le résultat sur le CPU pour l'affichage
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
        gen.train(50) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256)
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

