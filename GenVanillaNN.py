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
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image

from Skeleton import Skeleton
from VideoReader import VideoReader
from VideoSkeleton import VideoSkeleton

torch.set_default_dtype(torch.float32)

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ", Skeleton.full_dim, ")")

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        # Prétraitement du squelette (input)
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # Prétraitement de l'image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten())
            ske = ske.to(torch.float32)
            ske = ske.reshape(ske.shape[0], 1, 1)
        return ske


    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().cpu().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        return denormalized_image


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GenNNSkeToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(26, 128, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z):
        x = self.model(z)
        return x


class GenNNSkeImToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeImToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )
        print(self.model)

    def forward(self, z):
        img = self.model(z)
        return img


class GenVanillaNN():
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        if optSkeOrImage == 1:
            self.netG = GenNNSkeToImage()
            src_transform = None
            self.filename = 'data/Dance/DanceGenVanillaFromSke.pth'
        else:
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([SkeToImageTransform(image_size),
                                                transforms.ToTensor()])
            self.filename = 'data/Dance/DanceGenVanillaFromSkeim.pth'

        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform,
                                            source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
        
        # Charger le modèle si requis
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            self.netG.load_state_dict(torch.load(self.filename))

        # Déplacer le modèle vers le GPU
        self.netG.to(device)

    def train(self, n_epochs=20):
        optim = torch.optim.Adam(self.netG.parameters(), lr=0.0002, weight_decay=1e-5)
        for epoch in range(n_epochs):
            for _, data in enumerate(self.dataloader, 0):
                ske, image = data
                ske, image = ske.to(device), image.to(device)  # Déplacer les données sur le GPU
                self.netG.zero_grad()
                output = self.netG(ske)
                loss = F.mse_loss(output, image)
                loss.backward()
                optim.step()
                print('epoch:', epoch, 'loss:', loss.item())

        torch.save(self.netG.state_dict(), self.filename)

    def generate(self, ske):
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0).to(device)  # Déplacer sur le GPU
        normalized_output = self.netG(ske_t_batch)
        print(normalized_output.shape)
        res = self.dataset.tensor2image(normalized_output[0].cpu())  # Déplacer le résultat sur le CPU pour l'affichage
        return res


if __name__ == '__main__':
    force = False
    optSkeOrImage = 1
    n_epoch = 200
    train = 1

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"

    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=optSkeOrImage)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True, optSkeOrImage=optSkeOrImage)

    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        nouvelle_taille = (256, 256)
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
