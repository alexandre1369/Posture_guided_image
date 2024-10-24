
import math
import os
import pickle
import sys

import cv2
import numpy as np

from Skeleton import Skeleton
from VideoReader import VideoReader
from VideoSkeleton import VideoSkeleton


class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):
        """ generator of image from skeleton """
        empty = np.ones((64,64, 3), dtype=np.uint8)
        
        # Find the nearest skeleton in the target video
        ditance_min = 1000
        image = empty
        for i in range(self.videoSkeletonTarget.skeCount()):
            skeTgt = self.videoSkeletonTarget.ske[i]
            if skeTgt.distance(ske) < ditance_min:
                ditance_min = skeTgt.distance(ske)
                image = self.videoSkeletonTarget.readImage(i)
            
        
        
        return image




