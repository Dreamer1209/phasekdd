import os 
import glob 

import torch
import random
import cv2 
import h5py
import pickle
import numpy as np 
import PIL.Image as Image
import matplotlib.pyplot as plt

from .hsi import HSIDataset
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple


class Load(HSIDataset):
    resolution = {
        'height': 1024,
        'width': 1024,
        'bands': 31,
    }
    
    def __init__(self, 
                rgb_dir: str, 
                hsi_dir: str, 
                train_test_mask: bool = None, 
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None):
        super().__init__(root = rgb_dir, transform=transform, target_transform=target_transform)

        # files location
        self.rgb_dir = rgb_dir
        self.hsi_dir = hsi_dir
        self.rgb_files = sorted(glob.glob(self.rgb_dir + "/*.jpg"))
        self.cube_files = sorted(glob.glob(self.hsi_dir + "/*.mat"))

        # total data
        self.rgb_files = np.asarray(self.rgb_files)
        self.cube_files = np.asarray(self.cube_files)
        if train_test_mask is not None:
            self.rgb_files = self.rgb_files[train_test_mask]
            self.cube_files = self.cube_files[train_test_mask]
        self.total_files = len(self.rgb_files)


    def loadCube(self, cube_path):
        '''
        return cube in (h, w, c=31)
        range: (0, 1)
        '''
        with h5py.File(cube_path, 'r') as f:
            cube = np.squeeze(np.float32(np.array(f['cube'])))
            cube = np.transpose(cube, [2,1,0]) 
            f.close()
        return cube

    def loadData(self, img_path, cube_path):
        # load image file
        rgb = plt.imread(img_path)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        # load cube file
        cube = self.loadCube(cube_path)

        return rgb, cube

    def __getitem__(self, idx):
        rgb, cube = self.loadData(self.rgb_files[idx], self.cube_files[idx])

        if self.transform is not None:
            all = np.concatenate([rgb, cube], axis = -1)
            all = self.transform(all)
            rgb = all[:3, :, :]
            cube = all[3:, :, :]

        return rgb, cube

    def __len__(self):
        return self.total_files
   

# v2 for NIR
class Load_v2(HSIDataset):
    resolution = {
        'height': 1024,
        'width': 1024,
        'bands': 31,
    }
    
    def __init__(self, 
                rgb_dir: str, 
                hsi_dir: str, 
                train_test_mask: bool = None, 
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None):
        super().__init__(root = rgb_dir, transform=transform, target_transform=target_transform)

        # files location
        self.rgb_dir = rgb_dir
        self.hsi_dir = hsi_dir
        self.rgb_files = sorted(glob.glob(self.rgb_dir + "/*.mat"))
        self.cube_files = sorted(glob.glob(self.hsi_dir + "/*.mat"))

        # total data
        self.rgb_files = np.asarray(self.rgb_files)
        self.cube_files = np.asarray(self.cube_files)
        if train_test_mask is not None:
            self.rgb_files = self.rgb_files[train_test_mask]
            self.cube_files = self.cube_files[train_test_mask]
        self.total_files = len(self.rgb_files)


    def loadCube(self, cube_path):
        '''
        return cube in (h, w, c=31)
        range: (0, 1)
        '''
        with h5py.File(cube_path, 'r') as f:
            cube = np.squeeze(np.float32(np.array(f['cube'])))
            cube = np.transpose(cube, [2,1,0]) 
            f.close()
        return cube

    def loadData(self, img_path, cube_path):
        # load MSI data (RGB + 960nm) 1024*1024*4
        rgb = self.loadCube(img_path)

        # load cube file
        cube = self.loadCube(cube_path)

        return rgb, cube

    def __getitem__(self, idx):
        rgb, cube = self.loadData(self.rgb_files[idx], self.cube_files[idx])

        if self.transform is not None:
            all = np.concatenate([rgb, cube], axis = -1)
            all = self.transform(all)
            rgb = all[:4, :, :]
            cube = all[4:, :, :]

        return rgb, cube

    def __len__(self):
        return self.total_files

#####################################################################
def defineCrop(stride, crop_size, height, width):
    if stride < 0 or crop_size > np.minimum(height, width) or crop_size < 0:
        print('stride and crop size must be valid to perform crop operation')
        crop_size = -1
        return -1, -1, -1, -1, -1
    else:
        patch_per_line = (width - crop_size) // stride + 1
        patch_per_colume = (height - crop_size) // stride + 1
        patch_per_img = patch_per_line * patch_per_colume
    return crop_size, stride, patch_per_line, patch_per_colume, patch_per_img

def crop(idx, img, cube, patch_per_img, patch_per_line, stride, crop_size):
    patch_idx = idx % patch_per_img
    h_idx = patch_idx // patch_per_line
    w_idx = patch_idx % patch_per_line

    img = img[
        h_idx * stride:h_idx * stride + crop_size, \
        w_idx * stride:w_idx * stride + crop_size, \
        :]
    cube = cube[
        h_idx * stride:h_idx * stride + crop_size, \
        w_idx * stride:w_idx * stride + crop_size, \
        :]

    return img, cube

class SkinLoad(torch.utils.data.Dataset):
    resolution = {
        'height': 1024,
        'width': 1024,
        'bands': 31,
    }

    def __init__(self,
                rgb_dir: str,
                hsi_dir: str,
                filelist: Optional[list] = None,
                do_crop: bool = False,
                stride: int = 8,
                crop_size: int = 128,
                do_aug: bool = False,
                do_shuffle: bool = False,
                do_shift: bool = False,
                to_chw: bool = False,
                transform: Optional[Callable] = None,
                load_img_type: str = 'rgb',
                unsupervised: bool = False):
        super().__init__()
        
        self.rgb_dir = rgb_dir
        self.hsi_dir = hsi_dir
        self.transform = transform
        self.unsupervised = unsupervised

        if filelist is None:
            self.rgb_files = sorted(glob.glob(self.rgb_dir + "/*.jpg"))
            self.cube_files = sorted(glob.glob(self.hsi_dir + "/*.mat"))
        else:
            self.rgb_files = [os.path.join(self.rgb_dir, f) for f in filelist]
            self.cube_files = [os.path.join(self.hsi_dir, f.replace('.jpg', '.mat')) for f in filelist]

        self.rgb_files = np.asarray(self.rgb_files)
        self.cube_files = np.asarray(self.cube_files)
        self.total_files = len(self.rgb_files)

        if do_shuffle:
            id = np.random.permutation(self.total_files)
            self.rgb_files = self.rgb_files[id]
            self.cube_files = self.cube_files[id]

        if do_crop:
            self.crop_size, self.stride, \
              self.patch_per_line, self.patch_per_colume, self.patch_per_img \
                = defineCrop(stride, crop_size, self.resolution['height'], self.resolution['width'])
        else:
            self.patch_per_img = 1
            self.crop_size = -1
            self.stride = -1

        self.do_aug = do_aug
        self.do_shift = do_shift
        self.to_chw = to_chw
        self.load_img_type = load_img_type

    def loadCube(self, cube_path):
        with h5py.File(cube_path, 'r') as f:
            cube = np.squeeze(np.float32(np.array(f['cube'])))
            cube = np.transpose(cube, [2,1,0]) 
            f.close()
        return cube
    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def loadData(self, img_path, cube_path):

        if self.load_img_type == 'rgb':
            rgb = plt.imread(img_path)
        # print(f"SKIN Original RGB shape: {rgb.shape}, value range: ({rgb.min()}, {rgb.max()})")

        # Normalize RGB image
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        # print(f"SKIN Normalized RGB shape: {rgb.shape}, value range: ({rgb.min()}, {rgb.max()})")

        # Load hyperspectral image
        cube = self.loadCube(cube_path)
        # print(f"SKIN Original Cube shape: {cube.shape}, value range: ({cube.min()}, {cube.max()})")
        return rgb, cube

    def __getitem__(self, idx):
        # img_idx = idx // self.patch_per_img
        rgb, cube = self.loadData(self.rgb_files[idx], self.cube_files[idx])

        # if self.crop_size > 0:
        #     rgb, cube = crop(idx, rgb, cube, self.patch_per_img, self.patch_per_line, self.stride, self.crop_size)
        # if self.to_chw:
        #     rgb = np.transpose(rgb, [2, 0, 1])  # (c, h, w)
        #     cube = np.transpose(cube, [2, 0, 1])  # (c, h, w)
        if self.do_aug:  
            rotTimes = random.randint(0, 3)
            vFlip = random.randint(0, 1)
            hFlip = random.randint(0, 1)
            rgb = self.arguement(rgb, rotTimes, vFlip, hFlip)
            cube = self.arguement(cube, rotTimes, vFlip, hFlip)
        if self.transform is not None:
            all = np.concatenate([rgb, cube], axis = -1)
            all = self.transform(all)
            rgb = all[:3, :, :]
            cube = all[3:, :, :]
            
        if self.unsupervised:
            return rgb
        else:
            return rgb, cube

    def __len__(self):
        return self.patch_per_img * self.total_files

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for _ in range(rotTimes):
            img = np.rot90(img.copy(), axes=(0, 1))  
        # Random vertical Flip
        if vFlip:
            img = img[::-1, :, :].copy()  
        # Random horizontal Flip
        if hFlip:
            img = img[:, ::-1, :].copy()  
        return img