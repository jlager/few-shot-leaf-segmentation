import random
import numpy as np
from scipy import ndimage
import torch
import torchvision
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

class UNetTileGenerator(Dataset):

    def __init__(
        self, 
        images, 
        masks, 
        rois=None, 
        window_size=256, 
        n_samples=None,
        dilate=0,
        augment=False,
        verbose=True):
        
        # initialize
        super().__init__()
        self.images = images
        self.masks = masks
        self.rois = rois
        self.window_size = window_size
        self.n_samples = n_samples
        self.dilate = dilate
        self.augment = augment
        self.verbose = verbose
        
        # extract contour indices from masks [N, 3]
        self.indices = self.extract_locations(masks)
        
        # define row/col pixel indices for sampling rotated tiles
        w = int(self.window_size/2)
        X, Y = np.meshgrid(np.arange(-w, w), np.arange(-w, w), indexing='ij')
        self.image_slice = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        
    def __len__(self):
        return len(self.indices)

    # get contour pixel locations [img_idx, row, col]
    def extract_locations(self, masks):
        
        # loop over each mask
        indices = []
        if self.verbose:
            print('Extracting contours...')
        mask_range = tqdm(range(len(masks))) if self.verbose else range(len(masks))
        for i in mask_range:

            # get mask
            mask = masks[i].copy()

            # apply roi
            if self.rois is not None:
                mask = mask + self.rois[i]
                mask = np.clip(mask, 0, 1)

            # dilate mask to get more white space
            if self.dilate > 0:
                mask = ndimage.binary_dilation(masks[i], iterations=self.dilate)
            
            # get row/col pixels with image ids included
            idx = np.fliplr(np.argwhere(mask > 0)) # [N, 2]
            idx = np.concatenate([i*np.ones_like(idx[:,0:1]), idx], axis=1) # [N, 3]
            indices.append(idx)
            
        # combine and shuffle
        indices = np.concatenate(indices, axis=0)
        indices = indices[np.random.permutation(len(indices))].astype(int)

        # downsample
        if self.n_samples is not None:
            if len(indices) > self.n_samples:
                indices = indices[:self.n_samples]
            
        return indices
    
    # swap image axes [H, W, C] -> [C, H, W]
    def channels_first(self, image):
        return np.swapaxes(np.swapaxes(image, 0, 2), 1, 2)
    
    # swap image axes [C, H, W] -> [H, W, C]
    def channels_last(self, image):
        return np.swapaxes(np.swapaxes(image, 0, 2), 0, 1)
    
    # convert numpy image to torch tensor
    def image2torch(self, image):
        image = torch.tensor(self.channels_first(image), dtype=torch.float32) # [3, H, W]
        return image
    
    # convert numpy mask to torch tensor
    def mask2torch(self, mask):
        mask = torch.tensor(mask, dtype=torch.float32) # [H, W]
        return mask
    
    # convert torch tensor to numpy image
    def image2numpy(self, image):
        image = self.channels_last(image[:3].detach().cpu().numpy()) # [H, W, 3]
        return image
    
    # convert torch tensor to numpy mask
    def mask2numpy(self, mask):
        if len(mask.shape) == 3:
            mask = mask[0]
        mask = mask.detach().cpu().numpy() # [H, W]
        return mask
    
    # builds rotation matrix from angle theta
    def build_rotation_matrix(self, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta),  np.cos(theta)]])
    
    # get rgb tile and return half-sized mask
    def __getitem__(self, index, debug=False):
            
        # parse index
        idx = self.indices[index]
        i, k, j = idx[0], idx[1], idx[2] # image_idx, col, row
            
        # optionally apply rotation
        theta = np.random.choice(np.linspace(0, 2*np.pi, 360)) if self.augment else 0
        rotation = self.build_rotation_matrix(theta)
        locs = (np.array([[j, k]]) + np.dot(rotation, self.image_slice.T).T).astype(int)
        
        # extract tiles by (rotated) pixel locations, convert to torch
        w = self.window_size
        tile = self.images[i][locs[:,0], locs[:,1]].reshape(w, w, 3)
        mask = self.masks[i][locs[:,0], locs[:,1]].reshape(w, w)
        tile = self.image2torch(tile)
        mask = self.mask2torch(mask)

        # augment tiles
        if self.augment:
        
            # random horizontal/vertical flips
            if random.random() > 0.5:
                tile = torchvision.transforms.functional.hflip(tile)
                mask = torchvision.transforms.functional.hflip(mask)
            if random.random() > 0.5:
                tile = torchvision.transforms.functional.vflip(tile)
                mask = torchvision.transforms.functional.vflip(mask)

            # random color augmentation
            tile = torchvision.transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                hue=0.1, 
                saturation=0.1)(tile)

        # # zoom into center of mask
        # w = int(self.window_size/2/2)
        # mask = mask[w:-w, w:-w]

        # add channel dimension to mask
        mask = mask.unsqueeze(0)
            
        return tile, mask