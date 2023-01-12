import os, sys, pdb
import torch
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from torch.utils.data import Dataset, DataLoader

class TileGenerator(Dataset):
    
    def __init__(self, image, indices, w):
        
        super().__init__()
        self.image = image
        self.indices = indices
        self.w = w
        
        # ensure indices are in correct format
        if len(self.indices) == 2:
            try: # if this works, do nothing
                idx = self.indices[0]
                i, j = idx[0], idx[1]
            except: # if not, restructure 
                i = self.indices[0][0]
                j = self.indices[1][0]
                self.indices = np.array([[i,j]])
                
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        idx = torch.tensor(self.indices[index], dtype=torch.long)
        i, j = idx[0], idx[1]
        x = torch.tensor(
            self.image[:, i-self.w:i+self.w, j-self.w:j+self.w], 
            dtype=torch.float)
        return idx, x

class VeinGrower():
    
    '''
    Grows region on an image given starting location(s). Stops when no more
    pixels are added to region.
    
    Args:
        model    (callable): initialized CNN in eval mode
        window_size   (int): height and width of input tile
        device     (device): torch device
        verbose      (bool): whether to print progress updates

    Inputs:
        image       (array): image with shape (3, H, W) 
        roi         (array): optional 
        start_locs  (array): seed pixels with shape (N, 2)
        n_locs        (int): number of seed pixels to sample
        batch_size    (int): model prediciton batch size
        threshold   (float): mask probability threshold (default None)
        post_process (bool): whether to clean up the segmentation mask
        
    Returns:
        probs      (tensor): probability map (H, W)
        veins      (tensor): thresholded probabilities (H, W)
    '''
    
    def __init__(
        self, 
        model, 
        window_size, 
        device=None, 
        verbose=False):
        
        super().__init__()
        self.window_size = window_size
        self.model = model
        self.device = device
        self.verbose = verbose
        
    # swap image axes [H, W, C] -> [C, H, W]
    def channels_first(self, image):
        return np.swapaxes(np.swapaxes(image, 0, 2), 1, 2)
    
    # swap image axes [C, H, W] -> [H, W, C]
    def channels_last(self, image):
        return np.swapaxes(np.swapaxes(image, 0, 2), 0, 1)
        
    def grow(
        self, 
        image, 
        roi=None, 
        start_locs=None, 
        n_locs=10000, 
        batch_size=1024, 
        threshold=None,
        post_process=True):
        
        # pad image and mask with zeros
        w = int(self.window_size/2)
        image = np.pad(image, [[w,w],[w,w],[0,0]], 'constant', constant_values=1.0)
        if roi is not None:
            roi = np.pad(roi, [[w,w],[w,w]], 'constant', constant_values=0.0)
            
        # reorder image channels
        image = self.channels_first(image)
        
        # initialize book keeping
        mask = np.zeros((2, image.shape[-2], image.shape[-1]))
        sample = np.zeros((image.shape[-2], image.shape[-1]))
        unconsidered = np.zeros((image.shape[-2], image.shape[-1]))
        unconsidered[w:-w, w:-w] = 1 # don't consider padding

        # initialize sample
        if start_locs is None:
            if roi is None:
                i = np.random.choice(image.shape[1], n_locs)
                j = np.random.choice(image.shape[2], n_locs)
                start_locs = np.concatenate([i[:, None], j[:, None]], axis=1)
            else:
                start_locs = np.argwhere(roi == 1.0)
                p = np.random.permutation(len(start_locs))
                start_locs = start_locs[p[:n_locs]]
        for i in range(len(start_locs)):
            sample[start_locs[i, 0], start_locs[i, 1]] = 1
            
        # initialize sample locations
        locs = np.where(unconsidered*sample == 1)
        
        # predict until no more sample locations
        count = 0
        while len(locs) != 0:
            
            # instantiate data generator for current locations
            tile_generator = TileGenerator(image, locs, w)
            tile_batch_loader = DataLoader(tile_generator, 
                                           batch_size=batch_size, 
                                           shuffle=False, 
                                           num_workers=0)
            
            # loop over batches indices/tiles
            for idx_batch, tile_batch in tile_batch_loader:
                
                # make model prediction
                idx_batch = idx_batch.detach().cpu().numpy()
                tile_batch = tile_batch.to(self.device)
                with torch.no_grad():
                    pred_batch = self.model(tile_batch).detach().cpu().numpy()
                
                # update mask, sample, and unconsidered
                for idx, pred in zip(idx_batch, pred_batch):
                    i, j = idx[0], idx[1]
                    mask[:, i-1:i+2, j-1:j+2] += pred
                    unconsidered[i, j] = 0
                    f = mask[0, i-1:i+2, j-1:j+2] # foreground
                    b = mask[1, i-1:i+2, j-1:j+2] # background
                    sample[i-1:i+2, j-1:j+2] = 1.0*((f-b) > -0.2)
                
            # update sample and locations
            locs = np.argwhere(unconsidered*sample == 1)
            
            # update user
            if self.verbose:
                p = '\rIteration {0}'.format(count)
                p += ' | Samples = {0}'.format(len(locs))
                p += '           '
                sys.stdout.write(p)
            
            # update counter
            count += 1
            
        if self.verbose:
            print()
        
        # normalize mask probabilities
        probs = mask / ((mask[0:1] + mask[1:2]).clip(1.0, np.inf))
        
        # threshold probabilities for venation mask
        if self.verbose:
            print('Computing optimal threshold...')
        if threshold is None:
            thresholds = np.linspace(0.1, 0.9, 101)
            structure = ndimage.generate_binary_structure(2,2)
            n_objects, sizes = np.array(
                [[ndimage.label(probs[0]>t, structure=structure)[1], (probs[0]>t).sum()] for t in thresholds]).T
            peaks, _ = find_peaks(-n_objects/sizes, prominence=100/sizes.max(), distance=10)
            threshold = thresholds[peaks[0]]
        veins = 1.0*np.array(probs[0] > threshold)

        # keep largest object (e.g., petiole) outside of the ROI
        if post_process and roi is not None:
            
            if self.verbose:
                print('Post processing...')
            
            # separate vein/petiole
            venation = roi * veins
            petiole = ~roi * veins
            
            # format petiole
            labeled_array, num_features = ndimage.label(petiole)
            if num_features >= 2:
                sizes = np.zeros(num_features+1)
                unique_labels = np.unique(labeled_array)
                for i,u in enumerate(unique_labels):
                    size = (labeled_array==u).sum()
                    sizes[i] = size
                largest = unique_labels[np.argsort(sizes)[-2]]
                petiole = labeled_array == largest
                
            # add vein + petiole
            veins = venation + petiole
            
        # remove padding
        probs = probs[:, w:-w, w:-w]
        veins = veins[w:-w, w:-w] > 0.5
        
        return probs, veins