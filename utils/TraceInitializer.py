import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from skimage import measure

class TraceInitializer():
    
    '''
    Class for initializing the LeafTracer. Functionality includes loading images,
    generating rough segmentations based on auto-thresholding, and generating 
    initial image tiles and contours for LeafTracer initialization. 

    Inputs:
        window_size (int): height and width of the input image tile
        step_length (int): number of pixels to step during iteration
        device   (device): torch device
    '''
    
    def __init__(self, window_size, step_length, device=None):
        
        super().__init__()
        self.window_size = window_size
        self.step_length = step_length
        self.device = device
        
    # load entire scanned leaf
    def load_image(self, path, pad=False):
        w = int(self.window_size/2)
        image = np.array(Image.open(path), dtype=float)/255
        if pad:
            image = np.pad(
                array=image, 
                pad_width=[[w, w], [w, w], [0, 0]], 
                mode='constant', 
                constant_values=1.0)
        return image

    # auto-detect segmentation threshold
    def detect_threshold(self, image, tol=1e-4, maxiter=10):
        old_threshold = image.mean()
        count = 0
        while True:
            new_threshold = np.mean([
                image[image<=old_threshold].mean(),
                image[image>old_threshold].mean()])
            if np.abs(new_threshold - old_threshold) < tol:
                break
            if count > maxiter:
                break
            old_threshold = new_threshold
            count += 1
        return new_threshold
    
    # convert RGB image to grayscale
    def rgb2gray(self, rgb):
        gray = (rgb * np.array([[[0.2989, 0.5870, 0.1140]]])).sum(-1)
        return gray
    
    # extract rough segmentation mask with thresholding
    def threshold_segmentation(self, rgb):
        gray = self.rgb2gray(rgb)
        threshold = self.detect_threshold(gray)
        mask = (gray < threshold).astype(float)
        return mask
        
    # remove noise and fill holes in segmentation mask
    def clean_segmentation(self, mask):
        
        # identify all objects and compute their size
        labeled_array, num_features = ndimage.label(mask)
        index = torch.tensor(labeled_array, dtype=torch.long).reshape(-1)
        src = torch.ones_like(index)
        sizes = torch.zeros(index.max()+1, dtype=torch.long)
        sizes = sizes.scatter_add(dim=0, index=index, src=src)
        sizes = sizes.detach().cpu().numpy()

        # order objects by size
        labels = np.arange(len(sizes))
        idx = np.flip(np.argsort(sizes))
        labels = labels[idx]
        sizes = sizes[idx]

        # keep largest object in mask
        mask = (labeled_array==labels[1]).astype(float)

        # fill empty holes
        mask = ndimage.binary_fill_holes(mask)
        
        return mask

    # get vanilla segmentation
    def extract_segmentation(self, rgb):
        mask = self.threshold_segmentation(rgb)
        mask = self.clean_segmentation(mask)
        return mask
    
    # extract contour from segmentation mask
    def extract_contour(self, mask):
        contours = measure.find_contours(mask, 0.5)[0]
        contours = np.fliplr(contours)
        return contours

    # find point on contour closest to top center of image (for trace initialization)
    def extract_top_contour(self, mask):
        contour = self.extract_contour(mask)
        top_center = np.array([[mask.shape[1]/2, 0]])
        diffs = top_center - contour
        dists = np.linalg.norm(diffs, ord=2, axis=-1)
        initial_trace = contour[np.argmin(dists)]
        return initial_trace
    
    # extract image/mask tiles at specified location
    def extract_tile(self, rgb, mask, idx):
        row, col, w = idx[0], idx[1], self.window_size
        tile = rgb[int(row-w/2):int(row+w/2), int(col-w/2):int(col+w/2)]
        mask = mask[int(row-w/2):int(row+w/2), int(col-w/2):int(col+w/2)]
        return tile, mask
    
    # extract center-most contour from mask tile
    def extract_tile_contour(self, mask):
        contours = measure.find_contours(mask, 0.5)
        if len(contours) == 1:
            contours = contours[0]
        else: # choose contour that intersects midpoint
            window_size = mask.shape[0]
            dists_from_midpoint = []
            for i, contour in enumerate(contours):
                diffs = contour - np.array([[window_size/2,window_size/2]])
                dists = np.linalg.norm(diffs, ord=2, axis=-1)
                dists_from_midpoint.append(dists.min())
            contours = contours[np.argmin(dists_from_midpoint)] 
        contours = np.fliplr(contours)
        return contours
    
    # find index of contour pixel closest to tile center
    def extract_midpoint(self, contours):
        midpoint = np.array([[self.window_size/2, self.window_size/2]])
        diffs = contours - midpoint
        dists = np.linalg.norm(diffs, ord=2, axis=-1)
        idx = np.argmin(dists)
        return idx
    
    # split contour into two paths
    def split_contour(self, contour):
        midpoint_idx = self.extract_midpoint(contour)
        old_trace = contour[:midpoint_idx+1][-self.step_length:]
        new_trace = contour[midpoint_idx+1:][:self.step_length]
        return old_trace, new_trace
    
    # draw previously traced contour onto new image channel
    def contour2channel(self, contour, mask):
        contour_channel = np.zeros_like(mask)
        contour_channel[contour[:,1].astype(int), contour[:,0].astype(int)] = 1
        return contour_channel
    
    # swap image axes [H, W, C] -> [C, H, W]
    def channels_first(self, image):
        return np.swapaxes(np.swapaxes(image, 0, 2), 1, 2)
    
    # swap image axes [C, H, W] -> [H, W, C]
    def channels_last(self, image):
        return np.swapaxes(np.swapaxes(image, 0, 2), 0, 1)
    
    # convert numpy image to torch tensor
    def numpy2torch(self, image):
        image = self.channels_first(image)
        image = torch.tensor(image, dtype=torch.float32)
        return image
    
    # convert torch tensor to numpy image
    def torch2numpy(self, image):
        image = image.detach().cpu().numpy()
        image = self.channels_last(image)
        return image
    
    # get input tile for rgb image
    def initialize_trace(self, rgb):
        
        # extract mask
        mask = self.extract_segmentation(rgb)
        
        # initialize trace at leaf tip and zoom in
        idx = np.flip(self.extract_top_contour(mask))
        tile, mask = self.extract_tile(rgb, mask, idx)
        
        # add "previous" trace as additional channel
        contour = self.extract_tile_contour(mask)
        old_trace, new_trace = self.split_contour(contour)
        contour_channel = self.contour2channel(old_trace, mask)
        tile = np.concatenate([tile, contour_channel[:,:,None]], axis=-1)
        
        # convert to torch
        tile = self.numpy2torch(tile)
        if self.device is not None:
            tile = tile.to(self.device)
        
        return tile, idx