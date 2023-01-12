import os, glob
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm

class ImageLoader():

    '''
    Loads image/mask pairs by finding images corresponding to mask file names.
    Images are assumed to be RGB and are normalized to [0, 1] float arrays. 
    Masks are converted to boolean arrays. Images and masks are optionally 
    padded by half of the specified CNN window size. If ROI specified also 
    loads ROI (e.g., leaf segmentations used during vein growing).
    
    Args: 
        image_path  (str): path to image data
        mask_path   (str): path to mask data
        roi_path    (str): path to roi data
        image_ext   (str): image file extension (e.g., 'jpeg')
        mask_ext    (str): mask file extension (e.g., 'png')
        roi_ext     (str): roi file extension (e.g., 'png')
        window_size (int): width of the CNN input (e.g., 256)
        pad        (bool): whether to pad images and masks
        verbose    (bool): whether to update user during data loading
        
    Returns:
        images     (list): float arrays containing normalized RGB images
        masks      (list): bool arrays containing masks
        rois       (list): bool arrays containing ROIs (optional)
    '''
    
    def __init__(
        self, 
        image_path, 
        mask_path, 
        roi_path=None,
        image_ext='jpeg',
        mask_ext='png',
        roi_ext=None,
        window_size=256,
        pad=True,
        verbose=False,
        ):
        
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.roi_path = roi_path
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.roi_ext = roi_ext
        self.window_size = window_size
        self.pad = pad
        self.verbose = verbose
        self.file_names = [os.path.basename(f) for f in glob.glob(self.mask_path + '*' + self.mask_ext)]
        
    def __len__(self):
        return len(self.file_names)
        
    def load_image(self, path, pad=None):

        # load image as float array
        image = np.array(Image.open(path), dtype=float)

        # pre-process image
        if image.shape[2] > 3:
            image = image[:, :, :3]
        if image.max() > 1.0:
            image = image / 255

        # pad image (optional)
        w = int(self.window_size/2)
        if pad is not None:
            image = np.pad(
                array=image, 
                pad_width=[[w, w], [w, w], [0, 0]], 
                mode='constant', 
                constant_values=pad)

        # shape [H, W, 3]
        return image.astype(float)
        
    def load_mask(self, path, pad=None):

        # load mask as int array
        mask = np.array(Image.open(path))

        # pre-process mask
        if mask.max() > 1.0:
            mask = mask / 255
        if len(mask.shape) == 3:
            mask = mask[:,:,0]
        mask = mask > 0.5 

        # pad mask (optional)
        w = int(self.window_size/2)
        if pad is not None:
            mask = np.pad(
                array=mask, 
                pad_width=[[w, w], [w, w]], 
                mode='constant', 
                constant_values=pad)

        # shape [H, W]
        return mask.astype(bool)
    
    def load_data(self):

        images, masks, rois = [], [], []
        file_names = tqdm(self.file_names) if self.verbose else self.file_names

        for file_name in file_names:
            
            # load image
            image_name = self.image_path + file_name.replace(self.mask_ext, self.image_ext)
            pad = 1.0 if self.pad else None
            image = self.load_image(image_name, pad=pad)
            images.append(image)

            # load mask
            mask_name = self.mask_path + file_name
            pad = 0.0 if self.pad else None
            mask = self.load_mask(mask_name, pad=pad)
            masks.append(mask)

            # optionally load roi
            if self.roi_path is not None:
                roi_name = self.roi_path + file_name.replace(self.mask_ext, self.roi_ext)
                pad = 0.0 if self.pad else None
                roi = self.load_mask(roi_name, pad=pad)
                rois.append(roi)
            
        if self.roi_path is None:
            return images, masks
        else:
            return images, masks, rois