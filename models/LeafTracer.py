import torch
import numpy as np
from scipy import ndimage

class LeafTracer():

    '''
    This class applies the tracing algorithm at inference time.
    
    Inputs:
        model   (callable): initialized tracing CNN in eval mode
        window_size  (int): height and width of the input image tile
        path_length  (int): number of pixels to predict along contour
        step_length  (int): number of pixels to step during iteration
        close_dist   (int): number of pixels to stop trace iteration
        device    (device): torch device
        verbose     (bool): whether to print progress updates

    Args:
        image     (tensor): RGB image [3, H, W]
        tile      (tensor): initial input tile [3+1, window_size, window_size]
        idx        (tuple): global indices of tile center
        max_iter     (int): maximum number of trace iterations
        warmup_iter  (int): iterations before checking to stop trace

    Returns:
        mask       (array): output segmentation mask 
    '''
    
    def __init__(
        self, 
        model,
        window_size, 
        path_length, 
        step_length, 
        close_dist,  
        device, 
        verbose=False):

        super().__init__()
        self.model = model
        self.window_size = window_size
        self.path_length = path_length
        self.step_length = step_length
        self.close_dist = close_dist
        self.device = device
        self.verbose = verbose

    # tracing algorithm
    def trace(self, image, tile, idx, max_iter=500, warmup_iter=10):

        # initialize book keeping
        if self.verbose:
            print('Tracing...')
        row, col = idx
        rows = np.empty(shape=(0,), dtype=np.float)
        cols = np.empty(shape=(0,), dtype=np.float)

        for iter in range(max_iter):

            # use CNN predict new trace
            pred = self.model(tile[None].to(self.device))[0].detach().cpu().numpy()
            pred = pred.clip(-self.window_size/2, self.window_size/2)

            # only use pixel predictions up to step_length
            pred_cols, pred_rows = pred[0,:self.step_length], pred[1,:self.step_length]

            # convert local indices to global pixel values
            global_rows = row - pred_rows
            global_cols = col + pred_cols

            # ensure row/col predictions are within image boundary
            w = int(self.window_size/2)
            global_rows = global_rows.clip(w, image.shape[0]-w)
            global_cols = global_cols.clip(w, image.shape[1]-w)

            # check if trace is done
            if iter > 2*warmup_iter:
                index = warmup_iter * self.step_length
                dists = ((global_cols-cols[index])**2 + (global_rows-rows[index])**2)**0.5
                argmin = np.argmin(dists)
                if dists[argmin] < self.close_dist:
                    rows = np.concatenate([rows[index:], global_rows[:argmin]], axis=0)
                    cols = np.concatenate([cols[index:], global_cols[:argmin]], axis=0)
                    if self.verbose:
                        print('Loop closed!')
                    break

            # store global indices and update current index
            rows = np.concatenate([rows, global_rows], axis=0)
            cols = np.concatenate([cols, global_cols], axis=0)
            row, col = rows[-1], cols[-1]

            # construct new image tile
            tile = image[
                int(row - self.window_size/2):int(row + self.window_size/2), 
                int(col - self.window_size/2):int(col + self.window_size/2)]
            tile = np.swapaxes(np.swapaxes(tile, 0, 2), 1, 2)
            tile = torch.tensor(tile, dtype=torch.float32)
            if self.device is not None:
                tile = tile.to(self.device)

            # construct previous trace channel
            n = -min(self.path_length, len(rows))
            previous_rows = (rows[n:] - row + self.window_size/2).clip(0, self.window_size-1)
            previous_cols = (cols[n:] - col + self.window_size/2).clip(0, self.window_size-1)
            contour_channel = torch.zeros_like(tile[0])
            contour_channel[previous_rows.astype(int), previous_cols.astype(int)] = 1

            # combine tile with previous trace
            tile = torch.cat([tile, contour_channel[None]], dim=0)

        # build mask out of contour
        mask = np.zeros([image.shape[0], image.shape[1]])
        for i in range(len(cols)):
            r = np.round(np.linspace(rows[i-1], rows[i]), 10).astype(int)
            c = np.round(np.linspace(cols[i-1], cols[i]), 10).astype(int)
            mask[r, c] = 1
        mask = ndimage.binary_fill_holes(mask)

        return mask