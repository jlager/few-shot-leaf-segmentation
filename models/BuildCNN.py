import torch
import numpy as np

class CNN(torch.nn.Module):
    
    '''
    Make a CNN for leaf tracing or vein growing. Number of layers is automatically
    computed using the window size. Final conv layer is reshaped into output_shape
    with optional output activation function (e.g., Softmax). Uses BuildCNN class
    to make core CNN model, see below for BuildCNN.
    
    Args:
        window_size            (int): width of the CNN input (e.g., 256)
        layers                (list): layer sizes (e.g., [32, 32, 32, 32, 64, 128])
        output_shape          (list): shape of the CNN output (e.g., [2, 128])
        output_activation (callable): optional output activation function
    
    Inputs:
        x (tensor): batch of input image tensors [B, C, W, W]
    
    Returns:
        x (tensor): batch of output tensors [B, output_shape]
    '''
    
    def __init__(
        self, 
        window_size, 
        layers, 
        output_shape, 
        output_activation=None
        ):

        # initialize
        super().__init__()
        self.window_size = window_size
        self.input_channels = layers[0]
        self.layers = layers[1:]
        self.output_shape = output_shape
        self.output_activation = output_activation
        self.outputs = self.layers[-1] * int(window_size/2**(len(self.layers)-1))**2

        # build CNN
        self.cnn = BuildCNN(
            input_channels=self.input_channels,
            layers=self.layers,
            activation=torch.nn.LeakyReLU(),
            pool='max',
            num_convs=3,
            use_batchnorm=True,
            dropout_rate=0.0)

        # conv + reshape
        self.conv = torch.nn.Conv2d(
            in_channels=self.layers[-1],
            out_channels=np.prod(output_shape),
            kernel_size=int(window_size/2**(len(self.layers)-1)),
            stride=1,
            padding=0)
        
    def forward(self, x):

        # get shape
        batch_size = x.shape[0]
        output_shape = [batch_size] + self.output_shape

        # forward pass
        x = self.cnn(x) # [B, F, W, W] -> [B, F', 4, 4]
        x = self.conv(x) # [B, F', 4, 4] -> [B, prod(self.output_shape)]
        x = x.view(output_shape) # -> [B, *self.output_shape]
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x

class BuildCNN(torch.nn.Module):
    
    '''
    Builds a standard convolutional neural network (CNN) with optional
    number of convolutions between pooling and use of batch norm. If the 
    number of convolutions per block is greater than 1, then the model
    applies skip connections to the remaining convolutions in each block.
    
    Args:
        input_channels  (int): integer number of input features
        layers         (list): list of integer layer sizes
        activation (callable): instantiated activation function
        pool            (str): string for pooling type: max, avg, conv
        num_convs       (int): integer number of convolutions between pools
        linear_output  (bool): boolean indicator for linear output
        dim             (int): model dimension (1D, 2D, 3D)
    
    Inputs:
        x (tensor): batch of input image tensors
    
    Returns:
        x (tensor): batch of output tensors
    '''
    
    def __init__(self, 
                 input_channels, 
                 layers, 
                 activation=None, 
                 pool='max', 
                 num_convs=3, 
                 use_batchnorm=True,
                 dropout_rate=0.0,
                 dim=2):
        
        # initialization
        super().__init__()
        self.input_channels = input_channels
        self.layers = layers
        self.activation = activation if activation is not None else torch.nn.LeakyReLU()
        self.pool = pool
        self.num_convs = num_convs
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        self.dim = dim
        
        assert self.pool in ['max', 'avg', 'conv']
        assert self.dim in [1, 2, 3]
        
        # define operations
        if self.dim == 1:
            conv = torch.nn.Conv1d
            batchnorm = torch.nn.BatchNorm1d
            dropout = torch.nn.Dropout
            if self.pool == 'max':
                pool = torch.nn.MaxPool1d
            elif self.pool == 'avg':
                pool = torch.nn.AvgPool1d
            else:
                pool = torch.nn.Conv1d
        if self.dim == 2:
            conv = torch.nn.Conv2d
            batchnorm = torch.nn.BatchNorm2d
            dropout = torch.nn.Dropout2d
            if self.pool == 'max':
                pool = torch.nn.MaxPool2d
            elif self.pool == 'avg':
                pool = torch.nn.AvgPool2d
            else:
                pool = torch.nn.Conv2d
        if self.dim == 3:
            conv = torch.nn.Conv3d
            batchnorm = torch.nn.BatchNorm3d
            dropout = torch.nn.Dropout3d
            if self.pool == 'max':
                pool = torch.nn.MaxPool3d
            elif self.pool == 'avg':
                pool = torch.nn.AvgPool3d
            else:
                pool = torch.nn.Conv3d
        
        # instantiation
        self.conv_list = torch.nn.ModuleList()
        self.norm_list = torch.nn.ModuleList()
        self.drop_list = torch.nn.ModuleList()
        self.pool_list = torch.nn.ModuleList()
        
        # loop over blocks
        for i, layer in enumerate(self.layers): 
            
            # loop over convs
            for j in range(self.num_convs): 
                
                # add conv 
                self.conv_list.append(conv(
                    in_channels=self.input_channels,
                    out_channels=layer,
                    kernel_size=3,
                    stride=1,
                    padding=1))
                
                # add optional batchnorm
                if self.use_batchnorm:
                    self.norm_list.append(batchnorm(layer))

                # add optinal dropout
                if self.dropout_rate > 0:
                    self.drop_list.append(dropout(p=self.dropout_rate))
                    
                # update book keeping
                self.input_channels = layer
            
            # add pooling layer
            if i != len(self.layers)-1:
                if self.pool == 'max' or self.pool == 'avg':
                    self.pool_list.append(pool(
                        kernel_size=2, 
                        stride=2))
                else: 
                    
                    # conv pool
                    p = [pool(
                        in_channels=self.input_channels,
                        out_channels=layer,
                        kernel_size=3,
                        stride=2,
                        padding=1)]
                    
                    # optional batchnorm
                    if self.use_batchnorm:
                        p = p + [batchnorm(layer)]
                    
                    # activation
                    p = p + [self.activation]
                        
                    self.pool_list.append(torch.nn.Sequential(*p))
                    self.input_channels = layer
        
    def forward(self, x):
        
        # loop over blocks
        for i in range(len(self.layers)): 
            
            # loop over convs
            for j in range(self.num_convs): 
                       
                # store input for residual connection
                if j > 0:
                    x_copy = x.clone()
                
                # add conv 
                x = self.conv_list[i*self.num_convs + j](x)
                
                # add optional batchnorm
                if self.use_batchnorm:
                    x = self.norm_list[i*self.num_convs + j](x)
                
                # add activation
                x = self.activation(x)

                # add optinal dropout
                if self.dropout_rate > 0:
                    x = self.drop_list[i*self.num_convs + j](x)
                       
                # apply residual connection
                if j > 0:
                    x = x + x_copy
                    
            # add pooling layer
            if i != len(self.layers)-1:
                x = self.pool_list[i](x)
        
        return x