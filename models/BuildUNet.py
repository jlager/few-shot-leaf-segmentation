import torch

class BuildUNet(torch.nn.Module):

    def __init__(
        self,
        layers: list,
        input_channels: int = 3,
        output_channels: int = 1,
        hidden_activation: torch.nn.Module = torch.nn.LeakyReLU(),
        output_activation: torch.nn.Module = torch.nn.Sigmoid(),
        dropout_rate: float = 0.0,
        num_convs: int = 3,
        verbose: bool = False,
        ):

        # initialize
        super().__init__()
        self.layers = layers
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.num_convs = num_convs
        self.verbose = verbose

        if self.verbose: print('-----------------'); print('initialization'); print('-----------------'); print()

        # encoder (conv + ... + conv + copy + pool)
        self.encoder_convs = torch.nn.ModuleList()
        self.encoder_pools = torch.nn.ModuleList()
        self.encoder_drops = torch.nn.ModuleList()
        self.encoder_norms = torch.nn.ModuleList()
        encoder_layers = [input_channels] + layers
        for i, (in_channels, out_channels) in enumerate(zip(encoder_layers[:-1], encoder_layers[1:])):
            for j in range(num_convs):
                if verbose: print(f'encoder conv {i}-{j}: {in_channels} -> {out_channels}')
                self.encoder_convs.append(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1))
                self.encoder_norms.append(torch.nn.BatchNorm2d(out_channels))
                self.encoder_drops.append(torch.nn.Dropout2d(dropout_rate))
                in_channels = out_channels
            if i < len(layers)-1:
                self.encoder_pools.append(torch.nn.MaxPool2d(2))
                if verbose: print(f'encoder pool {i}  : {out_channels}'); print()
        if verbose: print()

        # decoder (upsample + cat + conv + ... + conv)
        self.decoder_convs = torch.nn.ModuleList()
        self.decoder_ups = torch.nn.ModuleList()
        self.decoder_drops = torch.nn.ModuleList()
        self.decoder_norms = torch.nn.ModuleList()
        decoder_layers = layers[::-1]
        for i, (in_channels, out_channels) in enumerate(zip(decoder_layers[:-1], decoder_layers[1:])):
            self.decoder_ups.append(torch.nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2))
            if verbose: print(f'decoder up   {i}  : {in_channels} -> {out_channels} + {encoder_layers[-i-2]}')
            for j in range(num_convs):
                in_channels = out_channels + encoder_layers[-i-2] if j == 0 else out_channels
                if verbose: print(f'decoder conv {i}-{j}: {in_channels} -> {out_channels}')
                self.decoder_convs.append(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1))
                self.decoder_norms.append(torch.nn.BatchNorm2d(out_channels))
                self.decoder_drops.append(torch.nn.Dropout2d(dropout_rate))
                in_channels = out_channels
            if verbose: print()

        # output
        self.output_conv = torch.nn.Conv2d(decoder_layers[-1], output_channels, 1)
        if verbose: print(f'output conv: {decoder_layers[-1]} -> {output_channels}'); print()

    def forward(self, x):

        if self.verbose: print('-----------------'); print('forward'); print('-----------------'); print()

        # store encoder outputs before each pool
        x_cache = []

        # encoder
        if self.verbose: print(f'input: {x.shape}'); print()
        for i in range(len(self.layers)):
            for j in range(self.num_convs):
                x = self.encoder_convs[i*self.num_convs+j](x)
                x = self.encoder_norms[i*self.num_convs+j](x)
                x = self.hidden_activation(x)
                x = self.encoder_drops[i*self.num_convs+j](x)
                if self.verbose: print(f'encoder conv {i}-{j}: {x.shape}')
            x_cache.append(x)
            if i < len(self.layers) - 1:
                x = self.encoder_pools[i](x)
                if self.verbose: print(f'encoder pool {i}  : {x.shape}'); print()
        if self.verbose: print()

        # decoder
        for i in range(len(self.layers) - 1):
            x = self.decoder_ups[i](x)
            if self.verbose: print(f'decoder up   {i}  : {x.shape} + {x_cache[-i-2].shape}')
            x = torch.cat([x, x_cache[-i-2]], dim=1)
            for j in range(self.num_convs):
                x = self.decoder_convs[i*self.num_convs+j](x)
                x = self.decoder_norms[i*self.num_convs+j](x)
                x = self.hidden_activation(x)
                x = self.decoder_drops[i*self.num_convs+j](x)
                if self.verbose: print(f'decoder conv {i}-{j}: {x.shape}')
            if self.verbose: print()

        # output
        x = self.output_conv(x)
        x = self.output_activation(x)
        if self.verbose: print(f'output conv: {x.shape}'); print()

        return x
    
# test with verbose = True
if __name__ == '__main__':
    
    # options
    in_channels = 3
    out_channels = 1
    layers = [16, 32, 64, 128]

    # test
    unet = BuildUNet(layers, in_channels, out_channels, verbose=True)
    x = torch.rand(1, in_channels, 256, 256)
    y = unet(x)