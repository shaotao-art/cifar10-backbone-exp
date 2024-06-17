import torch
from torch import nn

from typing import List, Tuple, Dict, Union



def act():
    return nn.ReLU(inplace=True)

def norm(**kwargs):
    return nn.BatchNorm2d(**kwargs)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm(num_features=out_channels),
            act()
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        
        self.skip_connection = nn.Conv2d(in_channels, out_channels, 3, 1, 1) if in_channels!= out_channels else nn.Identity()
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + self.skip_connection(residual)
    
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, 3, 1, 1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, 
                 channels=List[int], 
                 num_res_block_per_stage=Union[List[int], int], 
                 in_channel=3):
        super(Encoder, self).__init__()
        num_stages = len(channels) - 1
        if isinstance(num_res_block_per_stage, int):
            num_res_block_per_stage = [num_res_block_per_stage for _ in range(num_stages)]
        assert len(channels) == len(num_res_block_per_stage) + 1
        
        self.init_conv = nn.Conv2d(in_channel, channels[0], 3, 1, 1)
        
        self.encode_layers = nn.ModuleList()
        for i in range(num_stages):
            layer = nn.ModuleDict()
            res_layer = nn.Sequential()
            res_layer.append(ResidualBlock(channels[i], channels[i + 1]))
            for _ in range(num_res_block_per_stage[i] - 1):
                res_layer.append(ResidualBlock(channels[i + 1], channels[i + 1]))
            layer['res_layers'] = res_layer
            
            if i != num_stages - 1:
                layer['downsample'] = Downsample(channels[i + 1], channels[i + 1])
            else:
                layer['downsample'] = nn.Identity()
            
            self.encode_layers.append(layer)
    
    def forward(self, x):
        x = self.init_conv(x)
        for l in self.encode_layers:
            x = l['res_layers'](x)
            x = l['downsample'](x)
        return x


class Decoder(nn.Module):
    def __init__(self, 
                 channels=List[int], 
                 num_res_block_per_stage=Union[List[int], int],
                 use_in_unet=False
                 ):
        super(Decoder, self).__init__()
        num_stages = len(channels) - 1
        if isinstance(num_res_block_per_stage, int):
            num_res_block_per_stage = [num_res_block_per_stage for _ in range(num_stages)]
        assert len(channels) == len(num_res_block_per_stage) + 1
        
        self.decode_layers = nn.ModuleList()
        for i in range(num_stages):
            layer = nn.ModuleDict()
            res_layer = nn.Sequential()
            res_layer.append(ResidualBlock(channels[i] * 2 if use_in_unet else channels[i], channels[i + 1]))
            for _ in range(num_res_block_per_stage[i] - 1):
                res_layer.append(ResidualBlock(channels[i + 1], channels[i + 1]))
            layer['res_layers'] = res_layer
            
            if i != num_stages - 1:
                layer['upsample'] = Upsample(channels[i + 1], channels[i + 1])
            else:
                layer['upsample'] = nn.Identity()
            
            self.decode_layers.append(layer)
    
    def forward(self, x):
        for l in self.decode_layers:
            x = l['res_layers'](x)
            x = l['upsample'](x)
        return x
    
    

class EncoderDecoder(nn.Module):
    def __init__(self, 
                 channels=List[int], 
                 num_res_block_per_stage=Union[List[int], int],
                 in_channel=3):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(channels=channels, 
                               num_res_block_per_stage=num_res_block_per_stage, 
                               in_channel=in_channel)
        self.decoder = Decoder(channels=list(reversed(channels)), 
                               num_res_block_per_stage=list(reversed(num_res_block_per_stage)),
                               use_in_unet=True)
        
        
    
    def forward(self, x):
        x = self.encoder.init_conv(x)
        hiddens = []
        for l in self.encoder.encode_layers:
            x = l['res_layers'](x)
            hiddens.append(x)
            x = l['downsample'](x)
            
        print('hiddens: ')
        for h in hiddens:
            print(h.shape)
        
        for l in self.decoder.decode_layers:
            print(x.shape, hiddens[-1].shape)
            x = torch.cat([x, hiddens.pop()], dim=1)
            x = l['res_layers'](x)
            x = l['upsample'](x)
        return x