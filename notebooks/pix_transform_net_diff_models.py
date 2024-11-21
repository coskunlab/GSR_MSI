import torch
import torch.nn as nn

class PixTransformNetBase(nn.Module):
    def __init__(self, channels_in=5, kernel_size=1, weights_regularizer=None):
        super(PixTransformNetBase, self).__init__()
        self.channels_in = channels_in

        self.spatial_net = nn.Sequential(
            nn.Conv2d(2, 32, (1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 2048, (kernel_size, kernel_size), padding=(kernel_size-1)//2)
        )
        self.color_net = nn.Sequential(
            nn.Conv2d(channels_in-2, 32, (1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 2048, (kernel_size, kernel_size), padding=(kernel_size-1)//2)
        )
        self.head_net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2048, 32, (kernel_size, kernel_size), padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Conv2d(32, 1, (1, 1), padding=0)
        )

        if weights_regularizer is None:
            reg_spatial = 0.0001
            reg_color = 0.001
            reg_head = 0.0001
        else:
            reg_spatial = weights_regularizer[0]
            reg_color = weights_regularizer[1]
            reg_head = weights_regularizer[2]

        self.params_with_regularizer = []
        self.params_with_regularizer += [{'params': self.spatial_net.parameters(), 'weight_decay': reg_spatial}]
        self.params_with_regularizer += [{'params': self.color_net.parameters(), 'weight_decay': reg_color}]
        self.params_with_regularizer += [{'params': self.head_net.parameters(), 'weight_decay': reg_head}]

    def forward(self, input):
        input_spatial = input[:, self.channels_in-2:, :, :]
        input_color = input[:, 0:self.channels_in-2, :, :]
        merged_features = self.spatial_net(input_spatial) + self.color_net(input_color)
        
        return self.head_net(merged_features)
        
#################
class PixTransformNetDeeper(PixTransformNetBase):
    def __init__(self, channels_in=5, kernel_size=1, weights_regularizer=None):
        super(PixTransformNetDeeper, self).__init__(channels_in, kernel_size, weights_regularizer)
        
        # Adding more layers to spatial_net and color_net
        self.spatial_net = nn.Sequential(
            nn.Conv2d(2, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2048, (kernel_size, kernel_size), padding=(kernel_size-1)//2)
        )
        self.color_net = nn.Sequential(
            nn.Conv2d(channels_in-2, 64, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 2048, (kernel_size, kernel_size), padding=(kernel_size-1)//2)
        )

#################
class PixTransformNetAttention(PixTransformNetBase):
    def __init__(self, channels_in=5, kernel_size=1, weights_regularizer=None):
        super(PixTransformNetAttention, self).__init__(channels_in, kernel_size, weights_regularizer)

        # Adding channel attention
        self.channel_attention = nn.Sequential(
            nn.Conv2d(2048, 512, (1, 1), padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 2048, (1, 1), padding=0),
            nn.Sigmoid()
        )
        self.attention_weights = None  # To store the attention weights

    def forward(self, input):
        input_spatial = input[:, self.channels_in-2:, :, :]
        input_color = input[:, 0:self.channels_in-2, :, :]
        merged_features = self.spatial_net(input_spatial) + self.color_net(input_color)
        
        # Apply channel attention and store it
        self.attention_weights = self.channel_attention(merged_features)
        merged_features = merged_features * self.attention_weights
        
        return self.head_net(merged_features)

#################
class PixTransformNetMultiScale(PixTransformNetBase):
    def __init__(self, channels_in=5, kernel_size=1, weights_regularizer=None):
        super(PixTransformNetMultiScale, self).__init__(channels_in, kernel_size, weights_regularizer)
        
        # Multi-scale feature extraction with different kernel sizes
        self.spatial_net_small = nn.Sequential(
            nn.Conv2d(2, 32, (3, 3), padding=1),
            nn.ReLU()
        )
        self.spatial_net_large = nn.Sequential(
            nn.Conv2d(2, 32, (5, 5), padding=2),
            nn.ReLU()
        )
        self.color_net_small = nn.Sequential(
            nn.Conv2d(channels_in-2, 32, (3, 3), padding=1),
            nn.ReLU()
        )
        self.color_net_large = nn.Sequential(
            nn.Conv2d(channels_in-2, 32, (5, 5), padding=2),
            nn.ReLU()
        )
        
        self.head_net = nn.Sequential(
            nn.Conv2d(128, 32, (kernel_size, kernel_size), padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Conv2d(32, 1, (1, 1), padding=0)
        )

    def forward(self, input):
        input_spatial = input[:, self.channels_in-2:, :, :]
        input_color = input[:, 0:self.channels_in-2, :, :]

        spatial_small = self.spatial_net_small(input_spatial)
        spatial_large = self.spatial_net_large(input_spatial)
        color_small = self.color_net_small(input_color)
        color_large = self.color_net_large(input_color)

        merged_features = torch.cat([spatial_small, spatial_large, color_small, color_large], dim=1)
        
        return self.head_net(merged_features)

#################
class PixTransformNetResidual(PixTransformNetBase):
    def __init__(self, channels_in=5, kernel_size=1, weights_regularizer=None):
        super(PixTransformNetResidual, self).__init__()

        self.channels_in = channels_in
        
        # Adding residual connections in the networks
        self.spatial_net = nn.Sequential(
            nn.Conv2d(2, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.ReLU()
        )
        self.color_net = nn.Sequential(
            nn.Conv2d(channels_in-2, 32, (3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.ReLU()
        )
        
        # 1x1 conv to match the channel dimensions for residual connections
        self.match_spatial_channels = nn.Conv2d(2, 32, kernel_size=1)
        self.match_color_channels = nn.Conv2d(channels_in-2, 32, kernel_size=1)

        self.head_net = nn.Sequential(
            nn.Conv2d(32, 1, (kernel_size, kernel_size), padding=(kernel_size-1)//2)
        )

    def forward(self, input):
        input_spatial = input[:, self.channels_in-2:, :, :]
        input_color = input[:, 0:self.channels_in-2, :, :]

        # Match the channel dimensions for residual connection
        input_spatial_res = self.match_spatial_channels(input_spatial)
        input_color_res = self.match_color_channels(input_color)

        spatial_features = self.spatial_net(input_spatial) + input_spatial_res
        color_features = self.color_net(input_color) + input_color_res

        merged_features = spatial_features + color_features
        
        return self.head_net(merged_features)

