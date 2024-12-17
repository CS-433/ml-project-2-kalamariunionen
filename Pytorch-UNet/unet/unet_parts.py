""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    A block of two consecutive convolutional layers, each followed by Batch Normalization and ReLU activation.

    Structure:
        (Conv2d -> BatchNorm -> ReLU) * 2

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of channels for the intermediate convolution. Defaults to `out_channels`.

    Example:
        >>> block = DoubleConv(3, 64)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> output = block(x)
        >>> output.shape
        torch.Size([1, 64, 256, 256])
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block: MaxPooling followed by a DoubleConv block.

    This reduces the spatial dimensions (height and width) by a factor of 2.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Example:
        >>> down = Down(64, 128)
        >>> x = torch.randn(1, 64, 256, 256)
        >>> output = down(x)
        >>> output.shape
        torch.Size([1, 128, 128, 128])
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block: Upsampling followed by a DoubleConv block with skip connections.

    Args:
        in_channels (int): Number of input channels (combined channels from upsampling and skip connection).
        out_channels (int): Number of output channels.
        bilinear (bool): Whether to use bilinear upsampling instead of transposed convolutions (default: True).

    Example:
        >>> up = Up(512, 256)
        >>> x1 = torch.randn(1, 512, 64, 64)  # From decoder
        >>> x2 = torch.randn(1, 256, 128, 128)  # From encoder (skip connection)
        >>> output = up(x1, x2)
        >>> output.shape
        torch.Size([1, 256, 128, 128])
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling or transposed convolution for upscaling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        """
        Forward pass for the Up block with skip connections.

        Args:
            x1 (torch.Tensor): Feature map from the decoder (upscaled).
            x2 (torch.Tensor): Feature map from the encoder (skip connection).

        Returns:
            torch.Tensor: Output feature map after concatenation and DoubleConv.
        """
        # Upsample x1 to match the size of x2
        x1 = self.up(x1)

        # Calculate the difference in spatial dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match the size of x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate x2 (skip connection) and x1 (upsampled)
        x = torch.cat([x2, x1], dim=1)

        # Apply DoubleConv
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final output convolution to produce the segmentation mask.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (number of segmentation classes).

    Example:
        >>> out_conv = OutConv(64, 5)  # 5 classes
        >>> x = torch.randn(1, 64, 256, 256)
        >>> output = out_conv(x)
        >>> output.shape
        torch.Size([1, 5, 256, 256])
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass for the output convolution.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output segmentation map (logits).
        """
        return self.conv(x)
