""" Full assembly of the parts to form the complete network """

from .unet_parts import *

class UNet(nn.Module):
    """
    U-Net architecture for image segmentation tasks.

    The U-Net consists of an encoder (contracting path) to capture context and a decoder (expanding path) 
    to enable precise localization. The architecture supports both standard and bilinear upsampling.

    Args:
        n_channels (int): Number of input channels (e.g., 3 for RGB images).
        n_classes (int): Number of output classes for segmentation.
        bilinear (bool): Whether to use bilinear upsampling instead of transposed convolutions (default: False).
    """
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initial double convolution block
        self.inc = (DoubleConv(n_channels, 64))

        # Downsampling (contracting path)
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        # Upsampling (expanding path)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        # Final output convolution to produce segmentation mask
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, height, width).

        Returns:
            torch.Tensor: Output tensor (logits) of shape (batch_size, n_classes, height, width).
        """
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """
        Enable gradient checkpointing to save memory during training.

        Gradient checkpointing trades computation for reduced memory usage by recomputing certain 
        parts of the forward pass during the backward pass.

        This can be useful when training large models on limited GPU memory.
        """
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)