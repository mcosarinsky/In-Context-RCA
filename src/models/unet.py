import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )

def double_conv_batch_norm(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes=1, batch_norm=False):
        super().__init__()

        base = 32

        if batch_norm:
            conv_block = double_conv_batch_norm
        else:
            conv_block = double_conv

        self.dconv_down1 = conv_block(n_channels, base)
        self.dconv_down2 = conv_block(base, base*2)
        self.dconv_down3 = conv_block(base*2, base*4)
        self.dconv_down4 = conv_block(base*4, base*8)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = conv_block(base*4 + base*8, base*4)
        self.dconv_up2 = conv_block(base*2 + base*4, base*2)
        self.dconv_up1 = conv_block(base + base*2, base)

        self.conv_last = nn.Conv2d(base, n_classes, kernel_size=1)

        # Choose activation based on the number of classes
        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([conv3, x], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([conv2, x], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([conv1, x], dim=1)
        x = self.dconv_up1(x)

        x = self.conv_last(x)

        out = self.activation(x)

        return out