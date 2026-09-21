"""
ResNet50-based Brain Tumor Segmentation + Classification Model
Architecture: ResNet50 encoder → U-Net decoder (segmentation) + classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_bn)
        self.bn   = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """Upsample + concatenate skip connection + two conv blocks."""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBlock(in_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNet50SegCls(nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.4):
        super().__init__()

        # ── Encoder (ResNet50 pretrained) ──────────────────────────────────
        # Skip connections matching TF layer names:
        #   s1  → conv1_relu   : 128×128 ×  64
        #   s2  → conv2_block3 :  64×64  × 256
        #   s3  → conv3_block4 :  32×32  × 512
        #   s4  → conv4_block6 :  16×16  × 1024
        #   btn → conv5_block3 :   8×8   × 2048
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # → 128×128×64
        self.pool = backbone.maxpool                                              # → 64×64×64
        self.enc1 = backbone.layer1   # → 64×64×256
        self.enc2 = backbone.layer2   # → 32×32×512
        self.enc3 = backbone.layer3   # → 16×16×1024
        self.enc4 = backbone.layer4   # →  8×8×2048

        # ── Classification head ────────────────────────────────────────────
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.cls_fc1  = nn.Linear(2048, 256)
        self.cls_drop = nn.Dropout(dropout_rate)
        self.cls_out  = nn.Linear(256, num_classes)

        # ── Segmentation decoder ───────────────────────────────────────────
        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512,  512,  256)
        self.dec2 = DecoderBlock(256,  256,  128)
        self.dec1 = DecoderBlock(128,   64,   64)
        self.dec0 = DecoderBlock(64,     0,   32)   # no skip at final stage

        self.seg_out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        # ── Encoder ────────────────────────────────────────────────────────
        s1 = self.enc0(x)        # 128×128×64
        p  = self.pool(s1)       #  64×64×64
        s2 = self.enc1(p)        #  64×64×256
        s3 = self.enc2(s2)       #  32×32×512
        s4 = self.enc3(s3)       #  16×16×1024
        bn = self.enc4(s4)       #   8×8×2048

        # ── Classification head ────────────────────────────────────────────
        cls = self.gap(bn).flatten(1)
        cls = F.relu(self.cls_fc1(cls))
        cls = self.cls_drop(cls)
        cls_output = self.cls_out(cls)              # raw logits → CrossEntropyLoss

        # ── Decoder ────────────────────────────────────────────────────────
        d4 = self.dec4(bn, s4)   # 16×16×512
        d3 = self.dec3(d4, s3)   # 32×32×256
        d2 = self.dec2(d3, s2)   # 64×64×128
        d1 = self.dec1(d2, s1)   # 128×128×64
        d0 = self.dec0(d1, None) # 256×256×32

        seg_output = torch.sigmoid(self.seg_out(d0))   # (B, 1, H, W)

        return seg_output, cls_output


def build_resnet50_segmentation_model(input_shape=(256, 256, 3), num_classes=4, dropout_rate=0.4):
    """Drop-in factory matching the original TF interface."""
    return ResNet50SegCls(num_classes=num_classes, dropout_rate=dropout_rate)


if __name__ == "__main__":
    model = build_resnet50_segmentation_model()
    x = torch.randn(2, 3, 256, 256)
    seg, cls = model(x)
    print("seg:", seg.shape)   # (2, 1, 256, 256)
    print("cls:", cls.shape)   # (2, 4)
    print("Parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")