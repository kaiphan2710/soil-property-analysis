"""
ResNet-based models for hyperspectral microplastic training.

The key change from normal RGB ResNet is the first convolution:
    normal ResNet:          Conv2d(3, 64, kernel_size=7, ...)
    hyperspectral ResNet:   Conv2d(633, 64, kernel_size=7, ...)

This file includes:
    1. HyperspectralResNetClassifier
       For image/patch-level classification.

    2. HyperspectralResNetSegmenter
       For dish-level semantic segmentation:
           output class mask with 0=background, 1=PP, 2=PET, 3=PS.

For the current project, the segmenter is the more useful model because the goal
is auto-labelling microplastic locations inside each dish.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


ResNetName = Literal["resnet18", "resnet34", "resnet50"]


def _build_rgb_resnet(name: ResNetName, pretrained: bool = False):
    """
    Builds a torchvision ResNet.

    pretrained=False is the safest default for hyperspectral input because RGB
    pretrained weights were learned for 3 channels, not 633 channels.
    """
    if name == "resnet18":
        return resnet18(weights="DEFAULT" if pretrained else None)
    if name == "resnet34":
        return resnet34(weights="DEFAULT" if pretrained else None)
    if name == "resnet50":
        return resnet50(weights="DEFAULT" if pretrained else None)
    raise ValueError(f"Unsupported ResNet name: {name}")


def replace_first_conv(
    model: nn.Module,
    in_channels: int,
    strategy: Literal["random", "average_rgb"] = "average_rgb",
) -> nn.Module:
    """
    Replaces ResNet's first conv so it accepts hyperspectral channels.

    If pretrained RGB weights exist and strategy='average_rgb', the RGB filters
    are averaged and repeated across hyperspectral channels. This gives a stable
    starting point, but it is not true hyperspectral pretraining.
    """
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        if strategy == "average_rgb" and old_conv.weight.shape[1] == 3:
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1) / max(1, in_channels / 3))
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

    model.conv1 = new_conv
    return model


def _num_groups_for_channels(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def replace_batchnorm_with_groupnorm(module: nn.Module) -> nn.Module:
    """
    BatchNorm can fail with batch_size=1 when feature maps become 1x1.
    GroupNorm is more stable for this project because we train large
    hyperspectral images one at a time on a laptop.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            setattr(
                module,
                name,
                nn.GroupNorm(
                    num_groups=_num_groups_for_channels(channels),
                    num_channels=channels,
                ),
            )
        else:
            replace_batchnorm_with_groupnorm(child)
    return module


class HyperspectralResNetClassifier(nn.Module):
    """
    ResNet classifier for hyperspectral images or patches.

    Input:
        x: FloatTensor (batch, bands, height, width)

    Output:
        logits: FloatTensor (batch, num_classes)

    Use this if each training sample has one class label. For auto-labelling
    locations, this is less direct unless you train on patches and scan the dish.
    """

    def __init__(
        self,
        in_channels: int = 633,
        num_classes: int = 4,
        resnet_name: ResNetName = "resnet18",
        pretrained_rgb: bool = False,
    ):
        super().__init__()
        model = _build_rgb_resnet(resnet_name, pretrained=pretrained_rgb)
        model = replace_first_conv(
            model,
            in_channels=in_channels,
            strategy="average_rgb" if pretrained_rgb else "random",
        )
        model = replace_batchnorm_with_groupnorm(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HyperspectralResNetEncoder(nn.Module):
    """
    ResNet encoder that exposes intermediate feature maps for segmentation.
    """

    def __init__(
        self,
        in_channels: int = 633,
        resnet_name: ResNetName = "resnet18",
        pretrained_rgb: bool = False,
    ):
        super().__init__()
        model = _build_rgb_resnet(resnet_name, pretrained=pretrained_rgb)
        model = replace_first_conv(
            model,
            in_channels=in_channels,
            strategy="average_rgb" if pretrained_rgb else "random",
        )
        model = replace_batchnorm_with_groupnorm(model)

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if resnet_name in ("resnet18", "resnet34"):
            self.out_channels: Dict[str, int] = {
                "c1": 64,
                "c2": 64,
                "c3": 128,
                "c4": 256,
                "c5": 512,
            }
        else:
            self.out_channels = {
                "c1": 64,
                "c2": 256,
                "c3": 512,
                "c4": 1024,
                "c5": 2048,
            }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c1 = self.stem(x)             # /2
        x = self.maxpool(c1)          # /4
        c2 = self.layer1(x)           # /4
        c3 = self.layer2(c2)          # /8
        c4 = self.layer3(c3)          # /16
        c5 = self.layer4(c4)          # /32
        return {"c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}


class HyperspectralResNetSegmenter(nn.Module):
    """
    ResNet-FPN style semantic segmentation model for hyperspectral dish crops.

    Input:
        x: FloatTensor (batch, bands, height, width)

    Output:
        logits: FloatTensor (batch, num_classes, height, width)

    num_classes should include background:
        0=background, 1=PP, 2=PET, 3=PS
    """

    def __init__(
        self,
        in_channels: int = 633,
        num_classes: int = 4,
        resnet_name: ResNetName = "resnet18",
        fpn_channels: int = 128,
        pretrained_rgb: bool = False,
    ):
        super().__init__()
        self.encoder = HyperspectralResNetEncoder(
            in_channels=in_channels,
            resnet_name=resnet_name,
            pretrained_rgb=pretrained_rgb,
        )
        channels = self.encoder.out_channels

        self.lateral_c2 = nn.Conv2d(channels["c2"], fpn_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(channels["c3"], fpn_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(channels["c4"], fpn_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(channels["c5"], fpn_channels, kernel_size=1)

        self.smooth = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups_for_channels(fpn_channels), fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups_for_channels(fpn_channels), fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feats = self.encoder(x)

        p5 = self.lateral_c5(feats["c5"])
        p4 = self.lateral_c4(feats["c4"]) + F.interpolate(
            p5, size=feats["c4"].shape[-2:], mode="bilinear", align_corners=False
        )
        p3 = self.lateral_c3(feats["c3"]) + F.interpolate(
            p4, size=feats["c3"].shape[-2:], mode="bilinear", align_corners=False
        )
        p2 = self.lateral_c2(feats["c2"]) + F.interpolate(
            p3, size=feats["c2"].shape[-2:], mode="bilinear", align_corners=False
        )

        x = self.smooth(p2)
        x = self.classifier(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test hyperspectral ResNet model shapes.")
    parser.add_argument("--bands", type=int, default=633)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--task", choices=["classification", "segmentation"], default="segmentation")
    parser.add_argument("--resnet", choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    args = parser.parse_args()

    x = torch.randn(args.batch_size, args.bands, args.height, args.width)

    if args.task == "classification":
        model = HyperspectralResNetClassifier(
            in_channels=args.bands,
            num_classes=4,
            resnet_name=args.resnet,
        )
    else:
        model = HyperspectralResNetSegmenter(
            in_channels=args.bands,
            num_classes=4,
            resnet_name=args.resnet,
        )

    with torch.no_grad():
        y = model(x)

    print("input shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("trainable parameters:", count_parameters(model))


if __name__ == "__main__":
    main()
