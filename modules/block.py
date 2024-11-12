import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from modules.utils import multi_scale_deformable_attn_pytorch, _get_clones, inverse_sigmoid

# __all__ = (
#     "Conv",
#     "Conv2",
#     "LightConv",
#     "DWConv",
#     "DWConvTranspose2d",
#     "ConvTranspose",
#     "Focus",
#     "GhostConv",
#     "ChannelAttention",
#     "SpatialAttention",
#     "CBAM",
#     "Concat",
#     "RepConv",
# )

#ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py

# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs. Automatically calculates padding to ensure 'same' convolution"""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p


# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(x))


# class Conv2(Conv):
#     """Simplified RepConv module with Conv fusing."""

#     def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
#         self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x) + self.cv2(x)))

#     def forward_fuse(self, x):
#         """Apply fused convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))

#     def fuse_convs(self):
#         """Fuse parallel convolutions."""
#         w = torch.zeros_like(self.conv.weight.data)
#         i = [x // 2 for x in w.shape[2:]]
#         w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
#         self.conv.weight.data += w
#         self.__delattr__("cv2")
#         self.forward = self.forward_fuse


# class LightConv(nn.Module):
#     """
#     Light convolution with args(ch_in, ch_out, kernel).

#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
#     """

#     def __init__(self, c1, c2, k=1, act=nn.ReLU()):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv1 = Conv(c1, c2, 1, act=False)
#         self.conv2 = DWConv(c2, c2, k, act=act)

#     def forward(self, x):
#         """Apply 2 convolutions to input tensor."""
#         return self.conv2(self.conv1(x))


# class DWConv(Conv):
#     """Depth-wise convolution."""

#     def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
#         """Initialize Depth-wise convolution with given parameters."""
#         super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# class DWConvTranspose2d(nn.ConvTranspose2d):
#     """Depth-wise transpose convolution."""

#     def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
#         """Initialize DWConvTranspose2d class with given parameters."""
#         super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


# class ConvTranspose(nn.Module):
#     """Convolution transpose 2d layer."""

#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
#         """Initialize ConvTranspose2d layer with batch normalization and activation function."""
#         super().__init__()
#         self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
#         self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Applies transposed convolutions, batch normalization and activation to input."""
#         return self.act(self.bn(self.conv_transpose(x)))

#     def forward_fuse(self, x):
#         """Applies activation and convolution transpose operation to input."""
#         return self.act(self.conv_transpose(x))


# class Focus(nn.Module):
#     """Focus wh information into c-space."""

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
#         """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
#         super().__init__()
#         self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
#         # self.contract = Contract(gain=2)

#     def forward(self, x):
#         """
#         Applies convolution to concatenated tensor and returns the output.

#         Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
#         """
#         return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
#         # return self.conv(self.contract(x))


# class GhostConv(nn.Module):
#     """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

#     def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
#         """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
#         super().__init__()
#         c_ = c2 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
#         self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

#     def forward(self, x):
#         """Forward propagation through a Ghost Bottleneck layer with skip connection."""
#         y = self.cv1(x)
#         return torch.cat((y, self.cv2(y)), 1)


# class RepConv(nn.Module):
#     """
#     RepConv is a basic rep-style block, including training and deploy status.

#     This module is used in RT-DETR.
#     Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
#     """

#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
#         """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
#         super().__init__()
#         assert k == 3 and p == 1
#         self.g = g
#         self.c1 = c1
#         self.c2 = c2
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#         self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
#         self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
#         self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

#     def forward_fuse(self, x):
#         """Forward process."""
#         return self.act(self.conv(x))

#     def forward(self, x):
#         """Forward process."""
#         id_out = 0 if self.bn is None else self.bn(x)
#         return self.act(self.conv1(x) + self.conv2(x) + id_out)

#     def get_equivalent_kernel_bias(self):
#         """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
#         kernelid, biasid = self._fuse_bn_tensor(self.bn)
#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

#     @staticmethod
#     def _pad_1x1_to_3x3_tensor(kernel1x1):
#         """Pads a 1x1 tensor to a 3x3 tensor."""
#         if kernel1x1 is None:
#             return 0
#         else:
#             return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

#     def _fuse_bn_tensor(self, branch):
#         """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
#         if branch is None:
#             return 0, 0
#         if isinstance(branch, Conv):
#             kernel = branch.conv.weight
#             running_mean = branch.bn.running_mean
#             running_var = branch.bn.running_var
#             gamma = branch.bn.weight
#             beta = branch.bn.bias
#             eps = branch.bn.eps
#         elif isinstance(branch, nn.BatchNorm2d):
#             if not hasattr(self, "id_tensor"):
#                 input_dim = self.c1 // self.g
#                 kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
#                 for i in range(self.c1):
#                     kernel_value[i, i % input_dim, 1, 1] = 1
#                 self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
#             kernel = self.id_tensor
#             running_mean = branch.running_mean
#             running_var = branch.running_var
#             gamma = branch.weight
#             beta = branch.bias
#             eps = branch.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std

#     def fuse_convs(self):
#         """Combines two convolution layers into a single layer and removes unused attributes from the class."""
#         if hasattr(self, "conv"):
#             return
#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.conv = nn.Conv2d(
#             in_channels=self.conv1.conv.in_channels,
#             out_channels=self.conv1.conv.out_channels,
#             kernel_size=self.conv1.conv.kernel_size,
#             stride=self.conv1.conv.stride,
#             padding=self.conv1.conv.padding,
#             dilation=self.conv1.conv.dilation,
#             groups=self.conv1.conv.groups,
#             bias=True,
#         ).requires_grad_(False)
#         self.conv.weight.data = kernel
#         self.conv.bias.data = bias
#         for para in self.parameters():
#             para.detach_()
#         self.__delattr__("conv1")
#         self.__delattr__("conv2")
#         if hasattr(self, "nm"):
#             self.__delattr__("nm")
#         if hasattr(self, "bn"):
#             self.__delattr__("bn")
#         if hasattr(self, "id_tensor"):
#             self.__delattr__("id_tensor")


# class ChannelAttention(nn.Module):
#     """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

#     def __init__(self, channels: int) -> None:
#         """Initializes the class and sets the basic configurations and instance variables required."""
#         super().__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
#         self.act = nn.Sigmoid()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
#         return x * self.act(self.fc(self.pool(x)))


# class SpatialAttention(nn.Module):
#     """Spatial-attention module."""

#     def __init__(self, kernel_size=7):
#         """Initialize Spatial-attention module with kernel size argument."""
#         super().__init__()
#         assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
#         padding = 3 if kernel_size == 7 else 1
#         self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.act = nn.Sigmoid()

#     def forward(self, x):
#         """Apply channel and spatial attention on input for feature recalibration."""
#         return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


# class CBAM(nn.Module):
#     """Convolutional Block Attention Module."""

#     def __init__(self, c1, kernel_size=7):
#         """Initialize CBAM with given input channel (c1) and kernel size."""
#         super().__init__()
#         self.channel_attention = ChannelAttention(c1)
#         self.spatial_attention = SpatialAttention(kernel_size)

#     def forward(self, x):
#         """Applies the forward pass through C1 module."""
#         return self.spatial_attention(self.channel_attention(x))


# class Concat(nn.Module):
#     """Concatenate a list of tensors along dimension."""

#     def __init__(self, dimension=1):
#         """Concatenates a list of tensors along a specified dimension."""
#         super().__init__()
#         self.d = dimension

#     def forward(self, x):
#         """Forward pass for the YOLOv8 mask Proto module."""
#         return torch.cat(x, self.d)


# #ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py

# class DFL(nn.Module):
#     """
#     Integral module of Distribution Focal Loss (DFL).

#     Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
#     """

#     def __init__(self, c1=16):
#         """Initialize a convolutional layer with a given number of input channels."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
#         x = torch.arange(c1, dtype=torch.float)
#         self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
#         self.c1 = c1

#     def forward(self, x):
#         """Applies a transformer layer on input tensor 'x' and returns a tensor."""
#         b, _, a = x.shape  # batch, channels, anchors
#         return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
#         # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


# class Proto(nn.Module):
#     """YOLOv8 mask Proto module for segmentation models."""

#     def __init__(self, c1, c_=256, c2=32):
#         """
#         Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

#         Input arguments are ch_in, number of protos, number of masks.
#         """
#         super().__init__()
#         self.cv1 = Conv(c1, c_, k=3)
#         self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
#         self.cv2 = Conv(c_, c_, k=3)
#         self.cv3 = Conv(c_, c2)

#     def forward(self, x):
#         """Performs a forward pass through layers using an upsampled input image."""
#         return self.cv3(self.cv2(self.upsample(self.cv1(x))))


# class HGStem(nn.Module):
#     """
#     StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
#     """

#     def __init__(self, c1, cm, c2):
#         """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
#         super().__init__()
#         self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
#         self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
#         self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
#         self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
#         self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

#     def forward(self, x):
#         """Forward pass of a PPHGNetV2 backbone layer."""
#         x = self.stem1(x)
#         x = F.pad(x, [0, 1, 0, 1])
#         x2 = self.stem2a(x)
#         x2 = F.pad(x2, [0, 1, 0, 1])
#         x2 = self.stem2b(x2)
#         x1 = self.pool(x)
#         x = torch.cat([x1, x2], dim=1)
#         x = self.stem3(x)
#         x = self.stem4(x)
#         return x


# class HGBlock(nn.Module):
#     """
#     HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

#     https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
#     """

#     def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
#         """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
#         super().__init__()
#         block = LightConv if lightconv else Conv
#         self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
#         self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
#         self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         """Forward pass of a PPHGNetV2 backbone layer."""
#         y = [x]
#         y.extend(m(y[-1]) for m in self.m)
#         y = self.ec(self.sc(torch.cat(y, 1)))
#         return y + x if self.add else y


# class SPP(nn.Module):
#     """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

#     def __init__(self, c1, c2, k=(5, 9, 13)):
#         """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

#     def forward(self, x):
#         """Forward pass of the SPP layer, performing spatial pyramid pooling."""
#         x = self.cv1(x)
#         return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


# class SPPF(nn.Module):
#     """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

#     def __init__(self, c1, c2, k=5):
#         """
#         Initializes the SPPF layer with given input/output channels and kernel size.

#         This module is equivalent to SPP(k=(5, 9, 13)).
#         """
#         super().__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * 4, c2, 1, 1)
#         self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

#     def forward(self, x):
#         """Forward pass through Ghost Convolution block."""
#         y = [self.cv1(x)]
#         y.extend(self.m(y[-1]) for _ in range(3))
#         return self.cv2(torch.cat(y, 1))


# class C1(nn.Module):
#     """CSP Bottleneck with 1 convolution."""

#     def __init__(self, c1, c2, n=1):
#         """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
#         super().__init__()
#         self.cv1 = Conv(c1, c2, 1, 1)
#         self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

#     def forward(self, x):
#         """Applies cross-convolutions to input in the C3 module."""
#         y = self.cv1(x)
#         return self.m(y) + y


# class C2(nn.Module):
#     """CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
#         # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
#         self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

#     def forward(self, x):
#         """Forward pass through the CSP bottleneck with 2 convolutions."""
#         a, b = self.cv1(x).chunk(2, 1)
#         return self.cv2(torch.cat((self.m(a), b), 1))


# class C2f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))


# class C3(nn.Module):
#     """CSP Bottleneck with 3 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

#     def forward(self, x):
#         """Forward pass through the CSP bottleneck with 2 convolutions."""
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# class C3x(C3):
#     """C3 module with cross-convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initialize C3TR instance and set default parameters."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.c_ = int(c2 * e)
#         self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


# class RepC3(nn.Module):
#     """Rep C3."""

#     def __init__(self, c1, c2, n=3, e=1.0):
#         """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c2, 1, 1)
#         self.cv2 = Conv(c1, c2, 1, 1)
#         self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
#         self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

#     def forward(self, x):
#         """Forward pass of RT-DETR neck layer."""
#         return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


# class C3TR(C3):
#     """C3 module with TransformerBlock()."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initialize C3Ghost module with GhostBottleneck()."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)
#         self.m = TransformerBlock(c_, c_, 4, n)


# class C3Ghost(C3):
#     """C3 module with GhostBottleneck()."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


# class GhostBottleneck(nn.Module):
#     """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

#     def __init__(self, c1, c2, k=3, s=1):
#         """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
#         super().__init__()
#         c_ = c2 // 2
#         self.conv = nn.Sequential(
#             GhostConv(c1, c_, 1, 1),  # pw
#             DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
#             GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
#         )
#         self.shortcut = (
#             nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
#         )

#     def forward(self, x):
#         """Applies skip connection and concatenation to input tensor."""
#         return self.conv(x) + self.shortcut(x)


# class Bottleneck(nn.Module):
#     """Standard bottleneck."""

#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         """Applies the YOLO FPN to input data."""
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# class BottleneckCSP(nn.Module):
#     """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
#         self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
#         self.act = nn.SiLU()
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

#     def forward(self, x):
#         """Applies a CSP bottleneck with 3 convolutions."""
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


# class ResNetBlock(nn.Module):
#     """ResNet block with standard convolution layers."""

#     def __init__(self, c1, c2, s=1, e=4):
#         """Initialize convolution with given parameters."""
#         super().__init__()
#         c3 = e * c2
#         self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
#         self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
#         self.cv3 = Conv(c2, c3, k=1, act=False)
#         self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

#     def forward(self, x):
#         """Forward pass through the ResNet block."""
#         return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


# class ResNetLayer(nn.Module):
#     """ResNet layer with multiple ResNet blocks."""

#     def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
#         """Initializes the ResNetLayer given arguments."""
#         super().__init__()
#         self.is_first = is_first

#         if self.is_first:
#             self.layer = nn.Sequential(
#                 Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#             )
#         else:
#             blocks = [ResNetBlock(c1, c2, s, e=e)]
#             blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
#             self.layer = nn.Sequential(*blocks)

#     def forward(self, x):
#         """Forward pass through the ResNet layer."""
#         return self.layer(x)


# class MaxSigmoidAttnBlock(nn.Module):
#     """Max Sigmoid attention block."""

#     def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
#         """Initializes MaxSigmoidAttnBlock with specified arguments."""
#         super().__init__()
#         self.nh = nh
#         self.hc = c2 // nh
#         self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
#         self.gl = nn.Linear(gc, ec)
#         self.bias = nn.Parameter(torch.zeros(nh))
#         self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
#         self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

#     def forward(self, x, guide):
#         """Forward process."""
#         bs, _, h, w = x.shape

#         guide = self.gl(guide)
#         guide = guide.view(bs, -1, self.nh, self.hc)
#         embed = self.ec(x) if self.ec is not None else x
#         embed = embed.view(bs, self.nh, self.hc, h, w)

#         aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
#         aw = aw.max(dim=-1)[0]
#         aw = aw / (self.hc**0.5)
#         aw = aw + self.bias[None, :, None, None]
#         aw = aw.sigmoid() * self.scale

#         x = self.proj_conv(x)
#         x = x.view(bs, self.nh, -1, h, w)
#         x = x * aw.unsqueeze(2)
#         return x.view(bs, -1, h, w)


# class C2fAttn(nn.Module):
#     """C2f module with an additional attn module."""

#     def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
#         """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#         self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

#     def forward(self, x, guide):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         y.append(self.attn(y[-1], guide))
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x, guide):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         y.append(self.attn(y[-1], guide))
#         return self.cv2(torch.cat(y, 1))


# class ImagePoolingAttn(nn.Module):
#     """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

#     def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
#         """Initializes ImagePoolingAttn with specified arguments."""
#         super().__init__()

#         nf = len(ch)
#         self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
#         self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
#         self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
#         self.proj = nn.Linear(ec, ct)
#         self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
#         self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
#         self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
#         self.ec = ec
#         self.nh = nh
#         self.nf = nf
#         self.hc = ec // nh
#         self.k = k

#     def forward(self, x, text):
#         """Executes attention mechanism on input tensor x and guide tensor."""
#         bs = x[0].shape[0]
#         assert len(x) == self.nf
#         num_patches = self.k**2
#         x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
#         x = torch.cat(x, dim=-1).transpose(1, 2)
#         q = self.query(text)
#         k = self.key(x)
#         v = self.value(x)

#         # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
#         q = q.reshape(bs, -1, self.nh, self.hc)
#         k = k.reshape(bs, -1, self.nh, self.hc)
#         v = v.reshape(bs, -1, self.nh, self.hc)

#         aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
#         aw = aw / (self.hc**0.5)
#         aw = F.softmax(aw, dim=-1)

#         x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
#         x = self.proj(x.reshape(bs, -1, self.ec))
#         return x * self.scale + text


# class ContrastiveHead(nn.Module):
#     """Implements contrastive learning head for region-text similarity in vision-language models."""

#     def __init__(self):
#         """Initializes ContrastiveHead with specified region-text similarity parameters."""
#         super().__init__()
#         # NOTE: use -10.0 to keep the init cls loss consistency with other losses
#         self.bias = nn.Parameter(torch.tensor([-10.0]))
#         self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

#     def forward(self, x, w):
#         """Forward function of contrastive learning."""
#         x = F.normalize(x, dim=1, p=2)
#         w = F.normalize(w, dim=-1, p=2)
#         x = torch.einsum("bchw,bkc->bkhw", x, w)
#         return x * self.logit_scale.exp() + self.bias


# class BNContrastiveHead(nn.Module):
#     """
#     Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

#     Args:
#         embed_dims (int): Embed dimensions of text and image features.
#     """

#     def __init__(self, embed_dims: int):
#         """Initialize ContrastiveHead with region-text similarity parameters."""
#         super().__init__()
#         self.norm = nn.BatchNorm2d(embed_dims)
#         # NOTE: use -10.0 to keep the init cls loss consistency with other losses
#         self.bias = nn.Parameter(torch.tensor([-10.0]))
#         # use -1.0 is more stable
#         self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

#     def forward(self, x, w):
#         """Forward function of contrastive learning."""
#         x = self.norm(x)
#         w = F.normalize(w, dim=-1, p=2)
#         x = torch.einsum("bchw,bkc->bkhw", x, w)
#         return x * self.logit_scale.exp() + self.bias


# class RepBottleneck(Bottleneck):
#     """Rep bottleneck."""

#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
#         super().__init__(c1, c2, shortcut, g, k, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = RepConv(c1, c_, k[0], 1)


# class RepCSP(C3):
#     """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


# class RepNCSPELAN4(nn.Module):
#     """CSP-ELAN."""

#     def __init__(self, c1, c2, c3, c4, n=1):
#         """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
#         super().__init__()
#         self.c = c3 // 2
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
#         self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
#         self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

#     def forward(self, x):
#         """Forward pass through RepNCSPELAN4 layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
#         return self.cv4(torch.cat(y, 1))

#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
#         return self.cv4(torch.cat(y, 1))


# class ELAN1(RepNCSPELAN4):
#     """ELAN1 module with 4 convolutions."""

#     def __init__(self, c1, c2, c3, c4):
#         """Initializes ELAN1 layer with specified channel sizes."""
#         super().__init__(c1, c2, c3, c4)
#         self.c = c3 // 2
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = Conv(c3 // 2, c4, 3, 1)
#         self.cv3 = Conv(c4, c4, 3, 1)
#         self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


# class AConv(nn.Module):
#     """AConv."""

#     def __init__(self, c1, c2):
#         """Initializes AConv module with convolution layers."""
#         super().__init__()
#         self.cv1 = Conv(c1, c2, 3, 2, 1)

#     def forward(self, x):
#         """Forward pass through AConv layer."""
#         x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         return self.cv1(x)


# class ADown(nn.Module):
#     """ADown."""

#     def __init__(self, c1, c2):
#         """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
#         super().__init__()
#         self.c = c2 // 2
#         self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
#         self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

#     def forward(self, x):
#         """Forward pass through ADown layer."""
#         x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
#         x1, x2 = x.chunk(2, 1)
#         x1 = self.cv1(x1)
#         x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
#         x2 = self.cv2(x2)
#         return torch.cat((x1, x2), 1)


# class SPPELAN(nn.Module):
#     """SPP-ELAN."""

#     def __init__(self, c1, c2, c3, k=5):
#         """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
#         super().__init__()
#         self.c = c3
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
#         self.cv5 = Conv(4 * c3, c2, 1, 1)

#     def forward(self, x):
#         """Forward pass through SPPELAN layer."""
#         y = [self.cv1(x)]
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
#         return self.cv5(torch.cat(y, 1))


# class CBLinear(nn.Module):
#     """CBLinear."""

#     def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
#         """Initializes the CBLinear module, passing inputs unchanged."""
#         super().__init__()
#         self.c2s = c2s
#         self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

#     def forward(self, x):
#         """Forward pass through CBLinear layer."""
#         return self.conv(x).split(self.c2s, dim=1)


# class CBFuse(nn.Module):
#     """CBFuse."""

#     def __init__(self, idx):
#         """Initializes CBFuse module with layer index for selective feature fusion."""
#         super().__init__()
#         self.idx = idx

#     def forward(self, xs):
#         """Forward pass through CBFuse layer."""
#         target_size = xs[-1].shape[2:]
#         res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
#         return torch.sum(torch.stack(res + xs[-1:]), dim=0)


# class C3f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = [self.cv2(x), self.cv1(x)]
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv3(torch.cat(y, 1))


# class C3k2(C2f):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
#         """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(
#             C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
#         )


# class C3k(C3):
#     """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
#         """Initializes the C3k module with specified channels, number of layers, and configurations."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# class RepVGGDW(torch.nn.Module):
#     """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

#     def __init__(self, ed) -> None:
#         """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
#         super().__init__()
#         self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
#         self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
#         self.dim = ed
#         self.act = nn.SiLU()

#     def forward(self, x):
#         """
#         Performs a forward pass of the RepVGGDW block.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             (torch.Tensor): Output tensor after applying the depth wise separable convolution.
#         """
#         return self.act(self.conv(x) + self.conv1(x))

#     def forward_fuse(self, x):
#         """
#         Performs a forward pass of the RepVGGDW block without fusing the convolutions.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             (torch.Tensor): Output tensor after applying the depth wise separable convolution.
#         """
#         return self.act(self.conv(x))

#     @torch.no_grad()
#     def fuse(self):
#         """
#         Fuses the convolutional layers in the RepVGGDW block.

#         This method fuses the convolutional layers and updates the weights and biases accordingly.
#         """
#         conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
#         conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

#         conv_w = conv.weight
#         conv_b = conv.bias
#         conv1_w = conv1.weight
#         conv1_b = conv1.bias

#         conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

#         final_conv_w = conv_w + conv1_w
#         final_conv_b = conv_b + conv1_b

#         conv.weight.data.copy_(final_conv_w)
#         conv.bias.data.copy_(final_conv_b)

#         self.conv = conv
#         del self.conv1


# class CIB(nn.Module):
#     """
#     Conditional Identity Block (CIB) module.

#     Args:
#         c1 (int): Number of input channels.
#         c2 (int): Number of output channels.
#         shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
#         e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
#         lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
#     """

#     def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
#         """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = nn.Sequential(
#             Conv(c1, c1, 3, g=c1),
#             Conv(c1, 2 * c_, 1),
#             RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
#             Conv(2 * c_, c2, 1),
#             Conv(c2, c2, 3, g=c2),
#         )

#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         """
#         Forward pass of the CIB module.

#         Args:
#             x (torch.Tensor): Input tensor.

#         Returns:
#             (torch.Tensor): Output tensor.
#         """
#         return x + self.cv1(x) if self.add else self.cv1(x)


# class C2fCIB(C2f):
#     """
#     C2fCIB class represents a convolutional block with C2f and CIB modules.

#     Args:
#         c1 (int): Number of input channels.
#         c2 (int): Number of output channels.
#         n (int, optional): Number of CIB modules to stack. Defaults to 1.
#         shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
#         lk (bool, optional): Whether to use local key connection. Defaults to False.
#         g (int, optional): Number of groups for grouped convolution. Defaults to 1.
#         e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
#     """

#     def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
#         """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


# class Attention(nn.Module):
#     """
#     Attention module that performs self-attention on the input tensor.

#     Args:
#         dim (int): The input tensor dimension.
#         num_heads (int): The number of attention heads.
#         attn_ratio (float): The ratio of the attention key dimension to the head dimension.

#     Attributes:
#         num_heads (int): The number of attention heads.
#         head_dim (int): The dimension of each attention head.
#         key_dim (int): The dimension of the attention key.
#         scale (float): The scaling factor for the attention scores.
#         qkv (Conv): Convolutional layer for computing the query, key, and value.
#         proj (Conv): Convolutional layer for projecting the attended values.
#         pe (Conv): Convolutional layer for positional encoding.
#     """

#     def __init__(self, dim, num_heads=8, attn_ratio=0.5):
#         """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.key_dim = int(self.head_dim * attn_ratio)
#         self.scale = self.key_dim**-0.5
#         nh_kd = self.key_dim * num_heads
#         h = dim + nh_kd * 2
#         self.qkv = Conv(dim, h, 1, act=False)
#         self.proj = Conv(dim, dim, 1, act=False)
#         self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

#     def forward(self, x):
#         """
#         Forward pass of the Attention module.

#         Args:
#             x (torch.Tensor): The input tensor.

#         Returns:
#             (torch.Tensor): The output tensor after self-attention.
#         """
#         B, C, H, W = x.shape
#         N = H * W
#         qkv = self.qkv(x)
#         q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
#             [self.key_dim, self.key_dim, self.head_dim], dim=2
#         )

#         attn = (q.transpose(-2, -1) @ k) * self.scale
#         attn = attn.softmax(dim=-1)
#         x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
#         x = self.proj(x)
#         return x


# class PSABlock(nn.Module):
#     """
#     PSABlock class implementing a Position-Sensitive Attention block for neural networks.

#     This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
#     with optional shortcut connections.

#     Attributes:
#         attn (Attention): Multi-head attention module.
#         ffn (nn.Sequential): Feed-forward neural network module.
#         add (bool): Flag indicating whether to add shortcut connections.

#     Methods:
#         forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

#     Examples:
#         Create a PSABlock and perform a forward pass
#         >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
#         >>> input_tensor = torch.randn(1, 128, 32, 32)
#         >>> output_tensor = psablock(input_tensor)
#     """

#     def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
#         """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
#         super().__init__()

#         self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
#         self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
#         self.add = shortcut

#     def forward(self, x):
#         """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
#         x = x + self.attn(x) if self.add else self.attn(x)
#         x = x + self.ffn(x) if self.add else self.ffn(x)
#         return x


# class PSA(nn.Module):
#     """
#     PSA class for implementing Position-Sensitive Attention in neural networks.

#     This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
#     input tensors, enhancing feature extraction and processing capabilities.

#     Attributes:
#         c (int): Number of hidden channels after applying the initial convolution.
#         cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
#         cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
#         attn (Attention): Attention module for position-sensitive attention.
#         ffn (nn.Sequential): Feed-forward network for further processing.

#     Methods:
#         forward: Applies position-sensitive attention and feed-forward network to the input tensor.

#     Examples:
#         Create a PSA module and apply it to an input tensor
#         >>> psa = PSA(c1=128, c2=128, e=0.5)
#         >>> input_tensor = torch.randn(1, 128, 64, 64)
#         >>> output_tensor = psa.forward(input_tensor)
#     """

#     def __init__(self, c1, c2, e=0.5):
#         """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
#         super().__init__()
#         assert c1 == c2
#         self.c = int(c1 * e)
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv(2 * self.c, c1, 1)

#         self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
#         self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

#     def forward(self, x):
#         """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
#         a, b = self.cv1(x).split((self.c, self.c), dim=1)
#         b = b + self.attn(b)
#         b = b + self.ffn(b)
#         return self.cv2(torch.cat((a, b), 1))


# class C2PSA(nn.Module):
#     """
#     C2PSA module with attention mechanism for enhanced feature extraction and processing.

#     This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
#     capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

#     Attributes:
#         c (int): Number of hidden channels.
#         cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
#         cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
#         m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

#     Methods:
#         forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

#     Notes:
#         This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

#     Examples:
#         >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
#         >>> input_tensor = torch.randn(1, 256, 64, 64)
#         >>> output_tensor = c2psa(input_tensor)
#     """

#     def __init__(self, c1, c2, n=1, e=0.5):
#         """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
#         super().__init__()
#         assert c1 == c2
#         self.c = int(c1 * e)
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv(2 * self.c, c1, 1)

#         self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

#     def forward(self, x):
#         """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
#         a, b = self.cv1(x).split((self.c, self.c), dim=1)
#         b = self.m(b)
#         return self.cv2(torch.cat((a, b), 1))


# class C2fPSA(C2f):
#     """
#     C2fPSA module with enhanced feature extraction using PSA blocks.

#     This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

#     Attributes:
#         c (int): Number of hidden channels.
#         cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
#         cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
#         m (nn.ModuleList): List of PSA blocks for feature extraction.

#     Methods:
#         forward: Performs a forward pass through the C2fPSA module.
#         forward_split: Performs a forward pass using split() instead of chunk().

#     Examples:
#         >>> import torch
#         >>> from ultralytics.models.common import C2fPSA
#         >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
#         >>> x = torch.randn(1, 64, 128, 128)
#         >>> output = model(x)
#         >>> print(output.shape)
#     """

#     def __init__(self, c1, c2, n=1, e=0.5):
#         """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
#         assert c1 == c2
#         super().__init__(c1, c2, n=n, e=e)
#         self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


# class SCDown(nn.Module):
#     """
#     SCDown module for downsampling with separable convolutions.

#     This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
#     efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

#     Attributes:
#         cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
#         cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

#     Methods:
#         forward: Applies the SCDown module to the input tensor.

#     Examples:
#         >>> import torch
#         >>> from ultralytics import SCDown
#         >>> model = SCDown(c1=64, c2=128, k=3, s=2)
#         >>> x = torch.randn(1, 64, 128, 128)
#         >>> y = model(x)
#         >>> print(y.shape)
#         torch.Size([1, 128, 64, 64])
#     """

#     def __init__(self, c1, c2, k, s):
#         """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
#         super().__init__()
#         self.cv1 = Conv(c1, c2, 1, 1)
#         self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

#     def forward(self, x):
#         """Applies convolution and downsampling to the input tensor in the SCDown module."""
#         return self.cv2(self.cv1(x))

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)
    
    
class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Chuncat(nn.Module):
    def __init__(self, dimension=1):
        super(Chuncat, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1 = []
        x2 = []
        for xi in x:
            xi1, xi2 = xi.chunk(2, self.d)
            x1.append(xi1)
            x2.append(xi2)
        return torch.cat(x1+x2, self.d)


class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        return x[0]+x[1]


class Foldcut(nn.Module):
    def __init__(self, dimension=0):
        super(Foldcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1, x2 = x.chunk(2, self.d)
        return x1+x2


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation
    #d=1 is new added in YOLOv8
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        #self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

#new added in YOLOv8: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py
class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse

class LightConv(nn.Module):
    """Light convolution with args(ch_in, ch_out, kernel).
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))

class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))

#https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/transformer.py
class MSDeformAttn(nn.Module):
    """
    Original Multi-Scale Deformable Attention Module.
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {num_points}.')
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)

class DeformableTransformerDecoderLayer(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        # self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1),
                             attn_mask=attn_mask)[0].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # cross attention
        tgt = self.cross_attn(self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes,
                              padding_mask)
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # ffn
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
            self,
            embed,  # decoder embeddings
            refer_bbox,  # anchor
            feats,  # image features
            shapes,  # feature shapes
            bbox_head,
            score_head,
            pos_mlp,
            attn_mask=None,
            padding_mask=None):
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)
    
####end of new added in YOLOv8

class RobustConv(nn.Module):
    # Robust convolution (use high kernel size 7-11 for: downsampling and other layers). Train for 300 - 450 epochs.
    def __init__(self, c1, c2, k=7, s=1, p=None, g=1, act=True, layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv, self).__init__()
        self.conv_dw = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv1x1 = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv1x1(self.conv_dw(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1)) 
        return x


class RobustConv2(nn.Module):
    # Robust convolution 2 (use [32, 5, 2] or [32, 7, 4] or [32, 11, 8] for one of the paths in CSP).
    def __init__(self, c1, c2, k=7, s=4, p=None, g=1, act=True, layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv2, self).__init__()
        self.conv_strided = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv_deconv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=s, stride=s, 
                                              padding=0, bias=True, dilation=1, groups=1
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = self.conv_deconv(self.conv_strided(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1)) 
        return x
    

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class Stem(nn.Module):
    # Stem
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Stem, self).__init__()
        c_ = int(c2/2)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 2)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 2)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv4(torch.cat((self.cv3(self.cv2(x)), self.pool(x)), dim=1))


class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2//2, 3, k)
        self.cv3 = Conv(c1, c2//2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP, https://arxiv.org/abs/1406.4729.
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
    

# class Bottleneck(nn.Module):
#     # Darknet bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super(Bottleneck, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_, c2, 3, 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

#updated Bottleneck in yolov8, kernel is added
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Res(nn.Module):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Res, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1, g=g)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class ResX(Res):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels


class Ghost(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(Ghost, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

##### end of basic #####


##### cspnet #####

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class GhostSPPCSPC(SPPCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 1)
        self.cv4 = GhostConv(c_, c_, 1, 1)
        self.cv5 = GhostConv(4 * c_, c_, 1, 1)
        self.cv6 = GhostConv(c_, c_, 3, 1)
        self.cv7 = GhostConv(2 * c_, c2, 1, 1)


class GhostStem(Stem):
    # Stem
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, p, g, act)
        c_ = int(c2/2)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 3, 2)
        self.cv2 = GhostConv(c_, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 2)
        self.cv4 = GhostConv(2 * c_, c2, 1, 1)
        

class BottleneckCSPA(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPB(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class ResCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResXCSPA(ResCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPB(ResCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPC(ResCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class GhostCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])

##### end of cspnet #####


##### yolor #####

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
    
##### end of yolor #####


##### repvgg #####
#changed in YOLOv8
# class RepConv(nn.Module):
#     # Represented convolution
#     # https://arxiv.org/abs/2101.03697

#     def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
#         super(RepConv, self).__init__()

#         self.deploy = deploy
#         self.groups = g
#         self.in_channels = c1
#         self.out_channels = c2

#         assert k == 3
#         assert autopad(k, p) == 1

#         padding_11 = autopad(k, p) - k // 2

#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

#         if deploy:
#             self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

#         else:
#             self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

#             self.rbr_dense = nn.Sequential(
#                 nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
#                 nn.BatchNorm2d(num_features=c2),
#             )

#             self.rbr_1x1 = nn.Sequential(
#                 nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
#                 nn.BatchNorm2d(num_features=c2),
#             )

#     def forward(self, inputs):
#         if hasattr(self, "rbr_reparam"):
#             return self.act(self.rbr_reparam(inputs))

#         if self.rbr_identity is None:
#             id_out = 0
#         else:
#             id_out = self.rbr_identity(inputs)

#         return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
#         kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
#         return (
#             kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
#             bias3x3 + bias1x1 + biasid,
#         )

#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

#     def _fuse_bn_tensor(self, branch):
#         if branch is None:
#             return 0, 0
#         if isinstance(branch, nn.Sequential):
#             kernel = branch[0].weight
#             running_mean = branch[1].running_mean
#             running_var = branch[1].running_var
#             gamma = branch[1].weight
#             beta = branch[1].bias
#             eps = branch[1].eps
#         else:
#             assert isinstance(branch, nn.BatchNorm2d)
#             if not hasattr(self, "id_tensor"):
#                 input_dim = self.in_channels // self.groups
#                 kernel_value = np.zeros(
#                     (self.in_channels, input_dim, 3, 3), dtype=np.float32
#                 )
#                 for i in range(self.in_channels):
#                     kernel_value[i, i % input_dim, 1, 1] = 1
#                 self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
#             kernel = self.id_tensor
#             running_mean = branch.running_mean
#             running_var = branch.running_var
#             gamma = branch.weight
#             beta = branch.bias
#             eps = branch.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std

#     def repvgg_convert(self):
#         kernel, bias = self.get_equivalent_kernel_bias()
#         return (
#             kernel.detach().cpu().numpy(),
#             bias.detach().cpu().numpy(),
#         )

#     def fuse_conv_bn(self, conv, bn):

#         std = (bn.running_var + bn.eps).sqrt()
#         bias = bn.bias - bn.running_mean * bn.weight / std

#         t = (bn.weight / std).reshape(-1, 1, 1, 1)
#         weights = conv.weight * t

#         bn = nn.Identity()
#         conv = nn.Conv2d(in_channels = conv.in_channels,
#                               out_channels = conv.out_channels,
#                               kernel_size = conv.kernel_size,
#                               stride=conv.stride,
#                               padding = conv.padding,
#                               dilation = conv.dilation,
#                               groups = conv.groups,
#                               bias = True,
#                               padding_mode = conv.padding_mode)

#         conv.weight = torch.nn.Parameter(weights)
#         conv.bias = torch.nn.Parameter(bias)
#         return conv

#     def fuse_repvgg_block(self):    
#         if self.deploy:
#             return
#         print(f"RepConv.fuse_repvgg_block")
                
#         self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
#         self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
#         rbr_1x1_bias = self.rbr_1x1.bias
#         weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
#         # Fuse self.rbr_identity
#         if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
#             # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
#             identity_conv_1x1 = nn.Conv2d(
#                     in_channels=self.in_channels,
#                     out_channels=self.out_channels,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0,
#                     groups=self.groups, 
#                     bias=False)
#             identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
#             identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
#             # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
#             identity_conv_1x1.weight.data.fill_(0.0)
#             identity_conv_1x1.weight.data.fill_diagonal_(1.0)
#             identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
#             # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

#             identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
#             bias_identity_expanded = identity_conv_1x1.bias
#             weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
#         else:
#             # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
#             bias_identity_expanded = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
#             weight_identity_expanded = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        

#         #print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
#         #print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
#         #print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

#         self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
#         self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
#         self.rbr_reparam = self.rbr_dense
#         self.deploy = True

#         if self.rbr_identity is not None:
#             del self.rbr_identity
#             self.rbr_identity = None

#         if self.rbr_1x1 is not None:
#             del self.rbr_1x1
#             self.rbr_1x1 = None

#         if self.rbr_dense is not None:
#             del self.rbr_dense
#             self.rbr_dense = None

class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status. This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class RepBottleneck(Bottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut=True, g=1, e=0.5)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c2, 3, 1, g=g)


class RepBottleneckCSPA(BottleneckCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPB(BottleneckCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPC(BottleneckCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepRes(Res):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResCSPA(ResCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPB(ResCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPC(ResCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResX(ResX):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResXCSPA(ResXCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPB(ResXCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPC(ResXCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])

##### end of repvgg #####


##### transformer #####

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x

##### end of transformer #####


##### yolov5 #####

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))
        

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
    
    
class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


# class NMS(nn.Module):
#     # Non-Maximum Suppression (NMS) module
#     conf = 0.25  # confidence threshold
#     iou = 0.45  # IoU threshold
#     classes = None  # (optional list) filter by class

#     def __init__(self):
#         super(NMS, self).__init__()

#     def forward(self, x):
#         return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)

# YOOv5 Related, new add
class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        # if USE_MISHACT == True:
        #     self.act = Mish()
        # else:
        #     self.act = nn.LeakyReLU(0.1, inplace=True)
        #self.act = nn.LeakyReLU(0.1, inplace=True)
        self.act = nn.SiLU() #new updated in YOLOv8
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        #self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
        #new version in Yolov8, added kernel
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
        #return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)

class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)
##### end of yolov5 ######

##### yolov8 #####
#https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float) #0-16
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))

#https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/conv.py
class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))

class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))

#ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/transformer.py
class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        # from ...utils.torch_utils import TORCH_1_9
        # if not TORCH_1_9:
        #     raise ModuleNotFoundError(
        #         'TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True).')
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos=None):
        """Add position embeddings if given."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
    
class AIFI(TransformerEncoderLayer):

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]

class MLPBlock(nn.Module):

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
#####end of yolov8 ####

##### orepa #####

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std
    
    
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                             stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                                      stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv    

class OREPA_3x3_RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False):
        super(OREPA_3x3_RepConv, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.branch_counter = 0

        self.weight_rbr_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_rbr_origin, a=math.sqrt(1.0))
        self.branch_counter += 1


        if groups < out_channels:
            self.weight_rbr_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), 1, 1))
            self.weight_rbr_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_avg_conv, a=1.0)
            nn.init.kaiming_uniform_(self.weight_rbr_pfir_conv, a=1.0)
            self.weight_rbr_avg_conv.data
            self.weight_rbr_pfir_conv.data
            self.register_buffer('weight_rbr_avg_avg', torch.ones(kernel_size, kernel_size).mul(1.0/kernel_size/kernel_size))
            self.branch_counter += 1

        else:
            raise NotImplementedError
        self.branch_counter += 1

        if internal_channels_1x1_3x3 is None:
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels   # For mobilenet, it is better to have 2X internal channels

        if internal_channels_1x1_3x3 == in_channels:
            self.weight_rbr_1x1_kxk_idconv1 = nn.Parameter(torch.zeros(in_channels, int(in_channels/self.groups), 1, 1))
            id_value = np.zeros((in_channels, int(in_channels/self.groups), 1, 1))
            for i in range(in_channels):
                id_value[i, i % int(in_channels/self.groups), 0, 0] = 1
            id_tensor = torch.from_numpy(id_value).type_as(self.weight_rbr_1x1_kxk_idconv1)
            self.register_buffer('id_tensor', id_tensor)

        else:
            self.weight_rbr_1x1_kxk_conv1 = nn.Parameter(torch.Tensor(internal_channels_1x1_3x3, int(in_channels/self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv1, a=math.sqrt(1.0))
        self.weight_rbr_1x1_kxk_conv2 = nn.Parameter(torch.Tensor(out_channels, int(internal_channels_1x1_3x3/self.groups), kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv2, a=math.sqrt(1.0))
        self.branch_counter += 1

        expand_ratio = 8
        self.weight_rbr_gconv_dw = nn.Parameter(torch.Tensor(in_channels*expand_ratio, 1, kernel_size, kernel_size))
        self.weight_rbr_gconv_pw = nn.Parameter(torch.Tensor(out_channels, in_channels*expand_ratio, 1, 1))
        nn.init.kaiming_uniform_(self.weight_rbr_gconv_dw, a=math.sqrt(1.0))
        nn.init.kaiming_uniform_(self.weight_rbr_gconv_pw, a=math.sqrt(1.0))
        self.branch_counter += 1

        if out_channels == in_channels and stride == 1:
            self.branch_counter += 1

        self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

        self.fre_init()

        nn.init.constant_(self.vector[0, :], 0.25)    #origin
        nn.init.constant_(self.vector[1, :], 0.25)      #avg
        nn.init.constant_(self.vector[2, :], 0.0)      #prior
        nn.init.constant_(self.vector[3, :], 0.5)    #1x1_kxk
        nn.init.constant_(self.vector[4, :], 0.5)     #dws_conv


    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels/2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi*(h+0.5)*(i+1)/3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi*(w+0.5)*(i+1-half_fg)/3)

        self.register_buffer('weight_rbr_prior', prior_tensor)

    def weight_gen(self):

        weight_rbr_origin = torch.einsum('oihw,o->oihw', self.weight_rbr_origin, self.vector[0, :])

        weight_rbr_avg = torch.einsum('oihw,o->oihw', torch.einsum('oihw,hw->oihw', self.weight_rbr_avg_conv, self.weight_rbr_avg_avg), self.vector[1, :])
        
        weight_rbr_pfir = torch.einsum('oihw,o->oihw', torch.einsum('oihw,ohw->oihw', self.weight_rbr_pfir_conv, self.weight_rbr_prior), self.vector[2, :])

        weight_rbr_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_rbr_1x1_kxk_idconv1'):
            weight_rbr_1x1_kxk_conv1 = (self.weight_rbr_1x1_kxk_idconv1 + self.id_tensor).squeeze()
        elif hasattr(self, 'weight_rbr_1x1_kxk_conv1'):
            weight_rbr_1x1_kxk_conv1 = self.weight_rbr_1x1_kxk_conv1.squeeze()
        else:
            raise NotImplementedError
        weight_rbr_1x1_kxk_conv2 = self.weight_rbr_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_rbr_1x1_kxk_conv1.size()
            o, tg, h, w = weight_rbr_1x1_kxk_conv2.size()
            weight_rbr_1x1_kxk_conv1 = weight_rbr_1x1_kxk_conv1.view(g, int(t/g), ig)
            weight_rbr_1x1_kxk_conv2 = weight_rbr_1x1_kxk_conv2.view(g, int(o/g), tg, h, w)
            weight_rbr_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2).view(o, ig, h, w)
        else:
            weight_rbr_1x1_kxk = torch.einsum('ti,othw->oihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2)

        weight_rbr_1x1_kxk = torch.einsum('oihw,o->oihw', weight_rbr_1x1_kxk, self.vector[3, :])

        weight_rbr_gconv = self.dwsc2full(self.weight_rbr_gconv_dw, self.weight_rbr_gconv_pw, self.in_channels)
        weight_rbr_gconv = torch.einsum('oihw,o->oihw', weight_rbr_gconv, self.vector[4, :])    

        weight = weight_rbr_origin + weight_rbr_avg + weight_rbr_1x1_kxk + weight_rbr_pfir + weight_rbr_gconv

        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups):
        
        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t/groups)
        i = int(ig*groups)
        weight_dw = weight_dw.view(groups, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(o, groups, tg)
        
        weight_dsc = torch.einsum('gtihw,ogt->ogihw', weight_dw, weight_pw)
        return weight_dsc.view(o, i, h, w)

    def forward(self, inputs):
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return self.nonlinear(self.bn(out))

class RepConv_OREPA(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, nonlinear=nn.SiLU()):
        super(RepConv_OREPA, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = c1
        self.out_channels = c2

        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert k == 3
        assert padding == 1

        padding_11 = padding - k // 2

        if nonlinear is None:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nonlinear

        # if use_se:
        #     self.se = SEBlock(self.out_channels, internal_neurons=self.out_channels // 16)
        # else:
        #     self.se = nn.Identity()
        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k, stride=s,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=self.in_channels) if self.out_channels == self.in_channels and s == 1 else None
            self.rbr_dense = OREPA_3x3_RepConv(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k, stride=s, padding=padding, groups=groups, dilation=1)
            self.rbr_1x1 = ConvBN(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=s, padding=padding_11, groups=groups, dilation=1)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        out1 = self.rbr_dense(inputs)
        out2 = self.rbr_1x1(inputs)
        out3 = id_out
        out = out1 + out2 + out3

        return self.nonlinearity(self.se(out))


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()

    # Not used for OREPA
    def get_custom_L2(self):
        K3 = self.rbr_dense.weight_gen()
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if not isinstance(branch, nn.BatchNorm2d):
            if isinstance(branch, OREPA_3x3_RepConv):
                kernel = branch.weight_gen()
            elif isinstance(branch, ConvBN):
                kernel = branch.conv.weight
            else:
                raise NotImplementedError
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        print(f"RepConv_OREPA.switch_to_deploy")
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity') 

##### end of orepa #####


##### swin transformer #####    
    
class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # print(attn.dtype, v.dtype)
        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            #print(attn.dtype, v.dtype)
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):

    B, H, W, C = x.shape
    assert H % window_size == 0, 'feature map h and w can not divide by window size'
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer(nn.Module):

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # reshape x[b c h w] to x[b l c]
        _, _, H_, W_ = x.shape

        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size!=0 or W_ % self.window_size!=0:
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) window_size {self.window_size}, Padding.')
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))

        # print('2', x.shape)
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # b, L, c

        # create mask from init to forward
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            attn_mask = None

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)  # b c h w

        if Padding:
            x = x[:, :, :H_, :W_]  # reverse padding

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class STCSPA(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPB(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPC(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

##### end of swin transformer #####   


##### swin transformer v2 ##### 
  
class WindowAttention_v2(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
            
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    
class Mlp_v2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_v2(x, window_size):
    
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse_v2(windows, window_size, H, W):
    
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer_v2(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.SiLU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        #self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        #if min(self.input_resolution) <= self.window_size:
        #    # if window size is larger than input resolution, we don't partition windows
        #    self.shift_size = 0
        #    self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_v2(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=(pretrained_window_size, pretrained_window_size))

        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_v2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # reshape x[b c h w] to x[b l c]
        _, _, H_, W_ = x.shape

        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size!=0 or W_ % self.window_size!=0:
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) window_size {self.window_size}, Padding.')
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))

        # print('2', x.shape)
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # b, L, c

        # create mask from init to forward
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            attn_mask = None

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition_v2(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse_v2(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)  # b c h w
        
        if Padding:
            x = x[:, :, :H_, :W_]  # reverse padding

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class SwinTransformer2Block(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=7):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer_v2(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class ST2CSPA(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ST2CSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class ST2CSPB(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ST2CSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class ST2CSPC(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ST2CSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

##### end of swin transformer v2 #####   

##new add for ScaledYOLOv4
#from mish_cuda import MishCuda as Mish

class BottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_) 
        # if USE_MISHACT==True:
        #     self.act = Mish()
        # else:
        #     self.act = nn.LeakyReLU(0.1, inplace=True) #Mish()
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPPCSP(nn.Module):
    # CSP SPP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSP, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.bn = nn.BatchNorm2d(2 * c_) 
        # if USE_MISHACT==True:
        #     self.act = Mish()
        # else:
        #     self.act = nn.LeakyReLU(0.1, inplace=True) #Mish()
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class ViTEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (batch_size, num_patches, embed_dim)
        x_norm1 = self.norm1(x)
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = x + attn_output  # Residual connection

        x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)
        x = x + mlp_output  # Residual connection

        return x

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, n_encoder=6, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_encoder=n_encoder

        self.encoder_layers = nn.ModuleList([
            ViTEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout=dropout)
            for _ in range(n_encoder)  # Example: 6 layers
        ])

        # Learnable linear projection for Positional Embedding
        self.pos_proj = nn.Linear(2, embed_dim)
        nn.init.trunc_normal_(self.pos_proj.weight, std=0.02)
        nn.init.zeros_(self.pos_proj.bias)

    def forward(self, x):
        # x is (batch_size, c, h, w)
        B, C, H, W = x.shape
        assert C == self.embed_dim, f"Input channels ({C}) must match embed_dim ({self.embed_dim})"

        # Flatten spatial dimensions
        x = x.view(B, C, H * W).permute(0, 2, 1)  # Shape: (B, N, C), where N = H * W

        # Generate positional embeddings if necessary
        pos_embed = self.create_2d_positional_encoding(B, H, W, device=x.device)

        x = x + pos_embed  # Add positional embeddings

        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)  # Shape remains (B, N, C)

        # Reshape back to (B, C, H, W)
        x = x.permute(0, 2, 1).view(B, C, H, W).contiguous()

        return x


        # # Check if input dimensions exceed maximums
        # if h > self.max_h or w > self.max_w:
        #     raise ValueError(
        #         f"Input height ({h}) or width ({w}) exceeds maximum allowed "
        #         f"({self.max_h}, {self.max_w}). Please adjust the input size or the maximum embeddings."
        #     )
        
        # x = x.permute(0, 2, 3, 1).reshape(batch_size, h * w, c)  # (batch_size, num_patches, embed_dim)

        # # Add positional encoding
        # row_indices = torch.arange(h, device=x.device).unsqueeze(1).repeat(1, w)  # Shape: (h, w)
        # col_indices = torch.arange(w, device=x.device).unsqueeze(0).repeat(h, 1)  # Shape: (h, w)

        # # Retrieve row and column embeddings
        # row_embeddings = self.row_embed(row_indices)  # Shape: (h, w, embed_dim)
        # col_embeddings = self.col_embed(col_indices)  # Shape: (h, w, embed_dim)

        # # Combine row and column embeddings to form 2D positional embeddings
        # pos_embed = row_embeddings + col_embeddings  # Shape: (h, w, embed_dim)

        # # Flatten positional embeddings and add to input
        # pos_embed = pos_embed.reshape(1, h * w, c)  # Shape: (1, num_patches, embed_dim)
        # x = x + pos_embed  # Broadcasting over batch dimension

        # for layer in self.encoder_layers:
        #     x = layer(x)  # (batch_size, num_patches, embed_dim)

        # x = x.reshape(batch_size, h, w, c).permute(0, 3, 1, 2).contiguous()  # (batch_size, c, h, w)
        # return x

    def create_2d_positional_encoding(self, B, H, W, device):
        # Create 2D positional embeddings dynamically based on H and W
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, steps=H, device=device),
            torch.linspace(0, 1, steps=W, device=device),
            indexing='ij'
        )
        grid_y = grid_y.reshape(-1)  # Shape: (N,)
        grid_x = grid_x.reshape(-1)  # Shape: (N,)
        # grid_y, grid_x = torch.meshgrid(
        #     torch.arange(H, dtype=torch.float32, device=device),
        #     torch.arange(W, dtype=torch.float32, device=device),
        #     indexing='ij'
        # )
        # grid_y = grid_y.reshape(-1) / H  # Normalize to [0, 1]
        # grid_x = grid_x.reshape(-1) / W  # Normalize to [0, 1]

        pos_embed = torch.stack((grid_x, grid_y), dim=1)  # Shape: (N, 2)

        # Project to embedding dimension
        pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1)  # (B, N, 2)
        pos_embed = self.pos_proj(pos_embed)  # (B, N, C)

        return pos_embed