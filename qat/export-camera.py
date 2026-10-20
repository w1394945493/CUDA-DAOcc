# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Beijing Mechanical Equipment Institute. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys

import warnings

warnings.filterwarnings("ignore")

import argparse
import os

import onnx
import torch
from onnxsim import simplify
from torchpack.utils.config import configs
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval

from torch import nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
import lean.quantize as quantize
import torch.nn.functional as F
import einops


def trilinear_grid_sample(im, grid, align_corners=False):
    """
    Imitate F.grid_sample for 5D inputs (N, C, D, H, W) with 3D grid (N, Dg, Hg, Wg, 3).

    Args:
        im (torch.Tensor): Input feature map, shape (N, C, D, H, W)
        grid (torch.Tensor): Sampling grid, shape (N, Dg, Hg, Wg, 3), values in [-1,1]
        align_corners (bool): coordinate mapping mode

    Returns:
        torch.Tensor: Sampled output, shape (N, C, Dg, Hg, Wg)
    """
    N, C, D, H, W = im.shape
    _, Dg, Hg, Wg, _ = grid.shape
    # extract normalized coords
    x = grid[..., 0]
    y = grid[..., 1]
    z = grid[..., 2]

    if align_corners:
        x = ((x + 1) / 2) * (W - 1)
        y = ((y + 1) / 2) * (H - 1)
        z = ((z + 1) / 2) * (D - 1)
    else:
        x = ((x + 1) * W - 1) / 2
        y = ((y + 1) * H - 1) / 2
        z = ((z + 1) * D - 1) / 2

    # flatten grid
    x = x.view(N, -1)
    y = y.view(N, -1)
    z = z.view(N, -1)

    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    z0 = torch.floor(z).long()
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # distances
    xd = (x - x0.float())
    yd = (y - y0.float())
    zd = (z - z0.float())

    # corner weights
    w000 = ((1 - xd) * (1 - yd) * (1 - zd)).unsqueeze(1)
    w001 = ((1 - xd) * (1 - yd) * zd).unsqueeze(1)
    w010 = ((1 - xd) * yd * (1 - zd)).unsqueeze(1)
    w011 = ((1 - xd) * yd * zd).unsqueeze(1)
    w100 = (xd * (1 - yd) * (1 - zd)).unsqueeze(1)
    w101 = (xd * (1 - yd) * zd).unsqueeze(1)
    w110 = (xd * yd * (1 - zd)).unsqueeze(1)
    w111 = (xd * yd * zd).unsqueeze(1)

    # pad im with zeros
    im_padded = F.pad(im, pad=[1, 1, 1, 1, 1, 1], mode='constant', value=0)
    Dp, Hp, Wp = D + 2, H + 2, W + 2

    # shift indices after padding
    x0p, x1p = x0 + 1, x1 + 1
    y0p, y1p = y0 + 1, y1 + 1
    z0p, z1p = z0 + 1, z1 + 1

    # clip indices
    x0p = x0p.clamp(0, Wp - 1)
    x1p = x1p.clamp(0, Wp - 1)
    y0p = y0p.clamp(0, Hp - 1)
    y1p = y1p.clamp(0, Hp - 1)
    z0p = z0p.clamp(0, Dp - 1)
    z1p = z1p.clamp(0, Dp - 1)

    # flatten im_padded spatial dims
    im_flat = im_padded.view(N, C, -1)

    # compute linear indices for corners
    def lin_idx(xi, yi, zi):
        return (zi * Hp * Wp + yi * Wp + xi).unsqueeze(1).expand(-1, C, -1)

    idx000 = lin_idx(x0p, y0p, z0p)
    idx001 = lin_idx(x0p, y0p, z1p)
    idx010 = lin_idx(x0p, y1p, z0p)
    idx011 = lin_idx(x0p, y1p, z1p)
    idx100 = lin_idx(x1p, y0p, z0p)
    idx101 = lin_idx(x1p, y0p, z1p)
    idx110 = lin_idx(x1p, y1p, z0p)
    idx111 = lin_idx(x1p, y1p, z1p)

    Ia = torch.gather(im_flat, 2, idx000)
    Ib = torch.gather(im_flat, 2, idx001)
    Ic = torch.gather(im_flat, 2, idx010)
    Id = torch.gather(im_flat, 2, idx011)
    Ie = torch.gather(im_flat, 2, idx100)
    If = torch.gather(im_flat, 2, idx101)
    Ig = torch.gather(im_flat, 2, idx110)
    Ih = torch.gather(im_flat, 2, idx111)

    out = (Ia * w000 + Ib * w001 + Ic * w010 + Id * w011 +
           Ie * w100 + If * w101 + Ig * w110 + Ih * w111)

    return out.view(N, C, Dg, Hg, Wg)


def nearest_grid_sample(im, grid, align_corners=False):
    """
    Nearest-neighbor sampling for 5D input (N, C, D, H, W) using 3D normalized grid (N, Dg, Hg, Wg, 3)
    """
    N, C, D, H, W = im.shape
    _, Dg, Hg, Wg, _ = grid.shape
    x = grid[..., 0]
    y = grid[..., 1]
    z = grid[..., 2]

    if align_corners:
        x = ((x + 1) / 2) * (W - 1)
        y = ((y + 1) / 2) * (H - 1)
        z = ((z + 1) / 2) * (D - 1)
    else:
        x = ((x + 1) * W - 1) / 2
        y = ((y + 1) * H - 1) / 2
        z = ((z + 1) * D - 1) / 2

    # Flatten grid
    x = x.view(N, -1)
    y = y.view(N, -1)
    z = z.view(N, -1)

    # Round to nearest neighbor
    x0 = torch.round(x).long()
    y0 = torch.round(y).long()
    z0 = torch.round(z).long()

    # Pad feature map
    im_padded = F.pad(im, pad=[1, 1, 1, 1, 1, 1], mode='constant', value=0)
    Dp, Hp, Wp = D + 2, H + 2, W + 2

    # Shift indices for padding
    x0p = x0 + 1
    y0p = y0 + 1
    z0p = z0 + 1

    # Clip to valid range
    x0p = x0p.clamp(0, Wp - 1)
    y0p = y0p.clamp(0, Hp - 1)
    z0p = z0p.clamp(0, Dp - 1)

    im_flat = im_padded.view(N, C, -1)

    def lin_idx(xi, yi, zi):
        return (zi * Hp * Wp + yi * Wp + xi).unsqueeze(1).expand(-1, C, -1)

    idx = lin_idx(x0p, y0p, z0p)
    out = torch.gather(im_flat, 2, idx)
    return out.view(N, C, Dg, Hg, Wg)


class SubclassCameraModule(nn.Module):
    #* 这是用于 ONNX 导出的相机分支包装器。torch.onnx.export 只追踪本类 forward 实际执行的运算，
    #* 因而即使 self.model 指向完整 DAOcc，未在 forward 中调用的 LiDAR、fuser、decoder 和 heads
    #* 都不会写入 camera.backbone.onnx。
    def __init__(self, model, mode='bilinear'):
        super(SubclassCameraModule, self).__init__()
        # 保存完整模型只是为了访问其中已经加载好权重的相机子模块及 vtransform 配置。
        self.model = model
        self.mode = mode

    def forward(self, x, img_ref_points):
        # 输入 x 为 [batch, 6路相机, RGB, height, width]，先合并 batch 和相机维度。
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        #* ONNX 的神经网络部分从这里开始：只执行相机 ResNet backbone 和 FPN neck。
        x = self.model.encoders.camera.backbone(x)
        x = self.model.encoders.camera.neck(x)

        # FPN 可能返回多层特征列表，本配置只使用第一层特征。
        if not isinstance(x, torch.Tensor):
            x = x[0]

        # 恢复 batch 和六路相机维度，供后续体素采样使用。
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        B = x.shape[0]
        if self.model.encoders.camera.vtransform.transfer_conv is not None:
            # 使用 vtransform 中训练好的 1x1 卷积调整相机特征通道数。
            x = einops.rearrange(x, 'B N C H W -> (B N) C H W')
            x = self.model.encoders.camera.vtransform.transfer_conv(x)
            x = einops.rearrange(x, '(B N) C H W -> B C N H W', B=B)
        else:
            x = einops.rearrange(x, 'B N C H W -> B C N H W')

        #* 根据 img_ref_points 从六路二维图像特征中采样固定三维体素中心对应的特征。
        if self.mode == 'bilinear':
            sampling_feats = trilinear_grid_sample(x, img_ref_points, align_corners=True)
        elif self.mode == 'nearest':
            sampling_feats = nearest_grid_sample(x, img_ref_points, align_corners=True)
        else:
            raise NotImplementedError
        # 将采样结果整理为 [batch, 三维采样点数量, channel]。
        sampling_feats = einops.rearrange(sampling_feats,
                                          'b c num_points 1 1 -> b num_points c',
                                          b=B)

        #* 按 vtransform 的 z/h/w 网格恢复体素结构，并把高度 z 合并进通道，形成二维 BEV 特征。
        feats_volume = einops.rearrange(sampling_feats,
                                        'bs (z h w) c -> bs (z c) w h',
                                        z=self.model.encoders.camera.vtransform.volume_size[2],
                                        h=self.model.encoders.camera.vtransform.volume_size[0],
                                        w=self.model.encoders.camera.vtransform.volume_size[1])
        if self.model.encoders.camera.vtransform.down_sample is not None:
            # 执行训练模型中的下采样层，使输出尺寸和后续 fuser 需要的尺寸一致。
            feats_volume = self.model.encoders.camera.vtransform.down_sample(feats_volume)
        #* 此处 return 就是 ONNX 图的边界；返回 camera_feature 后不会继续执行 fuser 或预测头。
        return feats_volume


def parse_args():
    parser = argparse.ArgumentParser(description="Export DAOcc model")
    parser.add_argument('--ckpt', type=str, default='model/daocc_ptq.pth')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--mode', type=str, default='bilinear')
    # 可指定示例输入文件；不传时保持原来的默认路径。
    parser.add_argument(
        '--example-data', type=str, default='example-data/example-data.pth',
        help='example input .pth used for camera ONNX export')
    # 可指定 ONNX 输出目录；不传时根据精度沿用原目录。
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='output directory (default: model/resnet50[-int8])')
    args = parser.parse_args()
    return args


def main():
    # 读取输入权重、示例数据、精度模式、采样模式和输出目录。
    args = parse_args()

    # 将用户路径展开为绝对路径，并在加载前检查文件。
    checkpoint_path = os.path.abspath(os.path.expanduser(args.ckpt))
    example_data_path = os.path.abspath(os.path.expanduser(args.example_data))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not os.path.isfile(example_data_path):
        raise FileNotFoundError(f"Example data does not exist: {example_data_path}")

    # PTQ 文件保存的是 MMDataParallel 包装后的模型，.module 取出实际 DAOcc 网络。
    model = torch.load(checkpoint_path).module
    #* 默认保留 PTQ 量化节点，供 TensorRT 构建 INT8+FP16 混合精度引擎。
    default_output_dir = "model/resnet50-int8"
    if args.fp16:
        #* 指定 --fp16 时关闭全部伪量化器，导出不含 INT8 Q/DQ 约束的版本。
        default_output_dir = "model/resnet50"
        quantize.disable_quantization(model).apply()
    # 创建最终 ONNX 输出目录；显式参数优先于默认目录。
    save_root = os.path.abspath(os.path.expanduser(
        args.output_dir if args.output_dir is not None else default_output_dir))
    os.makedirs(save_root, exist_ok=True)

    # 示例文件提供真实六路相机张量，用于确定 ONNX 输入形状和前向路径。
    data = torch.load(example_data_path)
    img = data["img"].data[0].cuda()
    #* 该输入表示投影到图像特征上的采样坐标；此处零值只用于追踪固定形状，
    #* 实际 C++ 推理会传入当前样例的 img_ref_points.bin。
    img_ref_points = torch.zeros([1, 324000, 1, 1, 3]).cuda()

    #* model 虽是完整 DAOcc，但实际传给 ONNX 导出器的是包装器 camera_model；
    #* 导出边界由 SubclassCameraModule.forward 的调用链和 return 位置决定。
    camera_model = SubclassCameraModule(model, args.mode)
    # 放到 GPU 并进入推理模式，关闭 BatchNorm/Dropout 的训练行为。
    camera_model.cuda().eval()

    # 导出只执行前向传播，不计算梯度，也不生成反向传播计算图。
    with torch.no_grad():
        # 拼出相机 ONNX 的最终保存路径，例如 <output-dir>/camera.backbone.onnx。
        camera_backbone_onnx = f"{save_root}/camera.backbone.onnx"

        # 让 pytorch-quantization 使用 ONNX 可识别的 fake-quant 表达方式。
        # 启用后，校准得到的 scale 会导出为onnx中的 QuantizeLinear/DequantizeLinear（Q/DQ）节点，
        # 后续 TensorRT 根据这些 Q/DQ 节点决定可使用 INT8 的区段及其量化范围。
        # TensorRT构建.plan 时读取这些 scale，决定网络中哪些张量使用 INT8，以及如何在 INT8 和 FP16 之间转换。
        TensorQuantizer.use_fb_fake_quant = True

        #* 调用 PyTorch ONNX 导出器：它会用下面的示例输入执行一次 camera_model.forward，
        #* 并把这次前向实际经过的算子、连接关系和所用权重记录成 ONNX 计算图。
        torch.onnx.export(
            # 被追踪的 nn.Module；导出范围由 SubclassCameraModule.forward 决定。
            camera_model,
            # 按 forward(x, img_ref_points) 的参数顺序提供两份示例输入。
            (img, img_ref_points),
            # 指定生成的 ONNX 文件路径。
            camera_backbone_onnx,
            # 给两个输入张量命名，C++/TensorRT 会按这些名字绑定输入。
            input_names=["img", "img_ref_points"],
            # 给 forward 返回的相机 BEV 特征命名，作为引擎输出绑定名。
            output_names=["camera_feature"],
            # 使用 ONNX opset 13 表达算子；该版本与本项目导出逻辑和 TensorRT 8.5兼容。
            opset_version=13,
            # 导出时提前计算纯常量子图，例如固定 shape 计算，以减少运行时节点。
            do_constant_folding=True,
            # 如需排查导出图，可取消下一行注释打印完整 ONNX 节点信息；正常运行无需开启。
            # verbose=True,
        )

        # 重新从磁盘读取刚导出的原始 ONNX，得到可供 onnxsim 处理的 ModelProto。
        onnx_orig = onnx.load(camera_backbone_onnx)
        # simplify 返回简化后的模型和校验结果；它会消除部分冗余 reshape、常量和恒等节点。
        onnx_simp, check = simplify(onnx_orig)
        # 校验不通过就立即停止，避免用可能不等价的简化图继续构建 TensorRT 引擎。
        assert check, "Simplified ONNX model could not be validated"
        #* 用简化后的模型覆盖原始 ONNX；build_trt_engine.sh 随后将它转成 camera.backbone.plan。
        onnx.save(onnx_simp, camera_backbone_onnx)
        # 在终端报告导出完成及实际保存位置。
        print(f"🚀 The export is completed. ONNX save as {camera_backbone_onnx} 🤗, Have a nice day~")


# 只有直接执行本文件时才调用 main；作为模块导入时不会自动开始导出。
if __name__ == "__main__":
    main()
