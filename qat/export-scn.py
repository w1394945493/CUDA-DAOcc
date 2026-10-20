# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from torchpack.utils.config import configs
from mmcv.cnn import fuse_conv_bn
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval
from mmcv.runner import wrap_fp16_model
import torch
import argparse
import os

# 项目自定义导出工具；其中 exptool 负责把稀疏卷积转换为自定义 ONNX 节点。
import lean.funcs as funcs
import lean.exptool as exptool
import lean.quantize as quantize

if __name__ == "__main__":

    #* 解析 LiDAR 稀疏卷积网络（SCN）的导出参数。
    parser = argparse.ArgumentParser(description="Export scn to onnx file")
    #' 每个体素输入的特征维数；DAOcc 默认使用 5 维体素特征。
    parser.add_argument("--in-channel", type=int, default=5, help="SCN num of input channels")
    #' 这里应传入 ptq.py 校准并保存的完整 PTQ 模型。
    parser.add_argument("--ckpt", type=str, default="model/daocc_ptq.pth", help="SCN Checkpoint (scn backbone checkpoint)")
    #' 输出文件的基础路径；脚本稍后会自动补充 .xyz.onnx 或 .zyx.onnx。
    parser.add_argument("--save", type=str, default="model/lidar.backbone.onnx", help="output onnx")
    #' 开启后把稀疏体素坐标顺序由 xyz 调整为 zyx，以适配需要 zyx 坐标的部署后端。
    parser.add_argument("--inverse", action="store_true", help="Transfer the coordinate order of the index from xyz to zyx")

    #' 读取命令行参数，并将 checkpoint 路径展开为绝对路径。
    args = parser.parse_args()
    checkpoint_path = os.path.abspath(os.path.expanduser(args.ckpt))
    #' 提前检查模型文件，避免到 torch.load 时才出现不直观的报错。
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    #' 展开用户目录并转换为绝对路径；允许只提供当前目录下的文件名。
    args.save = os.path.abspath(os.path.expanduser(args.save))
    inverse_indices = args.inverse
    #* 根据体素坐标顺序生成最终文件名，文件名会明确标记 xyz 或 zyx。
    if inverse_indices:
        args.save = os.path.splitext(args.save)[0] + ".zyx.onnx"
    else:
        args.save = os.path.splitext(args.save)[0] + ".xyz.onnx"

    #' 创建 ONNX 输出目录。
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    # ptq.py 保存的是 MMDataParallel 包装后的完整模型，因此先通过 .module 取得实际 DAOcc 模型。
    model = torch.load(checkpoint_path).module
    # 切换到推理模式、移动到 GPU，并将普通浮点运算转换为 FP16。
    model.eval().cuda().half()
    # 只保留 LiDAR 编码器的稀疏卷积 backbone；相机、融合器、解码器和预测头均不在本 ONNX 中。
    model = model.encoders.lidar.backbone

    # 关闭 PyTorch 前向中的伪量化运算，避免导出普通 Q/DQ 节点。
    # 这不会删除 PTQ 校准得到的 amax；exptool 仍会读取量化器中的 amax，并将其写入
    # SparseConvolution 自定义节点的 input_dynamic_range 和 weight_dynamic_ranges 属性。
    quantize.disable_quantization(model).apply()

    # 为所有稀疏卷积模块设置部署精度属性；exptool 会把这些属性写入自定义 ONNX 节点。
    # precision 表示该模块内部计算精度，output_precision 表示输出张量精度。
    for name, module in model.named_modules():
        module.precision = "int8"
        module.output_precision = "int8"

    # 首层直接接收原始体素特征，指定其计算为 FP16；其后大部分稀疏卷积使用 INT8。
    model.conv_input.precision = "fp16"
    # 最后一层输出需要交给后续融合模块，因此将输出恢复为 FP16。
    model.conv_out.output_precision = "fp16"

    # 创建仅用于追踪网络结构的占位输入；它们不是实际 nuScenes 点云数据。
    # voxels 的形状为 [有效体素数, 特征维数]。导出时先用 1 个体素占位，特征维数由 --in-channel 决定。
    voxels = torch.zeros(1, args.in_channel).cuda().half()
    # coors 的每一行为 [batch_index, x, y, z]；坐标必须使用整型。
    coors = torch.zeros(1, 4).int().cuda()
    # 部署程序按单帧推理，因此导出时 batch_size 固定为 1。
    batch_size = 1

    # 执行自定义 ONNX 导出：追踪稀疏 backbone，把卷积权重、坐标规则、计算精度以及
    # PTQ 校准得到的动态范围写入 lidar.backbone.xyz.onnx（或 zyx.onnx）。
    #' 该文件随后由 CUDA-DAOcc 的 libspconv 稀疏卷积运行时读取，不通过 trtexec 转成 .plan。
    exptool.export_onnx(model, voxels, coors, batch_size, inverse_indices, args.save)
