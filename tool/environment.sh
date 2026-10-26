#!/bin/bash
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

# export CUDA_VISIBLE_DEVICES=2

#* TensorRT 安装根目录，需要改成你实际解压或安装的位置。
#* TensorRT 8.5.3 系统安装路径。
# export TensorRT_Lib=/usr/lib/x86_64-linux-gnu
# export TensorRT_Inc=/usr/include/x86_64-linux-gnu
# export TensorRT_Bin=/usr/src/tensorrt/bin
#* CUDA 12.x 路线：使用仓库 local 下独立解压的 SDK，避免混入系统 8.5.3。
#* 默认路径相对于本脚本定位；也可在 source 前设置 TRT_ROOT 指向其他安装目录。
export TRT_ROOT="${TRT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/local/TensorRT-8.6.1.6}"
export TensorRT_Lib="$TRT_ROOT/lib"
export TensorRT_Inc="$TRT_ROOT/include"
export TensorRT_Bin="$TRT_ROOT/bin"

#* CUDA 11.8 安装路径。
# export CUDA_HOME=/usr/local/cuda
# export CUDA_Lib=$CUDA_HOME/lib64
# export CUDA_Inc=$CUDA_HOME/include
# export CUDA_Bin=$CUDA_HOME/bin
#* cuda 12.1 安装路径
export CUDA_HOME=/usr/local/cuda-12.1
export CUDA_Lib="$CUDA_HOME/lib64"
export CUDA_Inc="$CUDA_HOME/include"
export CUDA_Bin="$CUDA_HOME/bin"

#* 旧系统默认是 CUDA 11.8 的 cuDNN 8.6；此路径保留为回退，不代表已验证兼容。
#* 配好 CUDA 12.x 对应 cuDNN 后，可在 source 前 export CUDNN_Lib=/实际路径/lib。
export CUDNN_Lib=${CUDNN_Lib:-/usr/lib/x86_64-linux-gnu}

#* WSL2 由 Windows 显卡驱动提供 libcuda.so，必须加入动态库搜索路径。
export CUDA_Driver_Lib=/usr/lib/wsl/lib

# export TensorRT_Lib=${TENSORRT_LIB}
# export TensorRT_Inc=${TENSORRT_INCLUDE}
# export TensorRT_Bin=${TENSORRT_BIN}

# export CUDA_Lib=${CUDA_LIB}
# export CUDA_Inc=${CUDA_INCLUDE}
# export CUDA_Bin=${CUDA_BIN}

# export CUDNN_Lib=${CUDNN_LIB}

# Just used to distinguish the libspconv version, it doesn't affect the version of cuda used by your application
# For CUDA-11.x:    SPCONV_CUDA_VERSION=11.4
# For CUDA-12.x:    SPCONV_CUDA_VERSION=12.6
# export SPCONV_CUDA_VERSION=11.4
#* 12.6 是预编译库目录标签，nvcc 仍使用 CUDA_HOME 的 12.1。
#* 该库与当前运行库/驱动的兼容性还需真实稀疏卷积推理验证。
export SPCONV_CUDA_VERSION=12.6

# resnet50/resnet50-int8/swint
export DEBUG_MODEL=${DEBUG_MODEL:-resnet50}

#* 可选：直接指定包含 ONNX 模型及 build 目录的路径；留空时仍使用 model/$DEBUG_MODEL。
export DEBUG_MODEL_DIR=${DEBUG_MODEL_DIR:-}

# fp16/int8
export DEBUG_PRECISION=${DEBUG_PRECISION:-fp16}
export DEBUG_DATA=${DEBUG_DATA:-example-data}
export USE_Python=${USE_Python:-OFF}

# check the configuration path
# clean the configuration status
export ConfigurationStatus=Failed
if [ ! -f "${TensorRT_Bin}/trtexec" ]; then
    echo "Can not find ${TensorRT_Bin}/trtexec, there may be a mistake in the directory you configured."
    return
fi

if [ ! -f "${CUDA_Bin}/nvcc" ]; then
    echo "Can not find ${CUDA_Bin}/nvcc, there may be a mistake in the directory you configured."
    return
fi

echo "=========================================================="
echo "||  MODEL: $DEBUG_MODEL"
echo "||  MODEL DIR: ${DEBUG_MODEL_DIR:-model/$DEBUG_MODEL}"
echo "||  PRECISION: $DEBUG_PRECISION"
echo "||  DATA: $DEBUG_DATA"
echo "||  USEPython: $USE_Python"
echo "||"
echo "||  TensorRT: $TensorRT_Lib"
echo "||  CUDA: $CUDA_HOME"
echo "||  CUDNN: $CUDNN_Lib"
echo "=========================================================="

BuildDirectory=`pwd`/build

if [ "$USE_Python" == "ON" ]; then
    export Python_Inc=`python3 -c "import sysconfig;print(sysconfig.get_path('include'))"`
    export Python_Lib=`python3 -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))"`
    export Python_Soname=`python3 -c "import sysconfig;import re;print(re.sub('.a', '.so', sysconfig.get_config_var('LIBRARY')))"`
    echo Find Python_Inc: $Python_Inc
    echo Find Python_Lib: $Python_Lib
    echo Find Python_Soname: $Python_Soname
fi

export PATH=$TensorRT_Bin:$CUDA_Bin:$PATH
#* 优先加载 WSL2 的真实 NVIDIA 驱动库，避免运行时找不到 libcuda.so。
export LD_LIBRARY_PATH=$CUDA_Driver_Lib:$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:$BuildDirectory:$LD_LIBRARY_PATH
export PYTHONPATH=$BuildDirectory:$PYTHONPATH
export ConfigurationStatus=Success

if [ -f "tool/cudasm.sh" ]; then
    echo "Try to get the current device SM"
    . "tool/cudasm.sh"
    echo "Current CUDA SM: $cudasm"
fi

export CUDASM=$cudasm

echo Configuration done!
