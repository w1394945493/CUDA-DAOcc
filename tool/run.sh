#!/bin/bash
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

#* 先解析命令行参数，使模型目录、输入目录和精度优先于环境配置。
while [ $# -gt 0 ]; do
    case "$1" in
        --model-dir)
            if [ $# -lt 2 ]; then
                echo "Error: --model-dir requires a path."
                exit 2
            fi
            DEBUG_MODEL_DIR=$2
            shift 2
            ;;
        --data-dir)
            if [ $# -lt 2 ]; then
                echo "Error: --data-dir requires a path."
                exit 2
            fi
            DEBUG_DATA=$2
            shift 2
            ;;
        --precision)
            if [ $# -lt 2 ]; then
                echo "Error: --precision requires fp16 or int8."
                exit 2
            fi
            DEBUG_PRECISION=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash tool/run.sh [--model-dir PATH] [--data-dir PATH] [--precision fp16|int8]"
            exit 0
            ;;
        *)
            echo "Error: unknown argument: $1"
            exit 2
            ;;
    esac
done

# configure the environment
. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit 1
fi

if [ "$DEBUG_PRECISION" != "fp16" ] && [ "$DEBUG_PRECISION" != "int8" ]; then
    echo "Error: --precision must be fp16 or int8."
    exit 2
fi

set -e

#* 使用本机 protoc 重新生成 ONNX 解析代码，避免仓库旧版 pb 文件与系统 protobuf 不兼容。
bash src/onnx/make_pb.sh

mkdir -p build

cd build
cmake ..
make -j

cd ..

#* 第四个参数传入可选模型目录；未配置时由 C++ 继续使用 model/$DEBUG_MODEL。
./build/bevfusion "$DEBUG_DATA" "$DEBUG_MODEL" "$DEBUG_PRECISION" "$DEBUG_MODEL_DIR"
