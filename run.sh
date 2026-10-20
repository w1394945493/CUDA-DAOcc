cd /home/wys/wangyushen/forks/CUDA-DAOcc

# 修改子模块地址，同时同步到本地 Git 配置
git submodule set-url DAOcc https://github.com/w1394945493/DAOcc.git

# 按主仓库锁定的提交更新子模块
git submodule update --init --recursive DAOcc
使用你 fork 的最新分支代码，需要再执行：
git -C DAOcc fetch origin
git -C DAOcc branch -r

# 让以后克隆 CUDA-DAOcc 时也使用你的地址和所选提交，需要在主仓库记录两项变更：
git add .gitmodules DAOcc
git commit -m "Use custom DAOcc submodule"
git push

#===============================================================#
# 安装包
PIP_CONFIG_FILE=/dev/null \
PIP_INDEX_URL=https://pypi.org/simple \
PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com \
python -m pip install \
  pytorch-quantization==2.1.3 \
  --timeout 30 \
  --retries 2
#----------------------------------------------------#
# 从源码编译pytorch-quantization，安装到当前环境
conda activate daocc
cd /home/wys/wangyushen/forks

git clone --depth 1 --branch release/8.6 \
  https://github.com/NVIDIA/TensorRT.git \
  TensorRT-quantization-src

cd TensorRT-quantization-src/tools/pytorch-quantization

export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"

python -m pip install -v \
  --no-build-isolation \
  --no-deps \
  --no-cache-dir \
  --force-reinstall \
  .
#===============================================================#


PIP_CONFIG_FILE=/dev/null \
PIP_INDEX_URL=https://pypi.org/simple \
PIP_EXTRA_INDEX_URL= \
python -m pip install onnx --timeout 30 --retries 2


PIP_CONFIG_FILE=/dev/null \
PIP_INDEX_URL=https://pypi.org/simple \
PIP_EXTRA_INDEX_URL= \
python -m pip install onnxsim --timeout 30 --retries 2

# PIP_CONFIG_FILE=/dev/null \
# PIP_INDEX_URL=https://pypi.org/simple \
# PIP_EXTRA_INDEX_URL= \
# python -m pip install onnxruntime --timeout 30 --retries 2

# 降级
PIP_CONFIG_FILE=/dev/null \
PIP_INDEX_URL=https://pypi.org/simple \
PIP_EXTRA_INDEX_URL= \
python -m pip install --force-reinstall \
  "onnxsim==0.4.10" \
  "onnxruntime==1.19.2"

conda activate daocc

PIP_CONFIG_FILE=/dev/null \
PIP_INDEX_URL=https://pypi.org/simple \
PIP_EXTRA_INDEX_URL= \
python -m pip install \
  --force-reinstall \
  --no-deps \
  "numpy==1.22.0"
#===============================================================#
# 1. 校准
# Pytorch FP32/16 模型 -> 统计INT8量化参数 -> 带量化器和scale的Pytorch模型
# 从训练集最多取 300 个batch，bs=1，进行校准：
#（1）将部分卷积、线性层替换成支持量化的模块；（2）输入真实样本并执行前向推理；
#（3）收集中间激活值的数值分布；（4）计算每个量化节点的INT8量化参数（scale/amax）；（5）将权重和量化参数保存到PTQ模型
python qat/ptq.py \
  --config DAOcc/configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_custom.yaml \
  --ckpt /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/epoch_6_ema.pth \
  --calibrate_batch 300 \
  --save-path /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/epoch_6_ema_ptq.pth

#---------------------------------------------------------------#
# 2. 导出INT8模型
# 拆分成五部分，各部分使用的执行后端不一致
cd /home/wys/wangyushen/forks/CUDA-DAOcc

ptq_model=/home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/epoch_6_ema_ptq.pth
example_data=/home/wys/wangyushen/data/Datasets/example/example-data/example-data.pth
output_dir=/home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/tensorrt-int8



# 2.1 相机分支：包含 六路图像 → resnet50 → FPN → 根据img_ref_points 采样 → 相机BEV特征
# example-data主要提供导出时需要的真实样例输入和形状信息。并非校准数据集
# 输出camera.backbone.onnx
python qat/export-camera.py \
  --ckpt "$ptq_model" \
  --example-data "$example_data" \
  --output-dir "$output_dir"

# 可视化
netron \
  /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/tensorrt-int8/camera.backbone.onnx \
  --host 0.0.0.0 \
  --port 8080

# 2.2 融合、decoder和两个Head
# 输出：fuser.onnx →  相机/lidar特征融合 + 共享Decoder
# head.object.onnx → CenterHead原始检测张量
# head.occ.onnx → occupancy分类结果
python qat/export-transfuser.py \
  --ckpt "$ptq_model" \
  --output-dir "$output_dir"

# 2.3 lidar稀疏卷积分支
# 输出：lidar.backbone.onnx → 表示：体素特征+体素坐标 → SparseEncoder → lidar BEV特征
# 该onnx后面不会转成普通TensorRT .plan,而是由项目中的稀疏卷积运行时读取。
# 注：LiDAR 稀疏卷积分支走专用运行时：lidar.backbone.xyz.onnx → 项目自己的onnx解析器，→ spconv::Engine → libspconv.so 的 CUDA 稀疏卷积 kernel 执行
python qat/export-scn.py \
  --ckpt "$ptq_model" \
  --save "$output_dir/lidar.backbone.onnx"

#---------------------------------------------------------------#
# 安装 TensorRT
sudo dpkg -i \
  /home/wys/wangyushen/data/nv-tensorrt-local-repo-ubuntu2204-8.5.3-cuda-11.8_1.0-1_amd64.deb

# 3. 构建tensorRT引擎
# 通过trtexec将普通onnx转换成GPU可直接执行的TensorRT engine
# camera.backbone.onnx → build/camera.backbone.plan
# fuser.onnx           → build/fuser.plan
# head.object.onnx     → build/head.object.plan
# head.occ.onnx        → build/head.occ.plan
# 精度说明：
# （1）. camera.backbone 和 fuser 内部采用 INT8+FP16 混合计算，模块输入输出保持 FP16。
# （2）. LiDAR SparseEncoder 由 libspconv 执行，INT8 模式下内部采用混合精度，输出 FP16 BEV 特征。
# 3. head.object 使用 FP16 计算并输出 FP16 检测参数，随后由 CUDA 解码为浮点检测框。
# 4. head.occ 主要使用 FP16 计算类别分数，末尾通过 argmax 输出 INT32 类别编号。
# 5. INT32 只是 occupancy 最终语义标签的数据类型，不代表 occupancy 卷积使用 INT32 计算。
bash tool/build_trt_engine.sh \
  --model-dir /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/tensorrt-int8 \
  --precision int8

# 4. 编译 CUDA-DAOcc 的 C++ 推理程序，并用示例数据运行
# 先执行bash src/onnx/make_pb.sh  cmake ..  make -j  命令，生成build/bevfusion C++推理程序（可执行程序）
bash tool/run.sh \
  --model-dir /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/tensorrt-int8 \
  --data-dir /home/wys/wangyushen/data/Datasets/example/example-data \
  --precision int8

# 5. 可视化
env -u LD_LIBRARY_PATH \
  QT_QPA_PLATFORM=xcb \
  python -c \
  'from PyQt5.QtWidgets import QApplication; app=QApplication([]); print("PyQt5 xcb 正常")'

env -u LD_LIBRARY_PATH \
  QT_QPA_PLATFORM=xcb \
  python DAOcc/tools/vis_occ3d.py \
    /home/wys/wangyushen/forks/CUDA-DAOcc/occ_pred.txt \
    --boxes /home/wys/wangyushen/forks/CUDA-DAOcc/object_pred.txt \
    --lidar2ego /home/wys/wangyushen/data/Datasets/example/example-data/example-data.pth