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
# 校准
python qat/ptq.py \
  --config DAOcc/configs/nuscenes/occ3d/daocc_occ3d_nus_w_mask_custom.yaml \
  --ckpt /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/epoch_6_ema.pth \
  --calibrate_batch 300 \
  --save-path /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/epoch_6_ema_ptq.pth

#---------------------------------------------------------------#
# 导出INT8模型
cd /home/wys/wangyushen/forks/CUDA-DAOcc

ptq_model=/home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/epoch_6_ema_ptq.pth
example_data=/home/wys/wangyushen/data/Datasets/example/example-data/example-data.pth
output_dir=/home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/tensorrt-int8

python qat/export-camera.py \
  --ckpt "$ptq_model" \
  --example-data "$example_data" \
  --output-dir "$output_dir"

python qat/export-transfuser.py \
  --ckpt "$ptq_model" \
  --output-dir "$output_dir"

python qat/export-scn.py \
  --ckpt "$ptq_model" \
   --save "$output_dir/lidar.backbone.onnx"
#---------------------------------------------------------------#
# 安装 TensorRT
sudo dpkg -i \
  /home/wys/wangyushen/data/nv-tensorrt-local-repo-ubuntu2204-8.5.3-cuda-11.8_1.0-1_amd64.deb

# 构建tensorRT引擎
bash tool/build_trt_engine.sh \
  --model-dir /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/tensorrt-int8 \
  --precision int8

# 编译 CUDA-DAOcc 的 C++ 推理程序，并用示例数据运行
bash tool/run.sh \
  --model-dir /home/wys/wangyushen/data/Weights/daocc/occ3d_nus_w_mask/tensorrt-int8 \
  --data-dir /home/wys/wangyushen/data/Datasets/example/example-data \
  --precision int8

# 可视化
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