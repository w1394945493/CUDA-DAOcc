/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 # SPDX-FileCopyrightText: Copyright (c) 2025 Beijing Mechanical Equipment Institute. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "bevfusion.hpp"

#include <numeric>

#include "common/check.hpp"
#include "common/timer.hpp"

namespace bevfusion {

class CoreImplement : public Core {
public:
  virtual ~CoreImplement() {
    if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
    if (lidar_points_host_) checkRuntime(cudaFreeHost(lidar_points_host_));
  }

  bool init(const CoreParameter& param) {
    camera_backbone_ = camera::create_backbone(param.camera_model);
    if (camera_backbone_ == nullptr) {
      printf("Failed to create camera backbone.\n");
      return false;
    }

    // camera_bevpool_ =
    //     camera::create_bevpool(camera_backbone_->camera_shape(), param.geometry.geometry_dim.x, param.geometry.geometry_dim.y);
    // if (camera_bevpool_ == nullptr) {
    //   printf("Failed to create camera bevpool.\n");
    //   return false;
    // }

    // camera_vtransform_ = camera::create_vtransform(param.camera_vtransform);
    // if (camera_vtransform_ == nullptr) {
    //   printf("Failed to create camera vtransform.\n");
    //   return false;
    // }

    transfusion_ = fuser::create_transfusion(param.transfusion);
    if (transfusion_ == nullptr) {
      printf("Failed to create transfusion.\n");
      return false;
    }

    // transbbox_ = head::transbbox::create_transbbox(param.transbbox);
    // if (transbbox_ == nullptr) {
    //   printf("Failed to create head transbbox.\n");
    //   return false;
    // }


    lidar_scn_ = lidar::create_scn(param.lidar_scn);
    if (lidar_scn_ == nullptr) {
      printf("Failed to create lidar scn.\n");
      return false;
    }

    occ_head_ = head::occ::create_bevocchead(param.occ);
    if (occ_head_ == nullptr) {
      printf("Failed to create occ head.\n");
      return false;
    }

    object_head_ = head::object::create_centerhead(param.object);
    if (object_head_ == nullptr) {
      printf("Failed to create object head.\n");
      return false;
    }

    normalizer_ = camera::create_normalization(param.normalize);
    if (normalizer_ == nullptr) {
      printf("Failed to create normalizer.\n");
      return false;
    }

    // camera_depth_ = camera::create_depth(param.normalize.output_width, param.normalize.output_height, param.normalize.num_camera);
    // if (camera_depth_ == nullptr) {
    //   printf("Failed to create depth.\n");
    //   return false;
    // }

    // camera_geometry_ = camera::create_geometry(param.geometry);
    // if (camera_geometry_ == nullptr) {
    //   printf("Failed to create geometry.\n");
    //   return false;
    // }

    capacity_points_ = 300000;
    bytes_capacity_points_ = capacity_points_ * param.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
    checkRuntime(cudaMallocHost(&lidar_points_host_, bytes_capacity_points_));
    param_ = param;
    return true;
  }

  void forward_only(const void* camera_images, const nvtype::half* img_ref_points, const nvtype::half* lidar_points, int num_points,
                    void* stream, bool do_normalization) {
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
    }

    num_points = std::min(cappoints, num_points);

    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    size_t bytes_points = num_points * param_.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));

    const nvtype::half* lidar_feature = this->lidar_scn_->forward(lidar_points_device_, num_points, stream);
    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
    }
    // const nvtype::half* depth = this->camera_depth_->forward(lidar_points_device_, num_points, 5, stream);

    this->camera_backbone_->forward(normed_images, img_ref_points, stream);
    // const nvtype::half* camera_bev = this->camera_bevpool_->forward(
    //     this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
    //     this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);

    // const nvtype::half* camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(this->camera_backbone_->feature(), lidar_feature, stream);
    // return this->transbbox_->forward(fusion_feature, param_.transbbox.confidence_threshold, stream,
    //                                  param_.transbbox.sorted_bboxes);
    object_output_ = this->object_head_->forward(fusion_feature, stream);
    occ_output_ = this->occ_head_->forward(fusion_feature, stream);
  }

  // *=================== 带分阶段计时的推理入口 ===================
  // * 各阶段使用同一 CUDA stream 上的 EventTimer；stop() 会同步结束事件并打印耗时。
  // 这里的计时用于分析部署流水线，末尾 Total 只累加五个模块阶段。
  void forward_timer(const void* camera_images, const nvtype::half* img_ref_points, const nvtype::half* lidar_points, int num_points,
                     void* stream, bool do_normalization) {
    // *=================== 限制点云数量并准备计时 ===================
    // 点数超过预分配容量时截断，避免写出 LiDAR 输入缓冲区。
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
    }

    num_points = std::min(cappoints, num_points);

    // 每次调用都覆盖上一条记录；传输和图像预处理单独打印。
    // 静默测速时不打印边框，原输出保留如下。
    // printf("==================DAOcc==================\n");
    if (timer_verbose_) printf("==================DAOcc==================\n");
    last_latency_ = ModuleLatency{};
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    timer_.start(_stream);

    // *=================== LiDAR 输入复制 ===================
    // * 点云先复制到锁页主机内存，再异步传入 GPU；这部分不计入末尾 Total。
    size_t bytes_points = num_points * param_.lidar_scn.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));
    timer_.stop("[NoSt] CopyLidar");

    // *=================== 相机输入复制与归一化 ===================
    // do_normalization 为 true 时，normalizer 会上传六路图像并在 GPU 上完成归一化。
    // * ImageNrom 合并了图像 H2D 和预处理耗时，也不计入末尾 Total。
    nvtype::half* normed_images = (nvtype::half*)camera_images;
    if (do_normalization) {
      timer_.start(_stream);
      normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
      timer_.stop("[NoSt] ImageNrom");
    }

    // *=================== LiDAR Backbone ===================
    // * lidar_scn_->forward 包含体素化和稀疏卷积，不能把该项解释为只有稀疏卷积耗时。
    timer_.start(_stream);
    const nvtype::half* lidar_feature = this->lidar_scn_->forward(lidar_points_device_, num_points, stream);
    last_latency_.lidar_ms = timer_.stop("Lidar Backbone");

    // timer_.start(_stream);
    // const nvtype::half* depth = this->camera_depth_->forward(lidar_points_device_, num_points, 5, stream);
    // timer_.stop("Camera Depth");

    // *=================== Camera Backbone ===================
    // 输入为归一化图像和预先计算的图像参考点，输出相机 BEV 特征。
    timer_.start(_stream);
    this->camera_backbone_->forward(normed_images, img_ref_points, stream);
    last_latency_.camera_ms = timer_.stop("Camera Backbone");

    // timer_.start(_stream);
    // const nvtype::half* camera_bev = this->camera_bevpool_->forward(
    //     this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
    //     this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);
    // timer_.stop("Camera Bevpool");

    // timer_.start(_stream);
    // const nvtype::half* camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
    // timer_.stop("VTransform");

    // *=================== 特征融合与 Decoder ===================
    // 相机和 LiDAR 特征以 GPU 指针传入融合引擎，结果继续保留在 GPU 上。
    timer_.start(_stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(this->camera_backbone_->feature(), lidar_feature, stream);
    last_latency_.fusion_decoder_ms = timer_.stop("Transfusion");

    // *=================== Occupancy Head ===================
    // * 当前 Head Occ 包含 TensorRT 推理、完整体素结果 D2H 复制和同步；不是纯引擎耗时。
    timer_.start(_stream);
    occ_output_ = this->occ_head_->forward(fusion_feature, stream);
    last_latency_.occ_head_ms = timer_.stop("Head Occ");

    // *=================== Object Head ===================
    // * 当前 Head Object 包含 TensorRT 推理以及检测后处理；其中 CPU 排序和数据往返
    // 无法由这里的 CUDA Event 完整拆出，若要测纯引擎耗时应在 head-object.cu 内部计时。
    timer_.start(_stream);
    object_output_ = this->object_head_->forward(fusion_feature, stream);
    last_latency_.object_head_ms = timer_.stop("Head Object");

    // *=================== 阶段耗时汇总 ===================
    // * Total 是五项记录值之和，不含 CopyLidar、ImageNrom、文件读取与结果保存；
    // 由于两个 Head 含有回传/后处理，也不能直接称为“纯模型推理延迟”。
    last_latency_.total_ms = last_latency_.lidar_ms + last_latency_.camera_ms +
                             last_latency_.fusion_decoder_ms + last_latency_.occ_head_ms +
                             last_latency_.object_head_ms;
    // 静默模式只更新统计值，不逐帧打印汇总。
    // printf("Total: %.3f ms\n", last_latency_.total_ms);
    // printf("=========================================\n");
    if (timer_verbose_) {
      printf("Total: %.3f ms\n", last_latency_.total_ms);
      printf("=========================================\n");
    }
    // return output;
  }

  virtual void forward(const unsigned char** camera_images, const nvtype::half* img_ref_points, const nvtype::half* lidar_points, int num_points, void* stream) override {
    if (enable_timer_) {
      this->forward_timer(camera_images, img_ref_points, lidar_points, num_points, stream, true);
    } else {
      this->forward_only(camera_images, img_ref_points, lidar_points, num_points, stream, true);
    }
  }

  virtual void forward_no_normalize(const nvtype::half* camera_normed_images_device, const nvtype::half* img_ref_points, const nvtype::half* lidar_points, int num_points, void* stream) override {
    if (enable_timer_) {
      this->forward_timer(camera_normed_images_device, img_ref_points, lidar_points, num_points, stream, false);
    } else {
      this->forward_only(camera_normed_images_device, img_ref_points, lidar_points, num_points, stream, false);
    }
  }

  virtual void set_timer(bool enable) override { enable_timer_ = enable; }
  // * 同时控制各阶段和汇总日志，不改变计时边界。
  virtual void set_timer_verbose(bool verbose) override {
    timer_verbose_ = verbose;
    timer_.set_verbose(verbose);
  }

  virtual ModuleLatency last_module_latency() const override { return last_latency_; }

  virtual void print() override {
    camera_backbone_->print();
    // camera_vtransform_->print();
    transfusion_->print();
    // transbbox_->print();
    occ_head_->print();
    object_head_->print();
  }

  // virtual void update(const float* camera2lidar, const float* camera_intrinsics, const float* lidar2image,
  //                     const float* img_aug_matrix, void* stream) override {
  //   camera_depth_->update(img_aug_matrix, lidar2image, stream);
  //   camera_geometry_->update(camera2lidar, camera_intrinsics, img_aug_matrix, stream);
  // }

  // virtual void free_excess_memory() override { camera_geometry_->free_excess_memory(); }

private:
  CoreParameter param_;
  nv::EventTimer timer_;
  ModuleLatency last_latency_;
  nvtype::half* lidar_points_device_ = nullptr;
  nvtype::half* lidar_points_host_ = nullptr;
  size_t capacity_points_ = 0;
  size_t bytes_capacity_points_ = 0;

  std::shared_ptr<camera::Normalization> normalizer_;
  std::shared_ptr<camera::Backbone> camera_backbone_;
  // std::shared_ptr<camera::BEVPool> camera_bevpool_;
  // std::shared_ptr<camera::VTransform> camera_vtransform_;
  // std::shared_ptr<camera::Depth> camera_depth_;
  // std::shared_ptr<camera::Geometry> camera_geometry_;
  std::shared_ptr<lidar::SCN> lidar_scn_;
  std::shared_ptr<fuser::Transfusion> transfusion_;
  // std::shared_ptr<head::transbbox::TransBBox> transbbox_;
  std::shared_ptr<head::occ::BevOccHead> occ_head_;
  std::shared_ptr<head::object::CenterHead> object_head_;
  float confidence_threshold_ = 0;
  bool enable_timer_ = false;
  bool timer_verbose_ = true;  // 原 C++ 调用方默认继续打印。
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace bevfusion
