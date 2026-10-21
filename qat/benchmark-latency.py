"""使用 DAOcc 原生数据加载流程，统计 CUDA 部署模型的逐帧模块耗时。"""

import argparse
import csv
import json
import logging
import pickle
import ctypes
import multiprocessing as mp
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path
import sys


METRICS = (
    "lidar_ms", "camera_ms", "fusion_decoder_ms",
    "occ_head_ms", "object_head_ms", "total_ms",
)


def parse_args():
    # *=================== 命令行参数 ===================
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="DAOcc YAML 配置路径")
    parser.add_argument("--daocc-root", default=None, help="已编译自定义扩展的 DAOcc 目录；默认使用子仓库")
    # 原 TensorRT 默认行为保留，PyTorch 后端使用原始 checkpoint。
    # parser.add_argument("--model-dir", required=True, help="包含 build/*.plan 和 LiDAR ONNX 的目录")
    # parser.add_argument("--precision", choices=("fp16", "int8"), default="int8")
    parser.add_argument("--backend", choices=("tensorrt", "pytorch"), default="tensorrt")
    parser.add_argument("--model-dir", help="TensorRT 引擎与 LiDAR ONNX 目录")
    parser.add_argument("--ckpt", help="PyTorch 原模型权重（非 PTQ 模型对象）")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--warmup", type=int, default=20, help="首帧预热次数，不计入统计")
    # 原参数名称保留兼容旧命令，正数现在表示准确的正式推理次数。
    # parser.add_argument("--max-samples", type=int, default=0, help="最多测量帧数；0 表示整个 split")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="正式推理次数：足够时等间隔取样，不足时循环取样；0 表示完整遍历一次")
    parser.add_argument("--workers", type=int, default=None, help="覆盖数据加载进程数")
    parser.add_argument("--device", type=int, default=0, help="CUDA 设备编号")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latency-csv", default="latency.csv", help="逐帧 CSV 输出路径")
    # * 指定目录后逐次保存正式推理结果；不指定时保持只测速的行为。
    parser.add_argument("--out-dir", default=None,
                        help="每次正式推理的预测 PKL 保存目录；预热结果不保存")
    # 可选统计原始网络参数，不加载权重，也不将额外模型放到 GPU。
    parser.add_argument("--count-params", action="store_true",
                        help="按配置统计五模块参数量，写入汇总 JSON（非 TensorRT 优化后权重数量）")
    # 显存采样独立于 CUDA Event 计时；间隔越小，监控开销越高。
    parser.add_argument("--memory-interval", type=float, default=0.1,
                        help="GPU 显存采样间隔，单位秒，默认 0.1")
    args = parser.parse_args()
    # 精度默认值按后端确定，避免把原 PyTorch 推理错误标为 INT8。
    args.precision = args.precision or ("int8" if args.backend == "tensorrt" else "fp32")
    if args.backend == "pytorch" and (not args.ckpt or args.precision == "int8"):
        parser.error("PyTorch 后端需要 --ckpt，且仅支持 --precision fp32/fp16")
    if args.backend == "tensorrt" and (not args.model_dir or args.precision == "fp32"):
        parser.error("TensorRT 后端需要 --model-dir，且仅支持 --precision int8/fp16")
    if not 0 < args.memory_interval < float("inf"):
        parser.error("memory-interval 必须是有限正数")
    if args.warmup < 0 or args.max_samples < 0 or (args.workers is not None and args.workers < 0):
        parser.error("warmup、max-samples、workers 不能为负数")
    return args


def gpu_memory_worker(pid, pci_bus_id, interval, stop, connection):
    # * 使用独立进程采样，避免 C++ 推理持有 Python GIL 时监控线程无法运行。
    peaks = {"process_peak_mib": None, "device_peak_mib": None}
    counts = {"process_samples": 0, "device_samples": 0}
    errors = {}
    ready_sent = False

    def query(fields, kind):
        command = ["nvidia-smi", "-i", pci_bus_id,
                   "--query-" + kind + "=" + fields, "--format=csv,noheader,nounits"]
        return subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=5)

    try:
        while True:
            started = time.monotonic()
            # WSL 的进程显存可能为 N/A；不伪造 0，也不冒充为模型进程显存。
            try:
                matches = [line.split(",") for line in query("pid,used_gpu_memory", "compute-apps").splitlines()
                           if line.split(",")[0].strip() == str(pid)]
                values = [float(row[1].strip()) for row in matches]
                if values:
                    value = sum(values)
                    peaks["process_peak_mib"] = max(peaks["process_peak_mib"] or 0, value)
                    counts["process_samples"] += 1
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors["process"] = str(exc)
            try:
                value = float(query("memory.used", "gpu").strip())
                peaks["device_peak_mib"] = max(peaks["device_peak_mib"] or 0, value)
                counts["device_samples"] += 1
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                errors["device"] = str(exc)
            # 第一次采样完成后再让主进程开始推理，保证短测试也能取得基线。
            if not sum(counts.values()) and errors.get("device"):
                connection.send(("ready", None))
                break
            if not ready_sent:
                connection.send(("ready", None))
                ready_sent = True
            if stop.wait(max(0, interval - (time.monotonic() - started))):
                break
    finally:
        connection.send(("result", dict(**peaks, **counts, errors=errors)))
        connection.close()


class GPUMemoryMonitor:
    # * 同时保存进程和整卡采样峰值；整卡值可能包含其他应用，不能混为一谈。
    def __init__(self, device, interval):
        self.device, self.interval = device, interval
        self.result = None
        self.baseline_mib = None
        self.baseline_error = None

    # def __enter__(self):
    # PCI 地址解析独立出来，基线与后续采样确保查询同一张卡。
    def resolve_device(self):
        # CUDA 驱动按逻辑设备编号解析 PCI 地址，兼容 CUDA_VISIBLE_DEVICES 重排。
        driver = ctypes.CDLL("libcuda.so.1")
        bus = ctypes.create_string_buffer(32)
        if driver.cuInit(0) or driver.cuDeviceGetPCIBusId(bus, len(bus), self.device):
            raise RuntimeError("无法获取 CUDA 设备 PCI 地址以监控显存")
        self.bus = bus.value.decode()

    def capture_baseline(self):
        # * 必须在加载部署引擎前调用；已有 CUDA 上下文及后台应用计入基线。
        self.resolve_device()
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "-i", self.bus, "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.PIPE, timeout=5)
            self.baseline_mib = float(output.strip())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            # 基线失败时不以零替代，否则会把整卡占用误认为部署增量。
            self.baseline_error = str(exc)

    def __enter__(self):
        self.resolve_device()
        context = mp.get_context("spawn")
        self.stop = context.Event()
        self.receiver, sender = context.Pipe(duplex=False)
        self.process = context.Process(target=gpu_memory_worker,
                                       args=(os.getpid(), self.bus, self.interval, self.stop, sender))
        self.process.start()
        sender.close()
        if not self.receiver.poll(15):
            self.process.terminate()
            self.process.join()
            raise RuntimeError("GPU 显存监控启动超时")
        self.receiver.recv()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.process.join(timeout=15)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        if self.receiver.poll():
            _, self.result = self.receiver.recv()
        self.receiver.close()
        self.result = self.result or {"process_peak_mib": None, "device_peak_mib": None}
        self.result.update(sample_interval_seconds=self.interval, pci_bus_id=self.bus,
                           method="nvidia-smi periodic sampling; short peaks may be missed",
                           scope="engines already loaded; warmup and measured inference loop",
                           device_scope="whole GPU including other processes")
        # * 优先使用进程显存；不可用时仅提供整卡基线差值估计，保留原始值供核对。
        self.result.update(select_memory_metric(self.result, self.baseline_mib))
        self.result.update(
            baseline_error=self.baseline_error,
            baseline_scope="before engine loading; existing CUDA context included in baseline",
            estimate_note="device peak minus baseline; affected by other processes; sampled, not exact")


def select_memory_metric(result, baseline_mib):
    # 负增量不强制截为零：它提示后台显存变化或测量条件不稳定。
    peak = result.get("device_peak_mib")
    delta = peak - baseline_mib if peak is not None and baseline_mib is not None else None
    process_peak = result.get("process_peak_mib")
    if process_peak is not None:
        selected, method = process_peak, "process_sampled_peak"
    elif delta is not None and delta >= 0:
        selected, method = delta, "device_delta_estimate"
    else:
        selected, method = None, "unavailable"
    return dict(device_baseline_mib=baseline_mib, device_peak_delta_mib=delta,
                reported_peak_mib=selected, reported_method=method,
                is_estimate=(method == "device_delta_estimate"))


def select_sample_indices(dataset_size, requested_count):
    # * 按原数据集索引确定取样顺序，结果可复现；预热不占用这些正式样本。
    if dataset_size <= 0 or requested_count < 0:
        raise ValueError("数据集大小必须为正，指定次数不能为负")
    count = requested_count or dataset_size
    if count > dataset_size:
        # 完整遍历后从头循环，最后一轮仅取所需数量。
        return [i % dataset_size for i in range(count)], "cyclic"
    if count == 1:
        return [0], "uniform"
    # 等间隔覆盖首尾；整数除法避免浮点舍入导致重复索引。
    return [i * (dataset_size - 1) // (count - 1) for i in range(count)], "uniform"


def count_module_parameters(model):
    # * 按导出脚本实际调用的子模块分组，避免导出包装器持有整个模型而重复计数。
    camera = model.encoders["camera"]
    occ = model.heads["occ"]
    groups = {
        "lidar": [model.encoders["lidar"]["backbone"]],
        "camera": [camera["backbone"], camera["neck"],
                   camera["vtransform"].transfer_conv, camera["vtransform"].down_sample],
        "fusion_decoder": [model.fuser, model.decoder["backbone"], model.decoder["neck"]],
        "occ_head": [occ.coordinate_transform.transfer_conv, occ.final_conv, occ.predicter],
        "object_head": [model.heads["object"].shared_conv, model.heads["object"].task_heads],
    }

    def describe(parameters):
        # parameters 不包含 BN 均值/方差、参考点等 buffer；冻结权重仍计入总参数。
        parameters = list(parameters)
        total = sum(p.numel() for p in parameters)
        trainable = sum(p.numel() for p in parameters if p.requires_grad)
        return dict(parameters=total, parameters_m=total / 1e6,
                    trainable_parameters=trainable, frozen_parameters=total - trainable)

    modules, covered = {}, {}
    for name, parts in groups.items():
        unique = {id(p): p for part in parts if part is not None for p in part.parameters()}
        modules[name] = describe(unique.values())
        covered.update(unique)
    # 全局按对象去重；额外参数单独列出，不悄悄归入任一部署模块。
    all_parameters = dict(model.named_parameters())
    unassigned = {name: p for name, p in all_parameters.items() if id(p) not in covered}
    return dict(
        source="YAML PyTorch architecture before PTQ/BN fusion; excludes buffers",
        trainable_scope="requires_grad at model construction; no checkpoint loaded",
        modules=modules, total=describe(covered.values()),
        full_model=describe(all_parameters.values()),
        unassigned=describe(unassigned.values()), unassigned_parameter_names=list(unassigned),
    )


class PyTorchBackend:
    # * 使用原模型 forward，不重写网络；仅在五阶段边界记录 CUDA Event。
    def __init__(self, cfg, checkpoint, precision, device):
        import torch
        from mmcv.runner import load_checkpoint, wrap_fp16_model
        from mmdet3d.models import build_model
        self.torch, self.device, self.enabled = torch, device, False
        model_cfg = deepcopy(cfg.model)

        def clear_pretrained(value):
            # checkpoint 已提供权重，禁止构造模型时额外下载预训练 backbone。
            if isinstance(value, dict):
                if "pretrained" in value:
                    value["pretrained"] = None
                init = value.get("init_cfg")
                if isinstance(init, dict) and init.get("type") == "Pretrained":
                    value["init_cfg"] = None
                elif isinstance(init, list):
                    value["init_cfg"] = [x for x in init if x.get("type") != "Pretrained"] or None
                for child in value.values():
                    clear_pretrained(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    clear_pretrained(child)

        clear_pretrained(model_cfg)
        previous_disable = logging.root.manager.disable
        try:
            logging.disable(max(previous_disable, logging.INFO))
            self.model = build_model(model_cfg)
        finally:
            logging.disable(previous_disable)
        load_checkpoint(self.model, str(Path(checkpoint).expanduser()), map_location="cpu")
        if precision == "fp16":
            # 与原 DAOcc 测试入口相同，通过 MMCV 控制 FP16 模块与输入转换。
            wrap_fp16_model(self.model)
        self.model.cuda(device).eval()
        self.events = {key: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                       for key in METRICS[:-1]}
        self.timing = {}

        def wrap(owner, name, key, start=True, finish=True, cpu_boxes=False):
            original = getattr(owner, name)

            def measured(*args, **kwargs):
                if self.enabled and start:
                    self.events[key][0].record()
                result = original(*args, **kwargs)
                if cpu_boxes:
                    # 原 forward 随后也会转 CPU；提前放在检测头计时内，包含回传。
                    result = [(boxes.to("cpu"), scores.cpu(), labels.cpu())
                              for boxes, scores, labels in result]
                if self.enabled and finish:
                    begin, end = self.events[key]
                    end.record()
                    end.synchronize()
                    self.timing[key] = begin.elapsed_time(end)
                return result

            setattr(owner, name, measured)

        wrap(self.model, "extract_lidar_features", "lidar_ms")
        wrap(self.model, "extract_camera_features", "camera_ms")
        # 连续计量 fuser 到 decoder.neck，包含完整融合与 Decoder。
        if self.model.fuser is not None:
            wrap(self.model.fuser, "forward", "fusion_decoder_ms", finish=False)
        else:
            wrap(self.model.decoder["backbone"], "forward", "fusion_decoder_ms", finish=False)
        wrap(self.model.decoder["neck"], "forward", "fusion_decoder_ms", start=False)
        wrap(self.model.heads["object"], "forward", "object_head_ms", finish=False)
        wrap(self.model.heads["object"], "get_bboxes", "object_head_ms", start=False, cpu_boxes=True)
        wrap(self.model.heads["occ"], "forward", "occ_head_ms", finish=False)
        wrap(self.model.heads["occ"], "get_occ", "occ_head_ms", start=False)

    def set_timer(self, enable):
        self.enabled = enable

    def set_timer_verbose(self, verbose):
        pass  # 统一接口，PyTorch 计时不逐帧打印。

    def prepare(self, data):
        from mmcv.parallel import scatter
        # 与原 MMDataParallel 相同的 DataContainer 解包与 GPU 搬运，在模块计时外完成。
        return scatter(data, [self.device])[0]

    def forward(self, data, refs=None, points=None, with_normalization=False):
        self.timing = {}
        result = self.model(**data)
        if self.enabled:
            if set(self.timing) != set(METRICS[:-1]):
                raise RuntimeError("原模型未经过完整五阶段，无法输出对应模块耗时")
            self.timing["total_ms"] = sum(self.timing.values())
        return result[0]

    def last_module_latency(self):
        return self.timing


def prediction_scene_names(dataset):
    # * 仅启用预测保存时读取一次 nuScenes 数据表，按真实场景归类，不从文件名猜测。
    table_dir = Path(dataset.dataset_root) / dataset.version
    with (table_dir / "scene.json").open(encoding="utf-8") as handle:
        scenes = {scene["token"]: scene["name"] for scene in json.load(handle)}
    with (table_dir / "sample.json").open(encoding="utf-8") as handle:
        samples = {sample["token"]: scenes[sample["scene_token"]] for sample in json.load(handle)}
    # 提前校验全部样本，避免推理中途才发现场景映射缺失。
    for info in dataset.data_infos:
        name = samples[info["token"]]
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ValueError(f"无效的场景目录名: {name!r}")
    return samples


# def save_prediction(out_dir, prediction, index, sample_index, info, timing, np):
def save_prediction(out_dir, prediction, index, sample_index, info, timing, np, scene_name):
    # * 文件名只使用样本 token；重复测量覆盖同一帧，保留最后一次预测。
    token = str(info["token"])
    if not token or token in (".", "..") or "/" in token or "\\" in token:
        raise ValueError(f"无效的 sample token: {token!r}")
    # 原平铺路径保留；现在同一场景的帧统一保存到场景名子目录。
    # path = out_dir / f"{index:06d}-{token}.pkl"
    if not scene_name or scene_name in (".", "..") or "/" in scene_name or "\\" in scene_name:
        raise ValueError(f"无效的场景目录名: {scene_name!r}")
    scene_dir = out_dir / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    # path = scene_dir / f"{index:06d}-{token}.pkl"
    path = scene_dir / f"{token}.pkl"
    # * 沿用 test-mAP-for-cuda.py 转为原 DAOcc 预测字段，供 visualize.py 直接读取。
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    # * 原生 PyTorch 输出已符合 DAOcc 格式，直接保存，不能再次减去半个框高。
    if "boxes_3d" in prediction:
        payload = dict(prediction)
        payload.update(format_version=2, sample_token=token, sample_index=sample_index,
                       measurement_index=index, timestamp=info.get("timestamp"),
                       scene_name=scene_name, latency_ms=timing,
                       output_format="DAOcc prediction fields; PyTorch predictions")
    else:
        boxes = torch.from_numpy(np.array(prediction["det_outputs"], dtype=np.float32, copy=True))
        if boxes.ndim != 2 or boxes.shape[1] != 11:
            raise ValueError(f"检测输出应为 (N, 11)，实际为 {tuple(boxes.shape)}")
        # 部署输出 z 为框中心高度，DAOcc 的 LiDARInstance3DBoxes 使用底面中心。
        # 仅转换高度基准，不作 LiDAR/自车坐标系变换；复制后修改避免影响原始输出。
        boxes[:, 2] -= boxes[:, 5] * 0.5
        occupancy = np.array(prediction["occ_outputs"], copy=True)
        if occupancy.size != 200 * 200 * 16:
            raise ValueError(f"Occ3D 输出应包含 640000 个体素，实际为 {occupancy.size}")
        # 原实现保存部署接口的原始字段，注释保留；不重复存储两份 occupancy。
        payload = dict(
            # format_version=1, sample_token=token, sample_index=sample_index,
            format_version=2, sample_token=token, sample_index=sample_index,
            measurement_index=index, timestamp=info.get("timestamp"),
            scene_name=scene_name,  # PKL 内同时记录场景，便于脱离目录后识别。
            # det_outputs=np.array(prediction["det_outputs"], copy=True),
            # occ_outputs=np.array(prediction["occ_outputs"], copy=True),
            # det_columns=["x", "y", "z", "w", "l", "h", "yaw", "vx", "vy", "class_id", "score"],
            # output_format="raw libpybev output; occupancy is flattened, no coordinate conversion",
            boxes_3d=LiDARInstance3DBoxes(boxes[:, :9].clone(), box_dim=9),
            scores_3d=boxes[:, 10].clone(),
            labels_3d=boxes[:, 9].to(torch.int64),
            occ_pred=occupancy.reshape(200, 200, 16),
            output_format="DAOcc prediction fields; TensorRT predictions; boxes use bottom-center z",
            latency_ms=timing,
        )
    # 先写临时文件再替换，避免中断后留下看似完整的半个 PKL 文件。
    temporary = path.with_suffix(".pkl.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(path)


def prepare_inputs(data, view_transform, np):
    # * 沿用 test-mAP-for-cuda.py 的输入准备流程；图像已由 dataset pipeline 归一化。
    images = np.ascontiguousarray(data["img"].data[0].numpy())
    points = np.ascontiguousarray(data["points"].data[0][0].half().numpy())
    camera_intrinsics = data["camera_intrinsics"].data[0]
    camera2lidar = data["camera2lidar"].data[0]
    img_aug = data["img_aug_matrix"].data[0]
    lidar_aug = data["lidar_aug_matrix"].data[0]

    # 根据当前帧标定与增强矩阵生成参考点，不复用示例数据中的固定参考点。
    refs, mask = view_transform.point_sampling(
        camera2lidar, camera_intrinsics[..., :3, :3],
        img_aug[..., :3, :3], img_aug[..., :3, 3], lidar_aug,
    )
    batch, count, _ = refs.shape
    refs = 2 * (refs * mask).view(batch, count, 1, 1, 3) - 1
    refs += (~mask).view(batch, count, 1, 1, 1) * 10
    return images, np.ascontiguousarray(refs.numpy()), points


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    # * 可指定现有已编译扩展的 DAOcc 目录，避免为测速重复编译一份训练框架。
    daocc_root = Path(args.daocc_root).expanduser().resolve() if args.daocc_root else root / "DAOcc"
    if not (daocc_root / "mmdet3d").is_dir():
        raise FileNotFoundError(f"DAOcc 目录中缺少 mmdet3d: {daocc_root}")
    sys.path.insert(0, str(daocc_root))
    sys.path.insert(0, str(root / "build"))
    import numpy as np
    import torch
    from tqdm import tqdm
    # Subset 只重排索引，继续调用原数据集的加载与预处理流程。
    from torch.utils.data import Subset
    # import libpybev
    # PyTorch 模式不依赖已编译的 TensorRT Python 扩展。
    if args.backend == "tensorrt":
        import libpybev
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import BEVTransformV2
    from mmdet3d.utils import recursive_eval

    torch.cuda.set_device(args.device)
    set_random_seed(args.seed, deterministic=True)
    config_path = Path(args.config).expanduser().resolve()
    configs.load(str(config_path), recursive=True)
    cfg = Config(recursive_eval(configs), filename=str(config_path))

    # * 参数统计只执行一次，不进入耗时测量；清除预训练初始化以免下载或读取权重。
    parameter_counts = None
    if args.count_params:
        from mmdet3d.models import build_model

        def remove_pretrained(value):
            if isinstance(value, dict):
                if "pretrained" in value:
                    value["pretrained"] = None
                init = value.get("init_cfg")
                if isinstance(init, dict) and init.get("type") == "Pretrained":
                    value["init_cfg"] = None
                elif isinstance(init, list):
                    value["init_cfg"] = [item for item in init
                                         if item.get("type") != "Pretrained"] or None
                for child in value.values():
                    remove_pretrained(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    remove_pretrained(child)

        parameter_cfg = deepcopy(cfg.model)
        remove_pretrained(parameter_cfg)
        # 保持数据加载的随机状态；模型在 CPU 构造、统计后立即释放。
        with torch.random.fork_rng(devices=[]):
            # 避免初始化逐层打印数百行 INFO；警告/错误保留，随后恢复日志级别。
            previous_disable = logging.root.manager.disable
            try:
                logging.disable(max(previous_disable, logging.INFO))
                parameter_model = build_model(parameter_cfg)
            finally:
                logging.disable(previous_disable)
            parameter_counts = count_module_parameters(parameter_model)
            del parameter_model

    # *=================== 使用原生数据集逐帧加载 ===================
    dataset_cfg = cfg.data[args.split].copy()
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset = build_dataset(dataset_cfg)
    if len(dataset) == 0:
        raise ValueError("所选数据集没有样本")
    # * 在构造加载器前确定全部索引，重复样本也会按顺序重新加载。
    sample_indices, sampling_strategy = select_sample_indices(len(dataset), args.max_samples)
    sampled_dataset = Subset(dataset, sample_indices)
    loader = build_dataloader(
        # dataset, samples_per_gpu=1,
        sampled_dataset, samples_per_gpu=1,
        workers_per_gpu=args.workers if args.workers is not None else cfg.data.workers_per_gpu,
        dist=False, shuffle=False,
    )
    # * 原 TensorRT 初始化保留如下；现在基线在两种后端加载前统一采集。
    # vt_cfg = dict(cfg.model.encoders.camera.vtransform)
    # if vt_cfg.pop("type") != "BEVTransformV2":
    #     raise ValueError("当前部署输入准备仅支持 BEVTransformV2 配置")
    # view_transform = BEVTransformV2(**vt_cfg).eval()

    # # *=================== 加载部署引擎一次 ===================
    # model_dir = Path(args.model_dir).expanduser().resolve()
    # required = ["build/camera.backbone.plan", "lidar.backbone.xyz.onnx",
    #             "build/fuser.plan", "build/head.object.plan", "build/head.occ.plan"]
    # for name in required:
    #     if not (model_dir / name).is_file():
    #         raise FileNotFoundError(model_dir / name)
    # # * 在引擎加载前记录基线，后续监控复用该实例；不修改五模块计时边界。
    # memory_monitor = GPUMemoryMonitor(args.device, args.memory_interval)
    # memory_monitor.capture_baseline()
    # # camera.vtransform 参数是旧接口占位项，当前 C++ 实现不加载该文件。
    # core = libpybev.load_bevfusion(
    #     str(model_dir / required[0]), str(model_dir / "build/camera.vtransform.plan"),
    #     str(model_dir / required[1]), str(model_dir / required[2]),
    #     str(model_dir / required[3]), str(model_dir / required[4]), args.precision,
    # )
    # if core is None:
    #     raise RuntimeError("部署引擎加载失败")
    # if not hasattr(core, "last_module_latency"):
    #     raise RuntimeError("libpybev 未更新，请重新编译 pybev 目标")
    # # * 只关闭 C++ 计时日志，不屏蔽错误信息，也不关闭正式推理的计时。
    # if not hasattr(core, "set_timer_verbose"):
    #     raise RuntimeError("libpybev 未更新，请重新编译 pybev 目标")
    # core.set_timer_verbose(False)
    # core.set_timer(False)

    memory_monitor = GPUMemoryMonitor(args.device, args.memory_interval)
    memory_monitor.capture_baseline()
    model_dir = None
    view_transform = None
    if args.backend == "pytorch":
        core = PyTorchBackend(cfg, args.ckpt, args.precision, args.device)
    else:
        vt_cfg = dict(cfg.model.encoders.camera.vtransform)
        if vt_cfg.pop("type") != "BEVTransformV2":
            raise ValueError("当前部署输入准备仅支持 BEVTransformV2 配置")
        view_transform = BEVTransformV2(**vt_cfg).eval()

        # *=================== 加载部署引擎一次 ===================
        model_dir = Path(args.model_dir).expanduser().resolve()
        required = ["build/camera.backbone.plan", "lidar.backbone.xyz.onnx",
                    "build/fuser.plan", "build/head.object.plan", "build/head.occ.plan"]
        for name in required:
            if not (model_dir / name).is_file():
                raise FileNotFoundError(model_dir / name)
        # camera.vtransform 参数是旧接口占位项，当前 C++ 实现不加载该文件。
        core = libpybev.load_bevfusion(
            str(model_dir / required[0]), str(model_dir / "build/camera.vtransform.plan"),
            str(model_dir / required[1]), str(model_dir / required[2]),
            str(model_dir / required[3]), str(model_dir / required[4]), args.precision,
        )
        if core is None:
            raise RuntimeError("部署引擎加载失败")
        if not hasattr(core, "last_module_latency"):
            raise RuntimeError("libpybev 未更新，请重新编译 pybev 目标")
        # * 只关闭 C++ 计时日志，不屏蔽错误信息，也不关闭正式推理的计时。
        if not hasattr(core, "set_timer_verbose"):
            raise RuntimeError("libpybev 未更新，请重新编译 pybev 目标")
        core.set_timer_verbose(False)
        core.set_timer(False)


    csv_path = Path(args.latency_csv).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # 预测立即落盘，不累积在内存中；结果保存位于 C++ 模块计时区间之外。
    prediction_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    # 不指定 --out-dir 时不读取场景表，也不会创建预测目录或保存 PKL。
    sample_scenes = prediction_scene_names(dataset) if prediction_dir is not None else None
    records = []
    # count = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    count = len(sample_indices)  # 正式推理次数，允许大于原数据集大小。
    # fields = ("sample_index", "sample_token", "num_points") + METRICS
    # 同时记录测量序号和原数据集索引，以区分同一样本的重复测量。
    fields = ("measurement_index", "sample_index", "sample_token", "num_points") + METRICS
    # 只有启用预测保存时才增加路径列，便于把每次耗时与预测文件关联。
    if prediction_dir is not None:
        fields += ("prediction_path",)
    # * 每帧运行后立即写 CSV；完整遍历结束再保存均值 JSON，预热不进入任何统计。
    # * 显存监控覆盖预热和正式循环；引擎已常驻，监控值含引擎占用。
    # with csv_path.open("w", newline="", encoding="utf-8") as handle, torch.no_grad():
    # with GPUMemoryMonitor(args.device, args.memory_interval) as memory_monitor, \
    with memory_monitor, \
            csv_path.open("w", newline="", encoding="utf-8") as handle, torch.no_grad(), \
            tqdm(total=count, desc="推理测速", unit="帧", dynamic_ncols=True) as progress:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, data in enumerate(loader):
            # 原先只测前 count 帧；现在加载器本身已按指定次数取样。
            # if index >= count:
            #     break
            # images, refs, points = prepare_inputs(data, view_transform, np)
            # 两种后端使用同一批数据；PyTorch 保留原模型所需标定与元数据。
            num_points = len(data["points"].data[0][0])
            if args.backend == "pytorch":
                images, refs, points = core.prepare(data), None, None
            else:
                images, refs, points = prepare_inputs(data, view_transform, np)
            if index == 0:
                # 预热进度单独显示，不占用正式推理的进度和样本数。
                # for _ in range(args.warmup):
                for _ in tqdm(range(args.warmup), desc="预热", unit="次", leave=False):
                    core.forward(images, refs, points, with_normalization=False)
                core.set_timer(True)
            # * 两种后端均使用五阶段 CUDA Event 计时，不使用 Python 外层墙钟时间。
            # 原来丢弃预测；现在保留当前帧返回值供可选保存。
            # core.forward(images, refs, points, with_normalization=False)
            prediction = core.forward(images, refs, points, with_normalization=False)
            timing = dict(core.last_module_latency())
            # token = dataset.data_infos[index]["token"]
            # row = dict(sample_index=index, sample_token=token, num_points=num_points, **timing)
            # * 通过原索引查 token，避免均匀取样或循环取样后关联错帧。
            sample_index = sample_indices[index]
            token = dataset.data_infos[sample_index]["token"]
            row = dict(measurement_index=index, sample_index=sample_index,
                       sample_token=token, num_points=num_points, **timing)
            # * 当前帧推理结束后立即保存；预热循环不进入此路径。
            if prediction_dir is not None:
                row["prediction_path"] = save_prediction(
                    prediction_dir, prediction, index, sample_index,
                    # dataset.data_infos[sample_index], timing, np)
                    dataset.data_infos[sample_index], timing, np, sample_scenes[token])
            del prediction  # 不让上一帧预测数组一直保留到下一次推理。
            writer.writerow(row)
            handle.flush()
            records.append(timing)
            # 在测量及 CSV 写入完成后更新进度，不加入模块 CUDA Event 计时。
            progress.update(1)
            # 取消逐帧终端输出；CSV 仍即时保存，最后统一打印均值。
            # print(f"[{index + 1}/{count}] {token}: Total={timing['total_ms']:.3f} ms")

    # *=================== 汇总跨帧平均耗时 ===================
    means = {key: sum(row[key] for row in records) / len(records) for key in METRICS}
    summary = dict(
        # 原 model_dir 在 PyTorch 模式不存在，显式记录后端与 checkpoint。
        config=str(config_path), daocc_root=str(daocc_root),
        model_dir=str(model_dir) if model_dir is not None else None,
        backend=args.backend, checkpoint=str(Path(args.ckpt).expanduser().resolve()) if args.backend == "pytorch" else None,
        precision=args.precision,
        split=args.split, measured_samples=len(records), warmup_repeats=args.warmup,
        # measured_samples 保留旧字段；明确区分推理次数和不同样本数。
        measured_inferences=len(records), unique_samples=len(set(sample_indices)),
        dataset_size=len(dataset), requested_inferences=args.max_samples,
        sampling_strategy=sampling_strategy,
        total_inferences=len(records) + args.warmup,
        batch_size=1, seed=args.seed, device=torch.cuda.get_device_name(args.device),
        mean_ms=means, latency_csv=str(csv_path),
        # timing_scope="existing five C++ module stages; total is their sum",
        timing_scope=("existing five C++ module stages; total is their sum" if args.backend == "tensorrt" else
                      "PyTorch CUDA events: lidar extraction including voxelization; camera extraction including reference-point preparation; fuser+decoder; heads including decoding and CPU output; total is stage sum"),
        # 明确本次是否保存预测、保存位置及正式结果数量。
        prediction_dir=str(prediction_dir) if prediction_dir is not None else None,
        # saved_predictions=len(records) if prediction_dir is not None else 0,
        # 最终文件数按不同样本统计，写入次数包含重复帧覆盖。
        saved_predictions=len(set(sample_indices)) if prediction_dir is not None else 0,
        prediction_writes=len(records) if prediction_dir is not None else 0,
    )
    summary_path = csv_path.with_name(csv_path.stem + "_summary.json")
    # MiB = 1024² 字节；这是采样峰值，不是分配器逐次跟踪的精确峰值。
    summary["gpu_memory"] = memory_monitor.result
    # 显存测量流程相同，PyTorch 模式将记录中的“引擎”改为原模型，避免口径误解。
    if args.backend == "pytorch":
        summary["gpu_memory"]["scope"] = "PyTorch model already loaded; warmup and measured inference loop"
        summary["gpu_memory"]["baseline_scope"] = "before PyTorch model loading; existing CUDA context included"
    # 参数量保存到同一汇总文件，不重复写入每帧耗时 CSV。
    if parameter_counts is not None:
        summary["parameter_counts"] = parameter_counts
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # print(f"\nMean latency over {len(records)} samples (ms):")
    # 最终汇总同时显示测量次数和不同样本数，不增加逐帧打印。
    print(f"\nMean latency over {len(records)} inferences "
          f"({len(set(sample_indices))} unique samples, {args.warmup} warmups excluded) (ms):")
    for key, value in means.items():
        print(f"  {key}: {value:.3f}")
    print(f"CSV: {csv_path}\nSummary: {summary_path}")
    if prediction_dir is not None:
        # print(f"逐帧预测: {prediction_dir}（{len(records)} 个 PKL，不含预热）")
        print(f"逐帧预测: {prediction_dir}（本次 {len(set(sample_indices))} 个样本 PKL，重复帧覆盖）")
    memory = memory_monitor.result
    if memory["process_peak_mib"] is not None:
        print(f"进程 GPU 显存采样峰值: {memory['process_peak_mib']:.1f} MiB")
    else:
        print("进程 GPU 显存: 驱动未提供有效数据（例如 WSL 返回 N/A），JSON 记录 null。")
    if memory["device_peak_mib"] is not None:
        print(f"整卡 GPU 显存采样峰值（含其他进程）: {memory['device_peak_mib']:.1f} MiB")
    else:
        print("整卡 GPU 显存采样失败，详见 JSON 的 gpu_memory 字段。")
    # 估计值与真实进程读数用不同标签，避免论文或后续处理混淆口径。
    if memory["device_baseline_mib"] is not None:
        # print(f"引擎加载前整卡显存基线: {memory['device_baseline_mib']:.1f} MiB")
        print(f"模型加载前整卡显存基线: {memory['device_baseline_mib']:.1f} MiB")
    if memory["reported_method"] == "device_delta_estimate":
        print(f"部署显存增量估计（整卡峰值减基线）: {memory['reported_peak_mib']:.1f} MiB")
    elif memory["reported_method"] == "unavailable":
        print("显存指标不可用：基线/峰值查询失败或增量为负，详见 gpu_memory。")
    # 最终按五模块打印原网络参数量，单位 M 为百万个参数。
    if parameter_counts is not None:
        print("\nParameters before PTQ/BN fusion (M):")
        for name, values in parameter_counts["modules"].items():
            print(f"  {name}: {values['parameters_m']:.3f}")
        print(f"  total: {parameter_counts['total']['parameters_m']:.3f}")
        if parameter_counts["unassigned"]["parameters"]:
            print("  注意：存在未归入五模块的参数，详见 JSON 的 unassigned 字段。")


if __name__ == "__main__":
    main()
