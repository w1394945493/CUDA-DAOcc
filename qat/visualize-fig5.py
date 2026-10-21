"""生成点云、真值、occupancy 预测、有 ADH 模型共享 BEV 特征的 2×2 图。

需提供训练好的原始 PyTorch checkpoint（不是 PTQ 模型对象或 TensorRT plan）。
论文未公开特征聚合/归一化细节，本脚本采用通道绝对值均值和逐帧百分位色标。
"""

import argparse
import ast
import json
import logging
from copy import deepcopy
from pathlib import Path
import sys


def load_occ_colors(path):
    # * 直接读取指定 vis_occ3d.py 中 colors 的原始列表，不另定义类别配色。
    # 只解析赋值表达式，不导入整个脚本，避免执行其 Qt 环境变量设置。
    import numpy as np
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "colors" for t in node.targets):
            # 当前源文件格式：colors = np.array([...]).astype(np.uint8)。
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "astype":
                value = value.func.value
            if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "array"):
                break
            palette = np.asarray(ast.literal_eval(value.args[0]), dtype=np.uint8)
            if palette.ndim != 2 or palette.shape[0] < 17 or palette.shape[1] < 3:
                break
            return palette[:, :3]
    raise ValueError(f"无法读取 {path} 的 colors 配色表；请核对源文件格式")


def occupancy_rgb(semantics, palette, mask=None):
    import numpy as np
    labels = top_labels(semantics, mask).T
    # * 空体素单独留白：vis_occ3d.py 的第 17 项是 ego car 装饰色，不能当成 free 类颜色。
    rgb = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    visible = (labels >= 0) & (labels < 17)
    rgb[visible] = palette[labels[visible].astype(int)]
    return rgb


def select_indices(total, requested):
    # * 在时间排序的数据集索引上均匀取样；超过总数时取全部，不循环、不重复。
    if total < 1 or requested < 1:
        raise ValueError("数据集样本数和 --num-samples 都必须大于 0")
    count = min(total, requested)
    if count == 1:
        return [total // 2]
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def clear_pretrained(value):
    # checkpoint 已包含权重，不另外读取 backbone 的预训练文件。
    if isinstance(value, dict):
        if "pretrained" in value:
            value["pretrained"] = None
        init = value.get("init_cfg")
        if isinstance(init, dict) and init.get("type") == "Pretrained":
            value["init_cfg"] = None
        for child in value.values():
            clear_pretrained(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            clear_pretrained(child)


def top_labels(semantics, camera_mask=None):
    import numpy as np
    # * Occ3D 数组顺序为 [X, Y, Z]；每个 XY 柱取最高的有效占据体素。
    # 17 为自由空间，255 为忽略；类别 0 仍是有效占据类别。
    valid = (semantics >= 0) & (semantics < 17)
    if camera_mask is not None:
        valid &= camera_mask.astype(bool)
    heights = np.where(valid, np.arange(semantics.shape[2]), -1).max(axis=2)
    labels = np.take_along_axis(semantics, np.maximum(heights, 0)[..., None], axis=2)[..., 0]
    return np.where(heights >= 0, labels, 17)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True, help="训练时使用 ADH 的原始 PyTorch 权重")
    parser.add_argument("--daocc-root", default=str(Path(__file__).resolve().parents[1] / "DAOcc"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=10, help="均匀抽取 N 帧；超过总数则全部可视化")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    # 原参数只作用于真值；现在预测与真值使用同一掩码，方便公平对照。
    # parser.add_argument("--camera-mask", action="store_true", help="仅在真值图中保留相机掩码内体素")
    parser.add_argument("--camera-mask", action="store_true", help="真值和预测均仅显示相机掩码内体素")
    args = parser.parse_args()
    if args.num_samples < 1 or args.workers < 0:
        parser.error("num-samples 必须为正数，workers 不能为负数")

    # 无窗口保存图片，避免 WSL 中 Qt/xcb 插件冲突。
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(args.daocc_root).expanduser().resolve()))
    from mmcv import Config
    from mmcv.parallel import scatter
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from torchpack.utils.config import configs
    from torch.utils.data import Subset
    from tqdm import tqdm
    from mmdet.apis import set_random_seed
    from mmdet3d.utils import recursive_eval
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model

    print("加载数据配置……", flush=True)
    torch.cuda.set_device(args.device)
    set_random_seed(0, deterministic=True)
    configs.load(str(Path(args.config).expanduser().resolve()), recursive=True)
    cfg = Config(recursive_eval(configs))
    data_cfg = deepcopy(cfg.data.val)
    data_cfg.test_mode = True
    data_cfg.pop("samples_per_gpu", None)
    dataset = build_dataset(data_cfg)
    indices = select_indices(len(dataset), args.num_samples)
    loader = build_dataloader(Subset(dataset, indices), samples_per_gpu=1,
                              workers_per_gpu=args.workers, dist=False, shuffle=False)

    # 同一场景的图片放在一起，场景名来自 nuScenes 数据表。
    table_dir = Path(dataset.dataset_root) / dataset.version
    # scenes = {s["token"]: s["name"] for s in json.loads((table_dir / "scene.json").read_text())}
    # sample_scenes = {s["token"]: scenes[s["scene_token"]]
    #                  for s in json.loads((table_dir / "sample.json").read_text())}
    scene_records = json.loads((table_dir / "scene.json").read_text())
    samples = {s["token"]: s for s in json.loads((table_dir / "sample.json").read_text())}
    sample_scenes = {}
    scene_frame_numbers = {}
    # * 沿场景的 sample 链表从 1 编号，保留原始关键帧顺序，不因均匀抽样重新编号。
    # 这里统计 sample 关键帧，不包含输入点云聚合使用的历史 sweeps。
    for scene_record in scene_records:
        sample_token = scene_record["first_sample_token"]
        frame_number = 1
        while sample_token:
            sample = samples[sample_token]
            if sample_token in scene_frame_numbers or sample["scene_token"] != scene_record["token"]:
                raise ValueError(f"场景 sample 链表重复或归属错误：{sample_token}")
            sample_scenes[sample_token] = scene_record["name"]
            scene_frame_numbers[sample_token] = frame_number
            sample_token = sample["next"]
            frame_number += 1

    model_cfg = deepcopy(cfg.model)
    clear_pretrained(model_cfg)
    print("构建模型并加载权重……", flush=True)
    # 屏蔽逐层初始化的 INFO 日志，保留警告和错误，让进度条易于阅读。
    previous_disable = logging.root.manager.disable
    try:
        logging.disable(max(previous_disable, logging.INFO))
        model = build_model(model_cfg)
    finally:
        logging.disable(previous_disable)
    load_checkpoint(model, str(Path(args.ckpt).expanduser()), map_location="cpu", strict=True)
    if args.precision == "fp16":
        wrap_fp16_model(model)
    print("将模型移至 GPU……", flush=True)
    model.cuda().eval()

    # * 复制模型原有坐标采样器，只保留几何采样；去掉学习得到的通道转换。
    # 这样展示的仍是共享 Decoder 特征，不是 occupancy 分支再次卷积后的特征。
    transform_cfg = cfg.model.heads.occ.coordinate_transform
    if (transform_cfg.type != "CrossCoordinateSample" or
            list(transform_cfg.point_num) != [200, 200, 1] or
            list(transform_cfg.point_range) != [-40., 40., -40., 40., 0., 0.]):
        raise ValueError("本脚本仅支持 Occ3D 的自车系 ±40m、200×200 网格配置")
    sampler = deepcopy(model.heads["occ"].coordinate_transform)
    if sampler.point_type != "ego":
        raise ValueError("真值对齐要求 CrossCoordinateSample.point_type 为 ego")
    sampler.transfer_conv = None
    sampler.extra_up = False
    sampler.float().eval()
    captured = {}

    def capture_shared(module, inputs, output):
        # hook 不改变网络输出；只取 Decoder Neck 的共享特征，供前向完成后绘图。
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise ValueError("预期 Decoder Neck 只输出一个共享 BEV 特征")
            output = output[0]
        captured["feature"] = output.detach()

    handle = model.decoder["neck"].register_forward_hook(capture_shared)
    # 原独立配色停用，保留供核对。
    # colors = np.array([
    #     [0, 0, 0], [112, 128, 144], [220, 20, 60], [255, 69, 0],
    #     [255, 158, 0], [233, 150, 70], [255, 61, 99], [0, 0, 230],
    #     [47, 79, 79], [255, 140, 0], [255, 99, 71], [0, 207, 191],
    #     [175, 0, 75], [75, 0, 75], [112, 180, 60], [222, 184, 135],
    #     [0, 175, 0], [255, 255, 255]], dtype=np.uint8)
    palette_path = Path(__file__).resolve().parents[1] / "DAOcc/tools/vis_occ3d.py"
    colors = load_occ_colors(palette_path)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = dict(config=str(Path(args.config).resolve()), checkpoint=str(Path(args.ckpt).expanduser().resolve()),
                  requested_samples=args.num_samples, total_samples=len(dataset), selected_indices=indices,
                  feature_layer="decoder.neck", feature_reduction="mean(abs(channels)), after geometric resampling",
                  normalization="per-frame 1st/99th percentile, clipped; not comparable absolute colors across frames",
                  coordinates="ego: horizontal +X, vertical +Y; XY [-40,40] metres",
                  gt_projection="highest occupied voxel per XY column; free/ignore omitted",
                  points="dataset input points, including configured history sweeps; ego Z [-1,5.4]",
                  camera_mask_gt=args.camera_mask, camera_mask_prediction=args.camera_mask, precision=args.precision,
                  occupancy_palette=str(palette_path), layout="2x2: point cloud, GT, BEV feature, prediction",
                  adh="assumed from supplied checkpoint training; not inferred or verified from weights",
                  paper_match="qualitative reproduction; original feature reduction and sample unspecified", frames=[])

    def first_tensor(value):
        # batch size 固定为 1；兼容 scatter 后的列表和带 batch 维的 Tensor。
        while isinstance(value, (list, tuple)):
            value = value[0]
        return value

    try:
        with torch.no_grad():
            for index, batch in zip(indices, tqdm(loader, total=len(indices), desc="图5可视化", unit="帧")):
                data = scatter(batch, [args.device])[0]
                captured.clear()
                # 原先忽略预测返回值；现在复用同一次推理的 occ_pred，不额外运行网络。
                # model(**data)
                outputs = model(**data)
                feature = captured.pop("feature").float()
                # 与 OccHead.forward 一致：共享特征 [B,C,X,Y] 先转成 [B,C,Y,X]。
                aligned = sampler(feature.transpose(-1, -2).contiguous(),
                                  data["lidar_aug_matrix"].float(), data["lidar2ego"].float(),
                                  data["occ_aug_matrix"].float())
                heat = aligned[0].abs().mean(0).cpu().numpy()
                if not np.isfinite(heat).all():
                    raise ValueError("BEV 特征包含 NaN/Inf")
                low, high = (float(v) for v in np.percentile(heat, [1, 99]))
                normalized = np.clip((heat - low) / max(high - low, 1e-12), 0, 1)

                sem = first_tensor(data["voxel_semantics"])
                if sem.ndim == 4:
                    sem = sem[0]
                sem = sem.cpu().numpy()
                if sem.shape != (200, 200, 16):
                    raise ValueError(f"非 Occ3D 真值形状：{sem.shape}")
                mask = None
                if args.camera_mask:
                    mask = first_tensor(data["mask_camera"])
                    if mask.ndim == 4:
                        mask = mask[0]
                    mask = mask.cpu().numpy()
                # rgb = colors[top_labels(sem, mask).T.astype(int)]
                rgb = occupancy_rgb(sem, colors, mask)
                prediction = outputs[0]["occ_pred"]
                if torch.is_tensor(prediction):
                    prediction = prediction.detach().cpu().numpy()
                prediction = np.asarray(prediction)
                if prediction.shape != sem.shape:
                    raise ValueError(f"预测与真值形状不一致：{prediction.shape} / {sem.shape}")
                pred_rgb = occupancy_rgb(prediction, colors, mask)

                # 点云先撤销 LiDAR 增强，再转到自车系，并应用与真值相同的 occ 增强。
                xyz = first_tensor(data["points"])[..., :3].float()
                aug = data["lidar_aug_matrix"][0].float()
                ego = data["lidar2ego"][0].float()
                xyz = (xyz - aug[:3, 3]) @ torch.inverse(aug[:3, :3]).T
                xyz = xyz @ ego[:3, :3].T + ego[:3, 3]
                xyz = xyz @ data["occ_aug_matrix"][0, :3, :3].float().T
                xyz = xyz.cpu().numpy()
                keep = ((xyz[:, :2] >= -40) & (xyz[:, :2] < 40)).all(1) & (xyz[:, 2] >= -1) & (xyz[:, 2] < 5.4)
                xyz = xyz[keep]
                token = dataset.data_infos[index]["token"]
                scene = sample_scenes[token]
                # path = out_dir / scene / f"{token}.png"
                # * 文件名前缀为场景内关键帧序号（从 1 开始），补零便于按名称排序。
                scene_frame_number = scene_frame_numbers[token]
                path = out_dir / scene / f"frame_{scene_frame_number:03d}_{token}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                # fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
                # * 2×2：左上点云，左下 BEV 热力图；右上真值、右下预测，同列对齐便于比较。
                fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
                axes = axes.ravel()
                axes[0].scatter(xyz[:, 0], xyz[:, 1], s=0.15, c="gray", linewidths=0, rasterized=True)
                axes[1].imshow(rgb, origin="lower", extent=(-40, 40, -40, 40), interpolation="nearest")
                # axes[2].imshow(pred_rgb, origin="lower", extent=(-40, 40, -40, 40), interpolation="nearest")
                axes[3].imshow(pred_rgb, origin="lower", extent=(-40, 40, -40, 40), interpolation="nearest")
                # image = axes[2].imshow(normalized, ...)
                # 热力图位于左下；采用论文式排版，不显示色条。
                image = axes[2].imshow(normalized, origin="lower", extent=(-40, 40, -40, 40),
                                       cmap="jet", vmin=0, vmax=1, interpolation="nearest")
                # 新增的预测图不冒用论文 (c)（原论文 (c) 为无 ADH 特征）。
                # for ax, title in zip(axes, ["(a) Point Cloud", "(b) Ground Truth", "(d) w/ ADH - BEV Feature"]):
                # for ax, title in zip(axes, ["(a) Point Cloud", "(b) Ground Truth", "Occupancy Prediction", "(d) w/ ADH - BEV Feature"]):
                # for ax, title in zip(axes, ["(a) Point Cloud", "(b) Ground Truth", "(d) w/ ADH - BEV Feature", "Occupancy Prediction"]):
                for ax, title in zip(axes, ["(a) Point Cloud", "(b) Ground Truth", "(d) w/ ADH", "Occupancy Prediction"]):
                    # ax.set(xlim=(-40, 40), ylim=(-40, 40), xlabel="Ego X (m)", ylabel="Ego Y (m)", title=title)
                    ax.set(xlim=(-40, 40), ylim=(-40, 40))
                    ax.set_aspect("equal")
                    # 显式关闭主/次网格，防止外部 Matplotlib 样式自动打开网格线。
                    ax.grid(False, which="both")
                    # * 隐藏坐标轴、边框、刻度线和刻度数字，只在子图下方保留图注。
                    ax.set_axis_off()
                    ax.text(0.5, -0.025, title, transform=ax.transAxes,
                            ha="center", va="top", fontsize=18, fontfamily="serif", clip_on=False)
                # fig.colorbar(image, ax=axes[2], shrink=0.75, label="Normalized mean |feature|")
                # colorbar = fig.colorbar(image, ax=axes[3], shrink=0.75, label="Normalized mean |feature|")
                # colorbar = fig.colorbar(image, ax=axes[2], shrink=0.75, label="Normalized mean |feature|")
                # colorbar.ax.grid(False, which="both")
                # 场景/token 和色标范围仍记录在 JSON，不写入图片。
                # fig.suptitle(f"{scene} / {token}", fontsize=10)
                # fig.savefig(path, dpi=180)
                fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.04, facecolor="white")
                plt.close(fig)
                report["frames"].append(dict(index=index, token=token, scene=scene, path=str(path.resolve()),
                                             # * 场景内从 1 开始的关键帧序号；index 仍为数据集索引。
                                             scene_frame_number=scene_frame_number,
                                             feature_shape=list(feature.shape), heat_p01=low, heat_p99=high,
                                             visible_points=len(xyz)))
                # 每帧完成后更新说明，长任务中断时仍保留已完成的结果索引。
                (out_dir / "fig5_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                del data, feature, aligned, batch, outputs, prediction
    finally:
        handle.remove()
    # print(f"已保存 {len(report['frames'])} 张三联图：{out_dir.resolve()}")
    print(f"已保存 {len(report['frames'])} 张 2×2 图：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
