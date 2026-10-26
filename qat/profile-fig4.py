"""复现实验设置：图 4 两种图像分辨率、两种 LiDAR 范围的编码器计算量。

统计卷积/线性层 MACs，FLOPs = 2 * MACs；不是 TensorRT 耗时测试。
不计 BN、激活、插值、残差加法、体素化和稀疏索引构造等操作。
论文未公开完整计数口径，不能把本脚本结果宣称为论文数值的精确复现。
"""

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="原始 DAOcc YAML 配置")
    parser.add_argument("--daocc-root", default=str(Path(__file__).resolve().parents[1] / "DAOcc"),
                        help="已编译 CUDA 扩展的 DAOcc 仓库")
    parser.add_argument("--out-dir", default="results/fig4", help="JSON、CSV 和 PNG/PDF 图表保存目录")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sample-token", help="指定点云；默认选验证集中原始关键帧点数最多者")
    parser.add_argument("--total-sweeps", type=int, default=10,
                        help="包含当前帧的总帧数；默认当前帧 + 9 个历史 sweep")
    parser.add_argument("--camera-count", type=int, default=6)
    #* 仅绘图时不加载模型、数据集或 CUDA，直接复用之前的测量报告。
    parser.add_argument("--plot-only", metavar="JSON", help="读取 fig4_summary.json，仅生成图表")
    parser.add_argument("--plot-camera-scope", choices=("single", "all"), default="single",
                        help="相机图统计单路或全部相机；默认单路，与 CSV 的 gflops 一致")
    args = parser.parse_args()
    if not args.plot_only and not args.config:
        parser.error("测量模式需要 --config；已有结果可使用 --plot-only JSON")
    if args.total_sweeps < 1 or args.camera_count < 1:
        parser.error("total-sweeps 和 camera-count 必须为正数")
    return args


class ConvCounter:
    # *=================== 按实际执行记录卷积计算量 ===================
    # * 整个类分三个阶段工作：
    # ① counter = ConvCounter(model, sparse_type)：找到各层，给它们登记统计函数；此时不运行网络。
    # ② model(input)：网络正常运行；每层运行结束后，PyTorch 自动调用已登记的统计函数。
    #    统计函数根据这一层的形状/通道数计算 MACs，将结果追加到 counter.rows。
    # ③ counter.close()：取消登记。以后再运行网络，就不会调用本计数器的统计函数。
    # 这里没有逐次监听 GPU 的加法/乘法指令，而是根据实际前向得到的信息，用公式计算运算次数。
    # MAC 表示一次乘加；本脚本后续将 1 MAC 换算为 2 FLOPs。这里不计偏置加法等操作。

    # *=================== 第一阶段：给模型中的层登记回调函数 ===================
    # 创建 ConvCounter(...) 对象时，Python 自动执行 __init__。
    # self 是当前计数器对象；self.rows 等属性属于这个计数器。
    # model 是要统计的网络对象，例如 camera；sparse_type 是识别稀疏卷积用的类。
    def __init__(self, model, sparse_type):
        import torch
        # 保存逐层统计结果。开始是空列表，只有网络真正前向后才会被填入数据。
        self.rows = []
        # 保存每次注册返回的 handle（登记凭据），以后用它取消对应的回调。
        # handles 保存的是登记凭据，不是模型权重，也不是计算量。
        self.handles = []
        # named_modules() 遍历整个模型及其子模块。
        # name 是层的名称字符串，如 "0.layer1.0.conv1"；module 是该层的实际对象。
        for name, module in model.named_modules():
            # isinstance 判断该对象属于哪种层：这里只登记 Conv2d、Linear、稀疏卷积。
            # BN、ReLU 等不满足条件，会跳过，因此不会出现在本计数器的计算量中。
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, sparse_type)):
                # * 下面这行从内到外执行，等价于以下三步（仅用于解释）：
                # callback = self.make_hook(name, sparse_type)  # 创建这个层专用的统计函数，但不执行它。
                # handle = module.register_forward_hook(callback)  # 把函数登记到该层，交给 PyTorch 管理。
                # self.handles.append(handle)  # 保存登记凭据，供 close() 移除。
                # register_forward_hook 是 PyTorch 提供的方法，不需要修改原来那层的 forward。
                # 注意 callback 后面没有 ()：登记的是函数本身，不是在这里调用函数。
                self.handles.append(module.register_forward_hook(self.make_hook(name, sparse_type)))

    # *=================== 创建回调：返回一个函数，等待前向结束后调用 ===================
    # make_hook 每调用一次，就创建一个新的 hook 函数，专门记住传进来的 name 和 sparse_type。
    # 例如，为 conv1 创建的函数记住 conv1，为 conv2 创建的函数记住 conv2，不会把层名混在一起。
    # 这种内部函数能记住外部变量的机制叫“闭包”；这里也记住 self，便于向同一个 rows 写入结果。
    def make_hook(self, name, sparse_type):
        # * 此处 def 只是定义函数；里面的统计代码此时尚未执行。
        # 只有相应层通过 module(...) 完成前向时，PyTorch 才自动执行 hook(module, inputs, output)。
        # 三个参数由 PyTorch 自动传入，不需要调用者手动填写：
        # module：刚执行完的层；inputs：该层位置参数组成的元组；output：该层前向返回的结果。
        # 例如 conv(x)，inputs 通常是 (x,)，所以 inputs[0] 才是输入 x。
        def hook(module, inputs, output):
            import math
            # 以下代码在该层每次执行完后运行；同一层被调用两次，就记录两次计算量。
            if isinstance(module, sparse_type):
                # *=================== 稀疏卷积的计算公式 ===================
                # 稀疏张量只存活跃体素：features 的形状为 [活跃体素数, 通道数]。
                # pairs 表示有效输入—输出连接数量，包含不同卷积核位置的连接。
                # 一个输入体素可能参与多个输出的计算，所以 pairs 不等于体素数量。
                if module.conv1x1:
                    # 本仓库 1×1 稀疏卷积直接对每个活跃体素做通道变换。
                    # 每个体素对应一次输入—输出连接，因此连接数就是输入 features 的行数。
                    pairs = inputs[0].features.shape[0]
                else:
                    # indice_dict 是 spconv 在前向中生成/复用的稀疏连接缓存。
                    # module.indice_key 定位当前层使用的缓存；[3] 取出各卷积核位置的有效连接数。
                    # 例如 counts=[10, 20, 10]，表示三个核位置分别参与 10、20、10 次连接。
                    # long() 转成 64 位整数，供后续求和使用。
                    counts = output.indice_dict[module.indice_key][3].long()
                    # 求和得到全部有效连接数；item() 将单元素张量取成 Python 数值。
                    pairs = int(counts.sum().item())
                    if module.subm:
                        # subm 表示子流形稀疏卷积，输出仍位于输入已有的活跃体素位置。
                        # * 这里专门匹配本仓库旧版 spconv 内核的执行方式，不是通用版本的假设。
                        # 内核将 counts 最大的位置单独处理，直接对所有输入 features 做矩阵乘法。
                        # 因此从刚才的总连接数中减去该位置的缓存计数，再加上实际输入体素数。
                        # 即 pairs = 其余核位置连接数 + 输入体素数；两者相等时这项修正为 0。
                        pairs += inputs[0].features.shape[0] - int(counts.max().item())
                # 每条空间连接都需要从 Cin 个输入通道计算 Cout 个输出通道，产生 Cin×Cout 次乘加。
                # 所以 MACs = 有效空间连接数 × 输入通道数 × 输出通道数。
                macs = pairs * module.in_channels * module.out_channels
                # 记录稀疏输出的特征矩阵形状，而不是整块包含大量空体素的三维网格形状。
                shape = list(output.features.shape)
            else:
                # *=================== 普通二维卷积/线性层的计算公式 ===================
                # 普通层不使用稀疏连接计数，记为 None，保存到 JSON 时对应 null。
                pairs = None
                # 在本计数器筛选出的普通层中，Conv2d 有 kernel_size 属性，Linear 没有。
                if hasattr(module, "kernel_size"):
                    # output.numel() 是所有输出元素数量：batch × 输出通道 × 输出高 × 输出宽。
                    # 每个输出元素需遍历 Cin/groups 个输入通道，以及 Kh×Kw 个卷积核位置。
                    # // 是整数除法；groups 是分组数；math.prod((Kh, Kw)) 就是 Kh×Kw。
                    # * MACs = 输出元素数量 × 每组输入通道数 × 卷积核面积。
                    # 例如输出形状 [1, 8, 10, 10]，Cin=3、groups=1、核为 3×3：
                    # MACs = (1×8×10×10) × 3 × 9 = 21600。
                    macs = output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
                else:
                    # 线性层：每个输出元素都是 in_features 个输入值的加权求和。
                    # 所以每个输出需要 in_features 次乘加，再乘全部输出元素数量。
                    macs = output.numel() * module.in_features
                # 普通层直接记录输出 Tensor 的形状，如 [1, 256, 32, 88]。
                shape = list(output.shape)
            # 将本次层调用的统计结果追加成一条字典记录：
            # layer：层名；type：层类型名称；output_shape：输出形状；
            # active_pairs：稀疏连接数（普通层为空）；macs：这一层本次调用的乘加次数。
            self.rows.append(dict(layer=name, type=type(module).__name__,
                                  output_shape=shape, active_pairs=pairs, macs=int(macs)))
            # hook 没有 return 输出值，因此隐式返回 None，表示不替换该层原本的输出。
            # 它只记录统计信息，不改变模型的预测特征。
        # * 返回刚定义的函数对象 hook；不是 hook(...)，所以这里不会执行统计。
        # 返回值会被 __init__ 中的 register_forward_hook 登记到对应层。
        return hook

    # *=================== 最后阶段：移除统计函数 ===================
    def close(self):
        # 遍历创建计数器时保存的登记凭据。
        for handle in self.handles:
            # 取消这一条 hook 登记；不会删除网络层，也不会清空已经保存的 rows。
            handle.remove()


#* =================== 将实测计算量绘制为论文图4风格 ===================
def plot_results(report, out_dir, camera_scope="single"):
    import math
    import matplotlib
    # 无窗口后端，适用于 WSL/服务器；避免 Qt xcb 插件冲突。
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["results"]
    # 按设置查找而不是依赖 JSON 行顺序，确保横轴与数值一一对应。
    def find(module, setting):
        matches = [r for r in rows if r["module"] == module and r["setting"] == setting]
        if len(matches) != 1:
            raise ValueError(f"需要唯一的测量记录：{module} / {setting}")
        return matches[0]

    camera = [find("camera_backbone_neck", x) for x in ("256x704", "512x1408")]
    lidar = [find("lidar_sparse_encoder", f"xy=+/-{r}m,z=[-5,3]m") for r in (40, 80)]
    key = "gflops" if camera_scope == "single" else "all_camera_gflops"
    values = [[float(r[key]) for r in camera], [float(r["gflops"]) for r in lidar]]
    if any(not math.isfinite(v) or v <= 0 for pair in values for v in pair):
        raise ValueError("绘图需要有限且为正的计算量")
    if camera_scope == "all" and camera[0]["cameras"] != camera[1]["cameras"]:
        raise ValueError("两种分辨率的相机数量不一致")

    #* 两侧统一使用真实 GFLOPs，不把 LiDAR 的 GFLOPs 误标成论文中的 MFLOPs。
    # 共享从零开始的纵轴，柱高可直接比较；图注注明这是本脚本的统计口径。
    scope = "single camera" if camera_scope == "single" else f"{camera[0]['cameras']} cameras"
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 12,
                         "axes.titlesize": 17, "axes.labelsize": 13, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        ymax = max(v for pair in values for v in pair) * 1.20
        for ax, pair, color, title, labels, xlabel in zip(
                axes, values, ("#87cee5", "#f4a45b"),
                ("Camera Encoder", "LiDAR Encoder"),
                (("256×704", "512×1408"), ("[-40, 40]", "[-80, 80]")),
                ("Image Resolution (H×W)", "XY Range (m)")):
            bars = ax.bar([0, 1], pair, width=0.55, color=color, edgecolor="#444444", linewidth=0.8)
            ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
            ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("FLOPs (G)")
            ax.set_ylim(0, ymax)
            ax.set_axisbelow(True); ax.grid(axis="y", linestyle="--", alpha=0.35)
            for bar, value in zip(bars, pair):
                ax.text(bar.get_x() + bar.get_width()/2, value + ymax*0.015,
                        f"{value:.3f}", ha="center", va="bottom")
        fig.text(0.5, 0.035,
                 f"Measured conv/linear FLOPs = 2 × MACs; camera: backbone + neck ({scope}).\n"
                 "LiDAR: same aggregated point cloud, z in [-5, 3] m. Paper values are not substituted.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        # 保存高分辨率 PNG 与矢量 PDF，便于查看和论文排版。
        try:
            for suffix in ("png", "pdf"):
                path = out_dir / f"fig4_measured_{camera_scope}.{suffix}"
                fig.savefig(str(path), dpi=300, bbox_inches="tight")
                print(f"图表已保存：{path.resolve()}")
        finally:
            plt.close(fig)


def main():
    args = parse_args()
    #* 读取现有结果可独立画图，不触发 CUDA 扩展导入和模型前向。
    if args.plot_only:
        report = json.loads(Path(args.plot_only).expanduser().read_text())
        plot_results(report, args.out_dir, args.plot_camera_scope)
        return
    print("加载 DAOcc 依赖与配置……", flush=True)
    sys.path.insert(0, str(Path(args.daocc_root).expanduser().resolve()))
    import numpy as np
    import torch
    import mmcv
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval
    from mmdet3d.models.builder import build_backbone, build_neck
    from mmdet3d.ops import Voxelization
    from mmdet3d.ops.spconv.conv import SparseConvolution
    from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile, LoadPointsFromMultiSweeps

    torch.cuda.set_device(args.device)
    torch.manual_seed(0)
    config_path = Path(args.config).expanduser().resolve()
    configs.load(str(config_path), recursive=True)
    cfg = Config(recursive_eval(configs), filename=str(config_path))
    encoders = cfg.model.encoders
    results = []
    print("开始统计相机 Backbone + Neck……", flush=True)

    def record(name, setting, counter, **metadata):
        macs = sum(row["macs"] for row in counter.rows)
        result = dict(module=name, setting=setting, macs=macs, gmacs=macs / 1e9,
                      flops=2 * macs, gflops=2 * macs / 1e9,
                      layers=counter.rows, **metadata)
        results.append(result)
        print(f"{name} {setting}: {macs / 1e9:.6f} GMACs / {2 * macs / 1e9:.6f} GFLOPs", flush=True)

    # *=================== 相机：只运行 Backbone + Neck ===================
    # 权重数值不影响这些层的运算次数，无需加载 checkpoint 或预训练文件。
    camera_cfg = deepcopy(encoders.camera.backbone)
    camera_cfg["init_cfg"] = None
    camera_cfg.pop("pretrained", None)
    camera = torch.nn.Sequential(build_backbone(camera_cfg), build_neck(encoders.camera.neck)).cuda().eval()

    # 仅做前向统计，不计算梯度，也不保留反向传播所需的中间状态。
    with torch.no_grad():
        # 分别测量图 4 的两种输入分辨率；每次循环使用独立的计数器。
        for height, width in ((256, 704), (512, 1408)):
            # * 给卷积/线性层注册 forward hook：每层执行完后，自动计算并记录该层 MACs。
            # counter.rows 是逐层记录列表；每条记录含层名 layer 和乘加次数 macs。
            # SparseConvolution 用于识别稀疏卷积类型；这里不会因此向相机网络添加稀疏卷积。
            # 安装一个“计算量计数器”: 找到其中的卷积层，线性层，挂熵回调函数。
            #当某层运行结束时，回调函数根据该层的输入输出形状和通道数，计算该层的乘加次数，并保存
            counter = ConvCounter(camera, SparseConvolution)
            # 配合 finally，保证正常完成或发生异常时都能移除 hook。
            try:
                # * 构造一张 GPU 上的全零 RGB 图像：[batch=1, 通道=3, 高, 宽]。
                # camera 是 Sequential(Backbone, Neck)，调用它会依次执行这两部分并触发 hook。
                # 这里统计普通稠密卷积，计算量取决于形状，不会因输入全零而减少；无需真实图像。
                # 不接收预测特征，因为所需的计算量已经由 hook 写入 counter.rows。
                camera(torch.zeros(1, 3, height, width, device="cuda"))
                # Sequential 中第 0 个模块是 Backbone，因此其子层名称以 "0." 开头。
                # 遍历逐层记录，只累加这些层的 macs，得到单路 Backbone 的乘加次数。
                backbone_macs = sum(r["macs"] for r in counter.rows if r["layer"].startswith("0."))
                # 第 1 个模块是 Neck；同理，累加以 "1." 开头的层，得到单路 Neck 的乘加次数。
                neck_macs = sum(r["macs"] for r in counter.rows if r["layer"].startswith("1."))
                # * 汇总当前分辨率：record 保存单路总量和逐层明细，并附加下面这些统计字段。
                # 此时结果先存入内存中的 results，脚本末尾才统一写入 JSON/CSV。

                record("camera_backbone_neck", f"{height}x{width}", counter,
                       # 相机数量，默认 6；下面以单路计算量乘数量，不实际重复运行六路。
                       cameras=args.camera_count,
                       # 单路 Backbone、Neck 的 GMACs：除以 10^9，将乘加次数换算为十亿次。
                       backbone_gmacs=backbone_macs / 1e9, neck_gmacs=neck_macs / 1e9,
                       # 全部相机的 Backbone + Neck 总 GMACs，假设各路分辨率和网络相同。
                       all_camera_gmacs=args.camera_count * (backbone_macs + neck_macs) / 1e9,
                       # 全部相机仅 Backbone 的 GMACs，便于比较不包含 Neck 的统计口径。
                       all_camera_backbone_gmacs=args.camera_count * backbone_macs / 1e9,
                       # 全部相机的总 GFLOPs：一次乘加按一次乘法 + 一次加法计，所以再乘 2。
                       all_camera_gflops=2 * args.camera_count * sum(r["macs"] for r in counter.rows) / 1e9)
            finally:
                # 移除本轮注册的 hook，避免下一种分辨率前向时仍触发旧计数器。
                counter.close()
    del camera
    torch.cuda.empty_cache()

    # *=================== 选验证关键帧并聚合历史点云 ===================
    # 图 4 的点云结果依赖具体样本；mini 验证集不能替代论文完整验证集。
    ann_path = Path(cfg.data.val.ann_file).expanduser()
    infos = mmcv.load(str(ann_path))["infos"]
    data_root = Path(cfg.data.val.get("dataset_root", cfg.get("dataset_root", ".")))

    def resolve_data(path):
        path = Path(path)
        if path.is_file():
            return str(path.resolve())
        candidate = data_root / path
        if candidate.is_file():
            return str(candidate.resolve())
        raise FileNotFoundError(f"找不到点云 {path}；检查 info 内路径及 dataset_root")

    def point_count(info):
        path = resolve_data(info["lidar_path"])
        if path.endswith(".npy"):
            return int(np.load(path, mmap_mode="r").shape[0])
        if Path(path).stat().st_size % 20:
            raise ValueError(f"{path} 不是每点 5 个 float32 的 nuScenes 点云")
        return Path(path).stat().st_size // 20

    if args.sample_token:
        selected = next((item for item in infos if item["token"] == args.sample_token), None)
        if selected is None:
            raise ValueError("sample-token 不在当前验证集 info 中")
    else:
        selected = max(infos, key=point_count)
    data = deepcopy(selected)
    data["lidar_path"] = resolve_data(data["lidar_path"])
    data["sweeps"] = data["sweeps"][:args.total_sweeps - 1]
    for sweep in data["sweeps"]:
        sweep["data_path"] = resolve_data(sweep["data_path"])
    data = LoadPointsFromFile(coord_type="LIDAR", load_dim=5, use_dim=5)(data)
    data = LoadPointsFromMultiSweeps(sweeps_num=args.total_sweeps - 1, load_dim=5,
                                    use_dim=[0, 1, 2, 3, 4], test_mode=True,
                                    remove_close=True, pad_empty_sweeps=False)(data)
    points = data["points"].tensor.cuda()
    actual_sweeps = 1 + len(data["sweeps"])
    print(f"选择 {selected['token']}，聚合 {actual_sweeps} 帧，{len(points)} 个点", flush=True)
    if actual_sweeps < args.total_sweeps:
        print("注意：该关键帧历史 sweep 不足；记录实际帧数，不复制点云补齐。", flush=True)

    # *=================== LiDAR：两种范围使用同一份聚合点云 ===================
    with torch.no_grad():
        for radius in (40, 80):

            voxel_cfg = deepcopy(encoders.lidar.voxelize)
            voxel_cfg["point_cloud_range"] = [-radius, -radius, -5, radius, radius, 3]
            voxel_cfg["max_voxels"] = tuple(voxel_cfg["max_voxels"])
            voxelizer = Voxelization(**voxel_cfg).cuda().eval()
            voxels, coordinates, sizes = voxelizer(points)

            if not len(voxels):
                raise ValueError("范围内没有有效体素")
            # 与 BEVFusion.voxelize 相同：体素内点特征取均值，坐标前补 batch 编号。
            features = voxels.sum(1) / sizes.type_as(voxels).view(-1, 1)
            coordinates = torch.nn.functional.pad(coordinates, (1, 0), value=0)
            lidar_cfg = deepcopy(encoders.lidar.backbone)
            grid = np.round(np.array([2 * radius, 2 * radius, 8]) / voxel_cfg["voxel_size"]).astype(int)
            lidar_cfg["sparse_shape"] = [int(grid[0]), int(grid[1]), int(grid[2]) + 1]
            lidar = build_backbone(lidar_cfg).cuda().eval()
            counter = ConvCounter(lidar, SparseConvolution)
            try:
                lidar(features.contiguous(), coordinates, 1)
                record("lidar_sparse_encoder", f"xy=+/-{radius}m,z=[-5,3]m", counter,
                       active_voxels=len(voxels), voxel_size=list(voxel_cfg["voxel_size"]),
                       sparse_shape=lidar_cfg["sparse_shape"], max_voxels=voxel_cfg["max_voxels"][1],
                       voxel_cap_reached=len(voxels) >= voxel_cfg["max_voxels"][1])
            finally:
                counter.close()
            del lidar, voxelizer, voxels, features, coordinates, sizes
            torch.cuda.empty_cache()

    # *=================== 保存结果和统计口径 ===================
    output = Path(args.out_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    report = dict(config=str(config_path), daocc_root=str(Path(args.daocc_root).resolve()),
                  annotation_file=str(ann_path), validation_samples=len(infos),
                  sample_token=selected["token"], raw_keyframe_points=point_count(selected),
                  merged_points=len(points), requested_total_sweeps=args.total_sweeps,
                  actual_total_sweeps=actual_sweeps, selection="specified_token" if args.sample_token else "largest_raw_keyframe",
                  counting="Convolution/linear MACs only; FLOPs=2*MACs; excludes bias, BN, activation, interpolation, residual adds, voxelization and indexing",
                  camera_scope="backbone+neck, one camera; all_camera_gflops also provided; excludes view transform",
                  camera_resolution_ratio=results[1]["macs"] / results[0]["macs"],
                  lidar_range_ratio=results[3]["macs"] / results[2]["macs"],
                  paper_reference=dict(camera_gflops=[89.04, 356.16], lidar_mflops=[156.49, 166.0],
                                       warning="Reference only: original counting convention/sample unspecified; not guaranteed numerically equivalent"),
                  results=results)
    (output / "fig4_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (output / "fig4_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["module", "setting", "gmacs", "gflops", "backbone_gmacs", "neck_gmacs", "all_camera_gmacs", "all_camera_backbone_gmacs", "all_camera_gflops"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"结果已保存：{output.resolve()}")
    #* 测量完自动绘图；JSON/CSV 已先保存，绘图失败也不会丢失测量结果。
    plot_results(report, output, args.plot_camera_scope)


if __name__ == "__main__":
    main()
