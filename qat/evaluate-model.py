"""一次推理，同时评估 DAOcc 检测、Occ3D mIoU 和 Ray-IoU。

支持 PyTorch fp32/fp16 和 TensorRT int8/fp16；完整遍历有真值的 split。
"""

import argparse
import csv
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys


# * 论文表 1 的占据类别顺序；末尾 free 不作为语义类别列。
OCC_CLASSES = ('others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
               'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade', 'vegetation')
# * 论文“检测性能”段落的十类顺序，与占据类别顺序不同。
DET_CLASSES = ('car', 'truck', 'bus', 'trailer', 'construction_vehicle',
               'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')


def geometry_iou(hist):
    # * 将 0—16 类合并为 occupied，17 类为 free；占据类别之间的误分类仍算几何 TP。
    # 使用评估器累计混淆矩阵，保持相机掩码和有效真值过滤完全一致。
    tp = float(hist[:17, :17].sum())
    union = tp + float(hist[:17, 17].sum()) + float(hist[17, :17].sum())
    return 100.0 * tp / union if union else None


def export_tables(result_dir, summary):
    # * 表格只整理本次实测结果，不混入论文数值；JSON 保持原指标单位。
    metrics = summary['metrics']
    label = f"DAOcc ({summary['backend']} {summary['precision']})"

    def percent(key, scale=1):
        value = metrics.get(key)
        return None if value is None else value * scale

    # *=================== 占据预测：体素 IoU 与射线 Ray-IoU ===================
    # 语义 IoU：逐个体素比较某个类别，IoU_c = TP_c / (TP_c + FP_c + FN_c)。
    # mIoU：对 17 个非 free 类别的 IoU 取平均；当前实现忽略分母为零的 NaN 类别。
    # geometry_iou 则先将这些类别合并为“占据”：例如汽车错分成卡车，几何正确、语义错误。
    # 上述体素统计使用 mask_camera；它们与下面的射线统计不是同一种匹配方式。
    #
    # Ray-IoU：从评估器构造的起点沿相同方向，分别对预测和真值体素网格发射射线，
    # 得到沿射线首先遇到的占据表面的类别与距离（深度），再比较两者。
    # 当前实现只保留真值命中非 free 的射线，不用 mask_camera 筛选；并非所有体素逐个打分。
    # 对类别 c、容差 t，只有“预测与真值类别均为 c”且“深度绝对误差 < t”才计为 TP。
    # RayIoU_c@t = TP_c@t / (GT_c + Pred_c - TP_c@t)，再对非 free 类别取 nanmean。
    # 因此它同时约束表面语义和几何深度，不是仅计算深度命中率或普通三维框 IoU。
    # * @1、@2、@4 的单位是米，分别表示深度误差严格小于 1、2、4 m，绝不是帧数。
    # 例如真值是距离 20 m 的汽车，预测是 21.5 m 的汽车：@1 不匹配，@2 和 @4 匹配；
    # 如果预测成卡车，即使深度完全正确，也不能作为汽车类别的 TP。
    # 容差越大，几何要求越宽松；正常情况下 RayIoU@1 <= RayIoU@2 <= RayIoU@4。
    # RayIoU 总分：对三个阈值、全部非 free 类别的有效 IoU 值统一取 nanmean；
    # 在有效类别集合相同的通常情况下，等于三个 RayIoU@t 的算术平均。
    # JSON 中 RayIoU 保持 0—1，表格乘 100；体素 IoU/mIoU 的 JSON 原本就是 0—100。
    tables = [
        ('occupancy_table', '占据预测（类别顺序参照论文表 1）',
         ['Method', 'IoU', 'mIoU'] + list(OCC_CLASSES),
         [label, percent('occ/IoU'), percent('occ/mIoU')] +
         [percent(f'occ/{name}/iou') for name in OCC_CLASSES]),
        ('ray_iou_table', '射线占据预测（参照论文表 3）',
         ['Method', 'RayIoU', 'RayIoU@1', 'RayIoU@2', 'RayIoU@4'],
         [label] + [percent(name, 100) for name in ('RayIoU', 'RayIoU@1', 'RayIoU@2', 'RayIoU@4')]),
    ]
    # *=================== 三维检测：AP、mAP 与 NDS ===================
    # AP（Average Precision，平均精度）综合衡量一个类别的检出能力和误检情况。
    # 将该类预测框按置信度从高到低处理，与同帧尚未匹配的同类真值框匹配：
    # 匹配成功为 TP；没有匹配的预测（包括重复检测）为 FP；漏掉的真值为 FN。
    # Precision = TP / (TP + FP)，Recall = TP / (TP + FN)。随着保留的预测数量变化，
    # 得到精确率—召回率曲线，AP 是按官方协议对该曲线进行插值、截断和归一化后的积分。
    # 因此 AP 不是某个固定置信度阈值处的 Precision，也不是“预测框中正确框的比例”。
    # * nuScenes 匹配使用地面 XY 平面的框中心欧氏距离，而不是 3D 框重叠 IoU。
    # 分别在 0.5、1、2、4 m 中心距离阈值下计算 AP；默认官方协议的最低 recall/precision
    # 均为 0.1。表中每类 AP 是上述四个阈值 AP 的平均，mAP 再对十个检测类别平均。
    # 所以这里检测 AP 的距离阈值与 Ray-IoU 的射线深度阈值含义完全不同。
    #
    # NDS（nuScenes Detection Score）同时考虑检测 AP 和匹配目标的框/运动/属性质量。
    # 官方默认：NDS = [5*mAP + sum(max(0, 1-error_k))] / 10，所有项使用原始单位计算。
    # 五个 error_k 为官方跨类别聚合后的误差（通常按 2 m 中心距离匹配计算）：
    #   mATE：位置误差，单位 m；越小表示框中心越准确。
    #   mASE：尺度误差，1 - 对齐位置和朝向后的框 IoU；越小表示长宽高越准确。
    #   mAOE：朝向误差，单位 rad；按类别处理角度周期，越小越好。
    #   mAVE：平面速度误差，单位 m/s；越小表示预测速度越准确。
    #   mAAE：属性误差，1 - 属性准确率，例如车辆 moving/parked；越小越好。
    # 不适用的类别—误差组合由官方协议排除，例如交通锥没有速度/朝向/属性指标。
    # mAP 占 NDS 的 50%，其余五项各占 10%；误差项超过 1 时该项得分截断为 0。
    # * AP、mAP、NDS 均越高越好；JSON 使用 0—1，论文式表格乘 100 显示为百分数。
    # NDS 是综合得分，不是检测准确率，也不能仅凭 mAP 推算出来。
    class_aps = []
    for name in DET_CLASSES:
        values = [metrics.get(f'object/{name}_ap_dist_{distance}') for distance in ('0.5', '1.0', '2.0', '4.0')]
        class_aps.append(sum(values) / 4 * 100 if all(v is not None for v in values) else None)
    tables.append(('detection_table', '三维检测（类别顺序参照论文检测性能段落）',
                   ['Method', 'mAP', 'NDS'] + list(DET_CLASSES),
                   [label, percent('object/map', 100), percent('object/nds', 100)] + class_aps))
    text = [f"数据划分：{summary['split']}；样本数：{summary['samples']}；状态：{summary['status']}。",
            '所有表格数值单位为 %，保留两位小数；— 表示未定义或尚未完成，不能当作 0。',
            'IoU 为忽略语义类别的占据/空闲几何 IoU；IoU、mIoU 和各类 IoU 均使用相机掩码。',
            'Ray-IoU 沿用原射线协议，不使用相机掩码筛选。检测类别列为各类 AP。',
            '表格只借鉴论文列顺序；当前数据划分及训练设置不一定与论文相同。']
    for filename, title, headers, values in tables:
        cells = [v if isinstance(v, str) else ('—' if v is None or not math.isfinite(v) else f'{v:.2f}') for v in values]
        with (result_dir / (filename + '.csv')).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerow(cells)
        text.extend(['', title, '', '| ' + ' | '.join(headers) + ' |',
                     '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                     '| ' + ' | '.join(cells) + ' |'])
    (result_dir / 'evaluation_tables.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--backend', choices=['pytorch', 'tensorrt'], default='pytorch')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'])
    parser.add_argument('--ckpt', help='原始 PyTorch checkpoint')
    parser.add_argument('--model-dir', help='已构建的 TensorRT 引擎与 LiDAR ONNX 目录')
    parser.add_argument('--daocc-root', help='已编译扩展的 DAOcc 代码目录，默认使用子仓库')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--result-dir', required=True, help='评估 JSON 与检测评估文件保存目录')
    parser.add_argument('--out-dir', help='可选：逐帧预测保存目录，按场景分组')
    args = parser.parse_args()
    args.precision = args.precision or ('fp32' if args.backend == 'pytorch' else 'int8')
    if args.backend == 'pytorch' and (not args.ckpt or args.precision == 'int8'):
        parser.error('PyTorch 需要 --ckpt，precision 仅支持 fp32/fp16')
    if args.backend == 'tensorrt' and (not args.model_dir or args.precision == 'fp32'):
        parser.error('TensorRT 需要 --model-dir，precision 仅支持 int8/fp16')
    if args.device < 0 or (args.workers is not None and args.workers < 0):
        parser.error('device 和 workers 不能为负数')
    return args


def json_value(value):
    # 未定义的类别指标转为 null，不输出非标准 JSON 的 NaN。
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, 'tolist'):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def clear_gpu_cache(torch, device, stage):
    # * 等待 GPU 工作完成，再回收循环引用与 PyTorch 空闲缓存。
    # empty_cache 不能释放仍被引用的张量，也不能代替 TensorRT 对象的析构。
    import gc
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    stats = dict(free_mib=free / 2**20, total_mib=total / 2**20,
                 torch_allocated_mib=torch.cuda.memory_allocated(device) / 2**20,
                 torch_reserved_mib=torch.cuda.memory_reserved(device) / 2**20)
    print(f'{stage}：GPU 可用 {stats["free_mib"]:.1f} MiB，'
          f'PyTorch 活跃分配 {stats["torch_allocated_mib"]:.1f} MiB', flush=True)
    return stats


def normalize_prediction(prediction):
    # * 部署输出转换为原生 DAOcc 评估格式；保持和 benchmark-latency.py 一致。
    import numpy as np
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    if 'boxes_3d' in prediction:
        # result = dict(prediction)
        # * 只保留评估需要的字段，避免额外输出中隐藏的 GPU 特征一直占用显存。
        result = {key: prediction[key] for key in ('boxes_3d', 'scores_3d', 'labels_3d', 'occ_pred')}
        result['boxes_3d'] = result['boxes_3d'].to('cpu')
        for key in ('scores_3d', 'labels_3d'):
            result[key] = result[key].detach().cpu()
        occ = result['occ_pred']
        if isinstance(occ, torch.Tensor):
            occ = occ.detach().cpu().numpy()
    else:
        boxes = torch.from_numpy(np.array(prediction['det_outputs'], dtype=np.float32, copy=True))
        if boxes.ndim != 2 or boxes.shape[1] != 11:
            raise ValueError(f'检测输出应为 (N, 11)，实际为 {tuple(boxes.shape)}')
        # C++ 返回框的几何中心，评估接口要求底面中心；只修正 z，不改变坐标系。
        boxes[:, 2] -= boxes[:, 5] * 0.5
        result = dict(boxes_3d=LiDARInstance3DBoxes(boxes[:, :9].clone(), box_dim=9),
                      scores_3d=boxes[:, 10].clone(), labels_3d=boxes[:, 9].long())
        occ = prediction['occ_outputs']
    occ = np.asarray(occ)
    if occ.size != 200 * 200 * 16 or not np.isin(occ, np.arange(18)).all():
        raise ValueError('Occ3D 输出必须为 200×200×16 个 0—17 类别编号')
    # 拷贝防止 C++ 下次推理复用输出缓冲区；uint8 降低完整验证集的 CPU 内存需求。
    result['occ_pred'] = np.array(occ, dtype=np.uint8, copy=True).reshape(200, 200, 16)
    return result


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    daocc_root = Path(args.daocc_root).expanduser().resolve() if args.daocc_root else root / 'DAOcc'
    if not (daocc_root / 'mmdet3d').is_dir():
        raise FileNotFoundError(daocc_root / 'mmdet3d')
    sys.path.insert(0, str(daocc_root))
    sys.path.insert(0, str(root / 'build'))
    # * 复用已有后端和输入准备，导入不会执行测速脚本的 main。
    spec = importlib.util.spec_from_file_location('daocc_benchmark', root / 'qat/benchmark-latency.py')
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    import numpy as np
    import torch
    from tqdm import tqdm
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet.apis import set_random_seed
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.utils import recursive_eval

    torch.cuda.set_device(args.device)
    # * 在模型加载和推理之前编译/加载 Ray-IoU 的 dvr 扩展；失败时直接停止。
    # get_dvr 会复用编译缓存和进程内模块，正式射线评估不会重新加载。
    print('预检查 Ray-IoU CUDA 扩展（已有有效缓存时无需重新编译）……', flush=True)
    from mmdet3d.core.evaluation.ray_metrics import get_dvr
    get_dvr()
    print('Ray-IoU CUDA 扩展已就绪。', flush=True)
    set_random_seed(args.seed, deterministic=True)
    if args.backend == 'pytorch' and args.precision == 'fp32':
        # FP32 对照关闭 TF32；FP16 沿用原测试入口的 MMCV 混合精度。
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    configs.load(str(Path(args.config).expanduser().resolve()), recursive=True)
    cfg = Config(recursive_eval(configs))
    dataset_cfg = deepcopy(cfg.data[args.split])
    dataset_cfg.test_mode = True
    dataset_cfg.pop('samples_per_gpu', None)
    dataset = build_dataset(dataset_cfg)
    if not len(dataset) or getattr(dataset, 'data_type', None) != 'occ3d':
        raise ValueError('需要非空的 Occ3D 数据集，并提供占据及检测真值')
    for info in dataset.data_infos:
        # * 提前检查场景字段，避免整轮推理完成后才在 EgoPoseDataset 中报错。
        if not info.get('scene_token'):
            raise ValueError('info 缺少 scene_token，请使用已更新的 create_data.py 重新生成 PKL')
        path = Path(info['occ3d']['occ_path']) / 'labels.npz'
        if not path.is_file():
            raise FileNotFoundError(path)
    loader = build_dataloader(dataset, samples_per_gpu=1,
                             workers_per_gpu=args.workers if args.workers is not None else cfg.data.workers_per_gpu,
                             dist=False, shuffle=False)
    result_dir = Path(args.result_dir).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    scenes = benchmark.prediction_scene_names(dataset) if out_dir else None
    view_transform = None
    if args.backend == 'pytorch':
        core = benchmark.PyTorchBackend(cfg, args.ckpt, args.precision, args.device)
    else:
        import libpybev
        from mmdet3d.models import BEVTransformV2
        vt_cfg = dict(cfg.model.encoders.camera.vtransform)
        if vt_cfg.pop('type') != 'BEVTransformV2':
            raise ValueError('当前部署输入准备仅支持 BEVTransformV2')
        view_transform = BEVTransformV2(**vt_cfg).eval()
        model_dir = Path(args.model_dir).expanduser().resolve()
        required = ['build/camera.backbone.plan', 'lidar.backbone.xyz.onnx',
                    'build/fuser.plan', 'build/head.object.plan', 'build/head.occ.plan']
        for name in required:
            if not (model_dir / name).is_file():
                raise FileNotFoundError(model_dir / name)
        # precision 选择稀疏运行时模式，普通 plan 的实际精度由构建时决定。
        core = libpybev.load_bevfusion(
            str(model_dir / required[0]), str(model_dir / 'build/camera.vtransform.plan'),
            *(str(model_dir / name) for name in required[1:]), args.precision)
        if core is None:
            raise RuntimeError('TensorRT 引擎加载失败')
        if hasattr(core, 'set_timer_verbose'):
            core.set_timer_verbose(False)
    core.set_timer(False)
    results = []
    # * 完整遍历一次，保持与 data_infos 顺序一致；检测官方评估不能用重复抽样代替。
    with torch.no_grad():
        for index, data in enumerate(tqdm(loader, desc='推理', unit='帧')):
            if args.backend == 'pytorch':
                prediction = core.forward(core.prepare(data))
            else:
                images, refs, points = benchmark.prepare_inputs(data, view_transform, np)
                prediction = core.forward(images, refs, points, with_normalization=False)
            prediction = normalize_prediction(prediction)
            results.append(prediction)
            if out_dir:
                info = dataset.data_infos[index]
                benchmark.save_prediction(out_dir, prediction, index, index, info, {}, np, scenes[info['token']])
    # 射线评估也使用 GPU；先释放推理模型，降低同时常驻的显存需求。
    # del core, view_transform, data
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()
    torch.cuda.synchronize(args.device)
    free_before, _ = torch.cuda.mem_get_info(args.device)
    if args.backend == 'pytorch':
        # * 计时包装器含循环引用，先显式将参数/缓冲区移到 CPU，再删除整个后端。
        core.model.cpu()
    # TensorRT 后端删除最后一个 Python 引用后，由 C++ 析构释放引擎及其缓冲区。
    del core, view_transform, data, prediction, loader
    release_stats = clear_gpu_cache(torch, args.device, '推理后端释放后')
    release_stats['free_before_release_mib'] = free_before / 2**20
    summary = dict(backend=args.backend, precision=args.precision, split=args.split,
                   config=str(Path(args.config).expanduser().resolve()), daocc_root=str(daocc_root),
                   checkpoint=args.ckpt, model_dir=args.model_dir, samples=len(results),
                   miou_camera_mask=True, status='evaluating', metrics={})
    # 整卡可用量会受其他进程影响，不把它标为本进程显存。
    summary['gpu_after_inference_release'] = release_stats
    write_json(result_dir / 'evaluation_summary.json', summary)
    # * 三项独立调用，不改动原数据集的二选一接口，也不重复运行检测或模型。
    stages = [
        ('detection', lambda: dataset.evaluate(
            [{k: v for k, v in r.items() if k != 'occ_pred'} for r in results],
            jsonfile_prefix=str(result_dir / 'detection'))),
        ('occupancy_miou', lambda: dataset.evaluate_occ(results, occ_metric='mIoU')),
        ('occupancy_ray_iou', lambda: dataset.evaluate_occ(results, occ_metric='Ray-IoU')),
    ]
    for name, evaluate in stages:
        print(f'\n开始评估：{name}', flush=True)
        try:
            if name == 'occupancy_ray_iou':
                # * 检测/mIoU 完成后再清理一次临时缓存，为射线计算留出空间。
                summary['gpu_before_ray_iou'] = clear_gpu_cache(torch, args.device, 'Ray-IoU 开始前')
            metrics = evaluate()
            if not metrics:
                raise RuntimeError(f'{name} 未返回指标，请检查预测字段和数据集')
            if name == 'occupancy_miou':
                # * 从同一次 mIoU 评估的混淆矩阵补充几何 IoU，无需重复推理或读取真值。
                metrics['occ/IoU'] = geometry_iou(dataset.occ_eval_metrics.hist)
            write_json(result_dir / (name + '.json'), metrics)
            summary['metrics'].update(metrics)
        except Exception as error:
            summary.update(status='failed', failed_stage=name, error=str(error))
            write_json(result_dir / 'evaluation_summary.json', summary)
            raise
        write_json(result_dir / 'evaluation_summary.json', summary)
    summary['status'] = 'complete'
    write_json(result_dir / 'evaluation_summary.json', summary)
    # * 完成全部评估后，除 JSON 外导出三张 CSV 和一个含三表的 Markdown 文件。
    export_tables(result_dir, summary)
    print(f'全部指标已保存：{result_dir / "evaluation_summary.json"}')


if __name__ == '__main__':
    main()
