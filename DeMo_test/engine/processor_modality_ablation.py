"""
多模态缺失训练与评估处理器
用于评估不同模态对ReID性能的影响
"""
import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist
import json
import numpy as np
from collections import defaultdict


def create_modality_missing_patterns():
    """
    创建模态缺失模式
    返回: 字典，key为模式名称，value为保留的模态列表
    """
    patterns = {
        'all': ['RGB', 'NI', 'TI'],          # 所有模态都存在（基准）
        'missing_RGB': ['NI', 'TI'],         # 缺失RGB
        'missing_NI': ['RGB', 'TI'],         # 缺失近红外
        'missing_TI': ['RGB', 'NI'],         # 缺失热红外
    }
    return patterns


def apply_modality_mask(img_dict, kept_modalities):
    """
    对输入图像应用模态掩码

    Args:
        img_dict: 包含RGB、NI、TI的字典
        kept_modalities: 保留的模态列表

    Returns:
        masked_img: 应用掩码后的图像字典（缺失的模态用零张量替代）
    """
    masked_img = {}
    for modality in ['RGB', 'NI', 'TI']:
        if modality in kept_modalities:
            masked_img[modality] = img_dict[modality]
        else:
            # 用零张量替代缺失的模态
            masked_img[modality] = torch.zeros_like(img_dict[modality])
    return masked_img


def do_train_with_modality_ablation(cfg,
                                    model,
                                    center_criterion,
                                    train_loader,
                                    val_loader,
                                    optimizer,
                                    optimizer_center,
                                    scheduler,
                                    loss_fn,
                                    num_query,
                                    local_rank,
                                    warmup_epochs=10):
    """
    带模态消融研究的训练函数

    在warm-up阶段后，每轮训练会：
    1. 使用四种模态缺失模式分别训练
    2. 评估每种模式的性能
    3. 记录并分析各模态的影响
    """
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("DeMo.modality_ablation")
    logger.info('=' * 80)
    logger.info('开始模态消融训练')
    logger.info(f'Warm-up轮数: {warmup_epochs}')
    logger.info(f'总训练轮数: {epochs}')
    logger.info('=' * 80)

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    scaler = amp.GradScaler()
    test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM

    # 记录最佳性能
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}

    # 记录各模态缺失的性能影响
    modality_ablation_results = defaultdict(lambda: defaultdict(list))

    # 获取模态缺失模式
    modality_patterns = create_modality_missing_patterns()

    # ==================== 训练循环 ====================
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # ========== Warm-up阶段：使用所有模态正常训练 ==========
        if epoch <= warmup_epochs:
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch}/{epochs} - WARM-UP阶段 (使用所有模态)")
            logger.info(f"{'='*80}")

            loss_meter.reset()
            acc_meter.reset()
            scheduler.step(epoch)
            model.train()

            for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
                optimizer.zero_grad()
                optimizer_center.zero_grad()

                img = {'RGB': img['RGB'].to(device),
                       'NI': img['NI'].to(device),
                       'TI': img['TI'].to(device)}
                target = vid.to(device)
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)

                with amp.autocast(enabled=True):
                    output = model(img, label=target, cam_label=target_cam, view_label=target_view)
                    loss = compute_loss(output, loss_fn, target, target_cam)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    scaler.step(optimizer_center)
                    scaler.update()

                acc = compute_accuracy(output, target)
                loss_meter.update(loss.item(), img['RGB'].shape[0])
                acc_meter.update(acc, 1)

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

            # Warm-up阶段评估
            if epoch % eval_period == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"Warm-up Epoch {epoch} - 评估所有模态")
                logger.info(f"{'='*80}")
                if test_sign:
                    _, _ = training_neat_eval(cfg, model, val_loader, device, evaluator,
                                            epoch, logger, return_pattern=1)
                    _, _ = training_neat_eval(cfg, model, val_loader, device, evaluator,
                                            epoch, logger, return_pattern=2)
                mAP, cmc = training_neat_eval(cfg, model, val_loader, device, evaluator,
                                            epoch, logger, return_pattern=3)

                # 记录warm-up阶段的性能
                modality_ablation_results['all']['mAP'].append(mAP)
                modality_ablation_results['all']['Rank-1'].append(cmc[0])

                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(),
                             os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))

        # ========== Warm-up后：使用不同模态缺失模式训练和评估 ==========
        else:
            logger.info(f"\n{'='*80}")
            logger.info(f"Epoch {epoch}/{epochs} - 模态消融阶段")
            logger.info(f"{'='*80}")

            # 对每种模态缺失模式进行训练和评估
            for pattern_name, kept_modalities in modality_patterns.items():
                logger.info(f"\n{'-'*80}")
                logger.info(f"当前模式: {pattern_name}")
                logger.info(f"保留的模态: {kept_modalities}")
                logger.info(f"{'-'*80}")

                loss_meter.reset()
                acc_meter.reset()
                scheduler.step(epoch)
                model.train()

                # 使用当前模态模式训练一个epoch
                for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
                    optimizer.zero_grad()
                    optimizer_center.zero_grad()

                    # 准备输入数据
                    img_raw = {'RGB': img['RGB'].to(device),
                             'NI': img['NI'].to(device),
                             'TI': img['TI'].to(device)}

                    # 应用模态掩码
                    img_masked = apply_modality_mask(img_raw, kept_modalities)

                    target = vid.to(device)
                    target_cam = target_cam.to(device)
                    target_view = target_view.to(device)

                    with amp.autocast(enabled=True):
                        output = model(img_masked, label=target, cam_label=target_cam,
                                     view_label=target_view)
                        loss = compute_loss(output, loss_fn, target, target_cam)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                        for param in center_criterion.parameters():
                            param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                        scaler.step(optimizer_center)
                        scaler.update()

                    acc = compute_accuracy(output, target)
                    loss_meter.update(loss.item(), img_raw['RGB'].shape[0])
                    acc_meter.update(acc, 1)

                    torch.cuda.synchronize()
                    if (n_iter + 1) % log_period == 0:
                        logger.info("[{}] Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}"
                                    .format(pattern_name, epoch, (n_iter + 1), len(train_loader),
                                            loss_meter.avg, acc_meter.avg))

                # 评估当前模态模式
                if epoch % eval_period == 0:
                    logger.info(f"\n评估模式: {pattern_name}")

                    # 使用相同的模态掩码进行评估
                    mAP, cmc = evaluate_with_modality_mask(
                        cfg, model, val_loader, device, evaluator,
                        epoch, logger, kept_modalities, return_pattern=3)

                    # 记录结果
                    modality_ablation_results[pattern_name]['mAP'].append(mAP)
                    modality_ablation_results[pattern_name]['Rank-1'].append(cmc[0])
                    modality_ablation_results[pattern_name]['Rank-5'].append(cmc[4])
                    modality_ablation_results[pattern_name]['Rank-10'].append(cmc[9])
                    modality_ablation_results[pattern_name]['epoch'].append(epoch)

                    logger.info(f"{pattern_name} 结果 - mAP: {mAP:.1%}, Rank-1: {cmc[0]:.1%}")

            # 每轮结束后分析模态影响
            if epoch % eval_period == 0:
                analyze_modality_impact(modality_ablation_results, epoch, logger)

        # 保存检查点
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                             os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth'))
            else:
                torch.save(model.state_dict(),
                         os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + f'_{epoch}.pth'))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / len(train_loader)
        logger.info(f"Epoch {epoch} 完成. 每批次时间: {time_per_batch:.3f}s")

    # ==================== 训练结束，生成最终分析报告 ====================
    logger.info(f"\n{'='*80}")
    logger.info("训练完成！生成模态影响分析报告...")
    logger.info(f"{'='*80}")

    generate_final_report(modality_ablation_results, cfg.OUTPUT_DIR, logger, warmup_epochs)

    return modality_ablation_results


def compute_loss(output, loss_fn, target, target_cam):
    """计算损失"""
    loss = 0
    if len(output) % 2 == 1:
        index = len(output) - 1
        for i in range(0, index, 2):
            loss_tmp = loss_fn(score=output[i], feat=output[i + 1],
                             target=target, target_cam=target_cam)
            loss = loss + loss_tmp
        loss = loss + output[-1]
    else:
        for i in range(0, len(output), 2):
            loss_tmp = loss_fn(score=output[i], feat=output[i + 1],
                             target=target, target_cam=target_cam)
            loss = loss + loss_tmp
    return loss


def compute_accuracy(output, target):
    """计算准确率"""
    if isinstance(output, list):
        acc = (output[0][0].max(1)[1] == target).float().mean()
    else:
        acc = (output[0].max(1)[1] == target).float().mean()
    return acc


def evaluate_with_modality_mask(cfg, model, val_loader, device, evaluator,
                                epoch, logger, kept_modalities, return_pattern=1):
    """
    使用指定的模态掩码进行评估
    """
    evaluator.reset()
    model.eval()

    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img_raw = {'RGB': img['RGB'].to(device),
                      'NI': img['NI'].to(device),
                      'TI': img['TI'].to(device)}

            # 应用模态掩码
            img_masked = apply_modality_mask(img_raw, kept_modalities)

            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)

            feat = model(img_masked, cam_label=camids, view_label=target_view,
                        return_pattern=return_pattern)

            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, scenceids, _))
            else:
                evaluator.update((feat, vid, camid, _))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()

    logger.info(f"保留模态: {kept_modalities}")
    logger.info(f"mAP: {mAP:.1%}")
    for r in [1, 5, 10]:
        logger.info(f"CMC curve, Rank-{r:<3}: {cmc[r - 1]:.1%}")

    torch.cuda.empty_cache()
    return mAP, cmc


def analyze_modality_impact(results, current_epoch, logger):
    """
    分析各模态对性能的影响
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Epoch {current_epoch} - 模态影响分析")
    logger.info(f"{'='*80}")

    # 获取基准性能（所有模态）
    if 'all' not in results or len(results['all']['mAP']) == 0:
        logger.warning("基准性能（所有模态）数据不可用")
        return

    baseline_mAP = results['all']['mAP'][-1]
    baseline_rank1 = results['all']['Rank-1'][-1]

    logger.info(f"\n基准性能（所有模态）:")
    logger.info(f"  mAP: {baseline_mAP:.1%}")
    logger.info(f"  Rank-1: {baseline_rank1:.1%}")

    # 分析各个缺失模式的影响
    modality_impacts = []

    for pattern_name in ['missing_RGB', 'missing_NI', 'missing_TI']:
        if pattern_name in results and len(results[pattern_name]['mAP']) > 0:
            current_mAP = results[pattern_name]['mAP'][-1]
            current_rank1 = results[pattern_name]['Rank-1'][-1]

            mAP_drop = baseline_mAP - current_mAP
            rank1_drop = baseline_rank1 - current_rank1

            mAP_drop_pct = (mAP_drop / baseline_mAP * 100) if baseline_mAP > 0 else 0
            rank1_drop_pct = (rank1_drop / baseline_rank1 * 100) if baseline_rank1 > 0 else 0

            modality_name = pattern_name.replace('missing_', '')

            logger.info(f"\n缺失 {modality_name} 的影响:")
            logger.info(f"  mAP: {current_mAP:.1%} (下降 {mAP_drop:.1%}, {mAP_drop_pct:.1f}%)")
            logger.info(f"  Rank-1: {current_rank1:.1%} (下降 {rank1_drop:.1%}, {rank1_drop_pct:.1f}%)")

            modality_impacts.append({
                'modality': modality_name,
                'mAP_drop': mAP_drop,
                'mAP_drop_pct': mAP_drop_pct,
                'rank1_drop': rank1_drop,
                'rank1_drop_pct': rank1_drop_pct
            })

    # 排序找出影响最大的模态
    if modality_impacts:
        modality_impacts.sort(key=lambda x: x['mAP_drop'], reverse=True)

        logger.info(f"\n{'='*80}")
        logger.info("模态重要性排名（按mAP下降幅度）:")
        logger.info(f"{'='*80}")

        for i, impact in enumerate(modality_impacts, 1):
            logger.info(f"{i}. {impact['modality']}: mAP下降 {impact['mAP_drop']:.1%} "
                       f"({impact['mAP_drop_pct']:.1f}%), "
                       f"Rank-1下降 {impact['rank1_drop']:.1%} ({impact['rank1_drop_pct']:.1f}%)")

        logger.info(f"\n结论: {modality_impacts[0]['modality']} 是主导模态（缺失后性能下降最大）")

    logger.info(f"{'='*80}\n")


def generate_final_report(results, output_dir, logger, warmup_epochs):
    """
    生成最终的模态影响分析报告
    """
    logger.info(f"\n{'='*80}")
    logger.info("最终模态影响分析报告")
    logger.info(f"{'='*80}")

    # 保存详细结果到JSON
    results_file = os.path.join(output_dir, 'modality_ablation_results.json')

    # 转换numpy类型为Python原生类型
    results_serializable = {}
    for pattern, metrics in results.items():
        results_serializable[pattern] = {}
        for metric, values in metrics.items():
            if isinstance(values, list):
                results_serializable[pattern][metric] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in values
                ]
            else:
                results_serializable[pattern][metric] = values

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)

    logger.info(f"详细结果已保存至: {results_file}")

    # 计算平均影响（warm-up后的所有epoch）
    logger.info(f"\n{'='*80}")
    logger.info(f"平均性能对比（Warm-up后所有epoch平均）")
    logger.info(f"{'='*80}")

    summary = {}

    for pattern_name, metrics in results.items():
        if len(metrics.get('mAP', [])) > 0:
            # 只统计warm-up后的结果
            mAP_values = metrics['mAP']
            rank1_values = metrics['Rank-1']

            avg_mAP = np.mean(mAP_values)
            avg_rank1 = np.mean(rank1_values)
            std_mAP = np.std(mAP_values)
            std_rank1 = np.std(rank1_values)

            summary[pattern_name] = {
                'avg_mAP': avg_mAP,
                'std_mAP': std_mAP,
                'avg_rank1': avg_rank1,
                'std_rank1': std_rank1
            }

            logger.info(f"\n{pattern_name}:")
            logger.info(f"  平均 mAP: {avg_mAP:.1%} ± {std_mAP:.1%}")
            logger.info(f"  平均 Rank-1: {avg_rank1:.1%} ± {std_rank1:.1%}")

    # 最终模态重要性排名
    if 'all' in summary:
        baseline = summary['all']

        logger.info(f"\n{'='*80}")
        logger.info("最终模态重要性排名")
        logger.info(f"{'='*80}")

        modality_importance = []

        for pattern_name in ['missing_RGB', 'missing_NI', 'missing_TI']:
            if pattern_name in summary:
                mAP_drop = baseline['avg_mAP'] - summary[pattern_name]['avg_mAP']
                rank1_drop = baseline['avg_rank1'] - summary[pattern_name]['avg_rank1']

                modality_name = pattern_name.replace('missing_', '')
                modality_importance.append({
                    'modality': modality_name,
                    'mAP_drop': mAP_drop,
                    'rank1_drop': rank1_drop
                })

        modality_importance.sort(key=lambda x: x['mAP_drop'], reverse=True)

        for i, item in enumerate(modality_importance, 1):
            logger.info(f"{i}. {item['modality']}: "
                       f"平均mAP下降 {item['mAP_drop']:.1%}, "
                       f"平均Rank-1下降 {item['rank1_drop']:.1%}")

        logger.info(f"\n{'*'*80}")
        logger.info(f"主导模态: {modality_importance[0]['modality']}")
        logger.info(f"缺失该模态后，mAP平均下降 {modality_importance[0]['mAP_drop']:.1%}")
        logger.info(f"{'*'*80}")

    # 保存摘要报告
    summary_file = os.path.join(output_dir, 'modality_impact_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("多模态ReID - 模态影响分析报告\n")
        f.write("="*80 + "\n\n")

        f.write(f"Warm-up轮数: {warmup_epochs}\n")
        f.write(f"总训练轮数: {len(results['all']['epoch']) if 'all' in results and 'epoch' in results['all'] else 'N/A'}\n\n")

        f.write("平均性能对比:\n")
        f.write("-"*80 + "\n")
        for pattern_name, stats in summary.items():
            f.write(f"\n{pattern_name}:\n")
            f.write(f"  平均 mAP: {stats['avg_mAP']:.1%} ± {stats['std_mAP']:.1%}\n")
            f.write(f"  平均 Rank-1: {stats['avg_rank1']:.1%} ± {stats['std_rank1']:.1%}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("模态重要性排名:\n")
        f.write("="*80 + "\n")
        for i, item in enumerate(modality_importance, 1):
            f.write(f"{i}. {item['modality']}: "
                   f"平均mAP下降 {item['mAP_drop']:.1%}, "
                   f"平均Rank-1下降 {item['rank1_drop']:.1%}\n")

        f.write("\n" + "*"*80 + "\n")
        f.write(f"主导模态: {modality_importance[0]['modality']}\n")
        f.write(f"缺失该模态后，mAP平均下降 {modality_importance[0]['mAP_drop']:.1%}\n")
        f.write("*"*80 + "\n")

    logger.info(f"\n摘要报告已保存至: {summary_file}")
    logger.info(f"\n{'='*80}\n")


def training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern=1):
    """原始的评估函数，保持兼容性"""
    evaluator.reset()
    model.eval()
    logger.info("~" * 50)
    if return_pattern == 1:
        logger.info("Current is the ori feature testing!")
    elif return_pattern == 2:
        logger.info("Current is the moe feature testing!")
    else:
        logger.info("Current is the [moe,ori] feature testing!")
    logger.info("~" * 50)

    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view, return_pattern=return_pattern)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, scenceids, _))
            else:
                evaluator.update((feat, vid, camid, _))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("~" * 50)
    torch.cuda.empty_cache()
    return mAP, cmc
