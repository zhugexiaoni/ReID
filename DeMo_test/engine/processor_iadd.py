"""
IADD Training Processor

Training processor for IADD Plugin integration
"""

import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP


def _iadd_is_active(cfg, epoch: int) -> bool:
    """Enable IADD only after warmup epochs.

    Semantics:
        - Set `MODEL.IADD.WARMUP_EPOCHS = N`.
        - For epochs <= N: do NOT apply any IADD losses (no distill, no hybrid triplet).
        - Base loss (CE + original Triplet from `loss_fn`) is ALWAYS enabled.

    This matches your requirement: warmup delays *enabling IADD*, rather than warming
    up `loss_hybrid` weight.
    """
    warm = int(getattr(cfg.MODEL.IADD, 'WARMUP_EPOCHS', 0))
    return epoch > warm


from torch.cuda import amp
import torch.distributed as dist
from layers.iadd import IADDPlugin, HybridTripletLoss


def do_train_iadd(cfg,
                  model,
                  center_criterion,
                  train_loader,
                  val_loader,
                  optimizer,
                  optimizer_center,
                  scheduler,
                  loss_fn,
                  num_query,
                  local_rank):
    """Train with IADD.

    Returns:
        best_index (dict): best metrics tracked during training.
    """
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    # Optional: disable any model saving (useful for grid search)
    no_save = bool(getattr(cfg.SOLVER, 'NO_SAVE', False))
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("DeMo.train")
    logger.info('start training with IADD')

    # IADD 配置
    use_iadd = bool(cfg.MODEL.IADD.ENABLED)
    if use_iadd:
        logger.info("Initializing IADD Plugin...")
        iadd_plugin = IADDPlugin(
            temperature=cfg.MODEL.IADD.TEMPERATURE,
            hard_neg_k=cfg.MODEL.IADD.HARD_NEG_K,
            lambda_distill=cfg.MODEL.IADD.LAMBDA_DISTILL,
            lambda_hybrid=cfg.MODEL.IADD.LAMBDA_HYBRID
        ).to(device)
        hybrid_triplet_loss = HybridTripletLoss(margin=cfg.SOLVER.MARGIN).to(device)
        logger.info("IADD Plugin initialized.")

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    mcd_rgb_meter = AverageMeter()
    mcd_ir_meter = AverageMeter()
    mcd_ni_meter = AverageMeter()
    mcd_ti_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM

    # train
    best_index = {
        'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0,
        'ori_mAP': 0, 'ori_Rank-1': 0, 'ori_Rank-5': 0, 'ori_Rank-10': 0,
        'moe_mAP': 0, 'moe_Rank-1': 0, 'moe_Rank-5': 0, 'moe_Rank-10': 0,
    }
    for epoch in range(1, epochs + 1):
        iadd_active = use_iadd and _iadd_is_active(cfg, epoch)
        if use_iadd and (epoch == 1 or epoch % eval_period == 0):
            logger.info(
                f"[IADD] active={int(iadd_active)} warmup_epochs={int(getattr(cfg.MODEL.IADD,'WARMUP_EPOCHS',0))} "
                f"lambda_distill={float(cfg.MODEL.IADD.LAMBDA_DISTILL):.4f} lambda_hybrid={float(cfg.MODEL.IADD.LAMBDA_HYBRID):.4f}"
            )

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        mcd_rgb_meter.reset()
        mcd_ir_meter.reset()
        mcd_ni_meter.reset()
        mcd_ti_meter.reset()

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

                loss_total = 0

               # --- 常规 Loss ---
                if isinstance(output, dict):
                    # Determine base loss input based on available outputs
                    if 'moe_score' in output and output['moe_score'] is not None:
                        # MOE mode: use moe output
                        score_final = output['moe_score']
                        feat_final = output['moe_feat']
                        loss_base = loss_fn(score=score_final, feat=feat_final,
                                           target=target, target_cam=target_cam)
                    elif 'ori_score' in output and output['ori_score'] is not None:
                        # DIRECT=1 mode: use ori output
                        score_final = output['ori_score']
                        feat_final = output['ori_feat']
                        loss_base = loss_fn(score=score_final, feat=feat_final,
                                           target=target, target_cam=target_cam)
                    elif 'logits_dict' in output:
                        # DIRECT=0 mode: sum up three modality losses
                        r_score = output['logits_dict']['RGB']
                        n_score = output['logits_dict']['NI']
                        t_score = output['logits_dict']['TI']
                        r_feat = output['feats_dict']['RGB']
                        n_feat = output['feats_dict']['NI']
                        t_feat = output['feats_dict']['TI']
                        
                        loss_r = loss_fn(score=r_score, feat=r_feat, target=target, target_cam=target_cam)
                        loss_n = loss_fn(score=n_score, feat=n_feat, target=target, target_cam=target_cam)
                        loss_t = loss_fn(score=t_score, feat=t_feat, target=target, target_cam=target_cam)
                        loss_base = (loss_r + loss_n + loss_t) / 3.0
                        score_final = r_score  # For accuracy logging
                    else:
                        raise ValueError("Unknown output format from model")
                    
                    loss_total += loss_base
                    # --- IADD 插件逻辑（warmup 前完全跳过） ---
                    if iadd_active:
                        if 'logits_dict' not in output or 'feats_dict' not in output:
                            logger.warning(
                                "IADD 已启用，但模型输出缺少 logits_dict 或 feats_dict，已跳过 IADD。"
                                "（常见原因：MODEL.DIRECT=1 时未返回 logits_dict）\n"
                                f"当前输出 keys={list(output.keys())}"
                            )

                    if iadd_active and 'logits_dict' in output and 'feats_dict' in output:
                        rgb_logits = output['logits_dict']['RGB']
                        rgb_feats = output['feats_dict']['RGB']

                        ni_logits = output['logits_dict']['NI']
                        ti_logits = output['logits_dict']['TI']
                        ni_feats = output['feats_dict']['NI']
                        ti_feats = output['feats_dict']['TI']

                        fusion_mode = getattr(cfg.MODEL.IADD, 'FUSION_MODE', 'mean')

                        if fusion_mode == 'mean':
                            ir_logits = (ni_logits + ti_logits) / 2.0
                            ir_feats = (ni_feats + ti_feats) / 2.0
                            m1_logits, m2_logits = rgb_logits, ir_logits
                            m1_feats, m2_feats = rgb_feats, ir_feats

                        elif fusion_mode in ('weak2_mcd', 'best2_mcd'):
                            with torch.no_grad():
                                mcd_rgb = iadd_plugin.compute_mcd_vectorized(
                                    torch.nn.functional.normalize(rgb_feats.detach(), p=2, dim=1), target
                                ).mean()
                                mcd_ni = iadd_plugin.compute_mcd_vectorized(
                                    torch.nn.functional.normalize(ni_feats.detach(), p=2, dim=1), target
                                ).mean()
                                mcd_ti = iadd_plugin.compute_mcd_vectorized(
                                    torch.nn.functional.normalize(ti_feats.detach(), p=2, dim=1), target
                                ).mean()
                            mcd_vec = torch.stack([mcd_rgb, mcd_ni, mcd_ti], dim=0)
                            sorted_idx = torch.argsort(mcd_vec)  # asc
                            idx_fuse = sorted_idx[:2]
                            idx_single = sorted_idx[2]

                            mcd_gap = mcd_vec.max() - mcd_vec.min()
                            mcd_gap_threshold = float(getattr(cfg.MODEL.IADD, 'MCD_GAP_THRESHOLD', 0.5))
                            if mcd_gap > mcd_gap_threshold:
                                logger.info(
                                    f"[IADD-FUSION weak2_mcd] MCD gap {mcd_gap:.4f} > threshold {mcd_gap_threshold:.4f}, fallback to mean fusion")
                                ir_logits = (ni_logits + ti_logits) / 2.0
                                ir_feats = (ni_feats + ti_feats) / 2.0
                                m1_logits, m2_logits = rgb_logits, ir_logits
                                m1_feats, m2_feats = rgb_feats, ir_feats
                            else:
                                feats_list = [rgb_feats, ni_feats, ti_feats]
                                logits_list = [rgb_logits, ni_logits, ti_logits]
                                name_list = ['RGB', 'NI', 'TI']

                                f1, f2 = feats_list[int(idx_fuse[0])], feats_list[int(idx_fuse[1])]
                                l1, l2 = logits_list[int(idx_fuse[0])], logits_list[int(idx_fuse[1])]
                                m1, m2 = mcd_vec[int(idx_fuse[0])], mcd_vec[int(idx_fuse[1])]
                                name_f1 = name_list[int(idx_fuse[0])]
                                name_f2 = name_list[int(idx_fuse[1])]

                                feat_single = feats_list[int(idx_single)]
                                logit_single = logits_list[int(idx_single)]
                                name_single = name_list[int(idx_single)]

                                tau = float(getattr(cfg.MODEL.IADD, 'FUSION_TAU', 1.0))
                                tau = max(tau, 1e-6)
                                w = torch.softmax(torch.stack([m1 / tau, m2 / tau], dim=0), dim=0)
                                if bool(getattr(cfg.MODEL.IADD, 'FUSION_DETACH', True)):
                                    w = w.detach()

                                fused_feat = w[0] * f1 + w[1] * f2
                                fused_logit = w[0] * l1 + w[1] * l2

                                m1_logits, m2_logits = fused_logit, logit_single
                                m1_feats, m2_feats = fused_feat, feat_single

                                log_every = int(getattr(cfg.MODEL.IADD, 'FUSION_LOG_PERIOD', log_period))
                                if log_every > 0 and ((n_iter + 1) % log_every == 0):
                                    logger.info(
                                        "[IADD-FUSION weak2_mcd] fused=({}+{}) w=({:.3f},{:.3f}) | strong={} | "
                                        "MCD: RGB={:.4f}, NI={:.4f}, TI={:.4f}".format(
                                            name_f1, name_f2, float(w[0].item()), float(w[1].item()), name_single,
                                            float(mcd_rgb.item()), float(mcd_ni.item()), float(mcd_ti.item())
                                        )
                                    )

                        elif fusion_mode == 'teacher2students':
                            t2s_T = float(getattr(cfg.MODEL.IADD, 'T2S_TEMPERATURE', 2.0))
                            instancewise = bool(getattr(cfg.MODEL.IADD, 'T2S_INSTANCEWISE', True))

                            if instancewise:
                                t2s_out = iadd_plugin.teacher2students_instancewise(
                                    logits_rgb=rgb_logits, logits_ni=ni_logits, logits_ti=ti_logits,
                                    feats_rgb=rgb_feats, feats_ni=ni_feats, feats_ti=ti_feats,
                                    labels=target,
                                    temperature=t2s_T,
                                    lambda_distill=cfg.MODEL.IADD.LAMBDA_DISTILL,
                                )

                                loss_total += t2s_out['loss_distill']
                                loss_hybrid = hybrid_triplet_loss(t2s_out['hybrid_dist'], target)
                                loss_total += loss_hybrid * float(cfg.MODEL.IADD.LAMBDA_HYBRID)

                                mcd_rgb_meter.update(t2s_out['mcd_rgb'], 1)
                                mcd_ni_meter.update(t2s_out['mcd_ni'], 1)
                                mcd_ti_meter.update(t2s_out['mcd_ti'], 1)

                                log_every = int(getattr(cfg.MODEL.IADD, 'T2S_LOG_PERIOD', 0))
                                if log_every > 0 and ((n_iter + 1) % log_every == 0):
                                    t_idx = t2s_out['teacher_idx']
                                    c_rgb = int((t_idx == 0).sum().item())
                                    c_ni = int((t_idx == 1).sum().item())
                                    c_ti = int((t_idx == 2).sum().item())
                                    logger.info(
                                        "[IADD-FUSION teacher2students][instance] teacher_count: RGB={} NI={} TI={} | "
                                        "MCD(mean): RGB={:.4f}, NI={:.4f}, TI={:.4f} | w_mean={:.3f}".format(
                                            c_rgb, c_ni, c_ti,
                                            float(t2s_out['mcd_rgb']), float(t2s_out['mcd_ni']), float(t2s_out['mcd_ti']),
                                            float(t2s_out['w_mean'])
                                        )
                                    )
                            else:
                                with torch.no_grad():
                                    mcd_rgb = iadd_plugin.compute_mcd_vectorized(
                                        torch.nn.functional.normalize(rgb_feats.detach(), p=2, dim=1), target
                                    ).mean()
                                    mcd_ni = iadd_plugin.compute_mcd_vectorized(
                                        torch.nn.functional.normalize(ni_feats.detach(), p=2, dim=1), target
                                    ).mean()
                                    mcd_ti = iadd_plugin.compute_mcd_vectorized(
                                        torch.nn.functional.normalize(ti_feats.detach(), p=2, dim=1), target
                                    ).mean()
                                mcd_vec = torch.stack([mcd_rgb, mcd_ni, mcd_ti], dim=0)
                                idx_teacher = int(torch.argmax(mcd_vec).item())
                                idx_students = [i for i in [0, 1, 2] if i != idx_teacher]
                                feats_list = [rgb_feats, ni_feats, ti_feats]
                                logits_list = [rgb_logits, ni_logits, ti_logits]
                                t2s_out = iadd_plugin.teacher2students(
                                    teacher_logits=logits_list[idx_teacher],
                                    students_logits=[logits_list[idx_students[0]], logits_list[idx_students[1]]],
                                    teacher_feats=feats_list[idx_teacher],
                                    students_feats=[feats_list[idx_students[0]], feats_list[idx_students[1]]],
                                    labels=target,
                                    temperature=t2s_T,
                                    lambda_distill=cfg.MODEL.IADD.LAMBDA_DISTILL,
                                    lambda_hybrid=cfg.MODEL.IADD.LAMBDA_HYBRID,
                                )
                                loss_total += t2s_out['loss_distill']
                                loss_hybrid = hybrid_triplet_loss(t2s_out['hybrid_dist'], target)
                                loss_total += loss_hybrid * float(cfg.MODEL.IADD.LAMBDA_HYBRID)

                                mcd_rgb_meter.update(t2s_out['mcd_teacher'], 1)
                                mcd_ir_meter.update((t2s_out['mcd_student1'] + t2s_out['mcd_student2']) / 2.0, 1)

                        else:
                            logger.warning(f"Unknown IADD FUSION_MODE={fusion_mode}, fallback to mean")
                            ir_logits = (ni_logits + ti_logits) / 2.0
                            ir_feats = (ni_feats + ti_feats) / 2.0
                            m1_logits, m2_logits = rgb_logits, ir_logits
                            m1_feats, m2_feats = rgb_feats, ir_feats

                        if fusion_mode != 'teacher2students':
                            iadd_out = iadd_plugin(
                                m1_logits, m2_logits,
                                m1_feats, m2_feats,
                                target
                            )
                            loss_total += iadd_out['loss_distill']
                            loss_hybrid = hybrid_triplet_loss(iadd_out['hybrid_dist'], target)
                            loss_total += loss_hybrid * float(cfg.MODEL.IADD.LAMBDA_HYBRID)
                            mcd_rgb_meter.update(iadd_out['mcd_m1'], 1)
                            mcd_ir_meter.update(iadd_out['mcd_m2'], 1)

                else:
                    # 兼容旧代码逻辑
                    if len(output) % 2 == 1:
                        loss_total += output[-1]
                        score_final = output[0]
                    else:
                        score_final = output[0]

                    if isinstance(output, tuple) or isinstance(output, list):
                        for i in range(0, len(output) // 2 * 2, 2):
                            loss_total += loss_fn(score=output[i], feat=output[i + 1], target=target,
                                                  target_cam=target_cam)

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (score_final.max(1)[1] == target).float().mean()
            acc_meter.update(acc, 1)
            loss_meter.update(loss_total.item(), img['RGB'].shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if iadd_active:
                    if getattr(cfg.MODEL.IADD, 'FUSION_MODE', 'mean') == 'teacher2students' and bool(getattr(cfg.MODEL.IADD, 'T2S_INSTANCEWISE', True)):
                        logger.info(
                            "Epoch[{}] Iter[{}/{}] Loss: {:.3f} Acc: {:.3f} MCD_RGB: {:.3f} MCD_NI: {:.3f} MCD_TI: {:.3f} lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg,
                                    mcd_rgb_meter.avg, mcd_ni_meter.avg, mcd_ti_meter.avg,
                                    scheduler._get_lr(epoch)[0]))
                    else:
                        logger.info(
                            "Epoch[{}] Iter[{}/{}] Loss: {:.3f} Acc: {:.3f} MCD_RGB: {:.3f} MCD_IR: {:.3f} lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg,
                                    mcd_rgb_meter.avg, mcd_ir_meter.avg,
                                    scheduler._get_lr(epoch)[0]))
                else:
                    # warmup 阶段：不显示 MCD
                    logger.info(
                        "Epoch[{}] Iter[{}/{}] Loss: {:.3f} Acc: {:.3f} lr: {:.2e}"
                        .format(epoch, (n_iter + 1), len(train_loader),
                                loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Speed: {:.1f} samples/s".format(epoch, train_loader.batch_size / time_per_batch))

        if (not no_save) and (epoch % checkpoint_period == 0):
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger)
            else:
                mAP_ori, cmc_ori = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,
                                                     return_pattern=1)

                if (not bool(getattr(cfg.MODEL, 'HDM', False))) and (not bool(getattr(cfg.MODEL, 'ATM', False))):
                    if mAP_ori >= best_index['ori_mAP']:
                        best_index['ori_mAP'] = mAP_ori
                        best_index['ori_Rank-1'] = cmc_ori[0]
                        best_index['ori_Rank-5'] = cmc_ori[4]
                        best_index['ori_Rank-10'] = cmc_ori[9]
                    if mAP_ori >= best_index['mAP']:
                        best_index['mAP'] = mAP_ori
                        best_index['Rank-1'] = cmc_ori[0]
                        best_index['Rank-5'] = cmc_ori[4]
                        best_index['Rank-10'] = cmc_ori[9]
                        if not no_save:
                            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))
                    logger.info("~" * 50)
                    logger.info("[Best Ori] mAP: {:.1%} | R1: {:.1%} R5: {:.1%} R10: {:.1%}".format(
                        best_index['ori_mAP'], best_index['ori_Rank-1'], best_index['ori_Rank-5'], best_index['ori_Rank-10']))
                    logger.info("~" * 50)
                else:
                    mAP_moe, cmc_moe = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,
                                                         return_pattern=2)
                    mAP_fused, cmc_fused = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,
                                                             return_pattern=3)

                    if mAP_ori >= best_index['ori_mAP']:
                        best_index['ori_mAP'] = mAP_ori
                        best_index['ori_Rank-1'] = cmc_ori[0]
                        best_index['ori_Rank-5'] = cmc_ori[4]
                        best_index['ori_Rank-10'] = cmc_ori[9]

                    if mAP_moe >= best_index['moe_mAP']:
                        best_index['moe_mAP'] = mAP_moe
                        best_index['moe_Rank-1'] = cmc_moe[0]
                        best_index['moe_Rank-5'] = cmc_moe[4]
                        best_index['moe_Rank-10'] = cmc_moe[9]

                    if mAP_fused >= best_index['mAP']:
                        best_index['mAP'] = mAP_fused
                        best_index['Rank-1'] = cmc_fused[0]
                        best_index['Rank-5'] = cmc_fused[4]
                        best_index['Rank-10'] = cmc_fused[9]
                        if not no_save:
                            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))

                    logger.info("~" * 50)
                    logger.info("[Best Fused] mAP: {:.1%} | R1: {:.1%} R5: {:.1%} R10: {:.1%}".format(
                        best_index['mAP'], best_index['Rank-1'], best_index['Rank-5'], best_index['Rank-10']))
                    logger.info("[Best Ori  ] mAP: {:.1%} | R1: {:.1%} R5: {:.1%} R10: {:.1%}".format(
                        best_index['ori_mAP'], best_index['ori_Rank-1'], best_index['ori_Rank-5'], best_index['ori_Rank-10']))
                    logger.info("[Best Moe  ] mAP: {:.1%} | R1: {:.1%} R5: {:.1%} R10: {:.1%}".format(
                        best_index['moe_mAP'], best_index['moe_Rank-1'], best_index['moe_Rank-5'], best_index['moe_Rank-10']))
                    logger.info("~" * 50)

    return best_index


def training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern=1):
    # 复用 processor.py 中的实现
    from engine.processor import training_neat_eval as eval_func
    return eval_func(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern)
