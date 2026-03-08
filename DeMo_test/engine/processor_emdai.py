"""E-MDAI Training Processor

Clean + DeMo-aligned version.

Key principle:
- Do NOT hack features for loss in processor.
- Keep the training loop identical to DeMo's original processor.py.
- E-MDAI is plugged into model forward (global features), not here.

This processor is essentially engine/processor.py + a small compatibility shim:
- If model returns a dict (our EMDAI-enabled DeMo), convert it to the legacy list format
  expected by the loss loop.
- Additionally, we log EMDAI stats (intervention ratio / entropy) if the model provides them.
"""

import logging
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda import amp

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP


def _dict_output_to_legacy_list(output_dict):
    """Convert model dict output into legacy list format.

    Returns:
        list: [score1, feat1, (score2, feat2, ...), (extra_loss)]
    """
    out_list = []
    if 'ori_score' in output_dict and 'ori_feat' in output_dict:
        out_list.extend([output_dict['ori_score'], output_dict['ori_feat']])
    if 'moe_score' in output_dict and 'moe_feat' in output_dict:
        out_list.extend([output_dict['moe_score'], output_dict['moe_feat']])
    if 'loss_moe' in output_dict:
        out_list.append(output_dict['loss_moe'])
    return out_list


def do_train_emdai(cfg,
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
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("DeMo.train")
    logger.info('start training (E-MDAI enabled in model)')

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                find_unused_parameters=True
            )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    interv_meter = AverageMeter()
    ent_meter = AverageMeter()
    # extra diagnostic meters (RGB head health)
    rgb_ent_norm_meter = AverageMeter()
    rgb_logits_std_meter = AverageMeter()
    rgb_logits_abs_mean_meter = AverageMeter()
    rgb_feat_l2_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    scaler = amp.GradScaler()
    test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM

    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        interv_meter.reset()
        ent_meter.reset()
        rgb_ent_norm_meter.reset()
        rgb_logits_std_meter.reset()
        rgb_logits_abs_mean_meter.reset()
        rgb_feat_l2_meter.reset()

        scheduler.step(epoch)
        model.train()

        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = {
                'RGB': img['RGB'].to(device),
                'NI': img['NI'].to(device),
                'TI': img['TI'].to(device)
            }
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                output = model(img, label=target, cam_label=target_cam, view_label=target_view)

                # Extract EMDAI stats (if provided)
                emdai_stats = None
                if isinstance(output, dict):
                    emdai_stats = output.get('emdai_stats', None)
                    output = _dict_output_to_legacy_list(output)

                loss = 0
                if len(output) % 2 == 1:
                    index = len(output) - 1
                    for i in range(0, index, 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
                    loss = loss + output[-1]
                else:
                    for i in range(0, len(output), 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            # acc: output[0] should be score tensor [B, num_classes]
            score_for_acc = output[0]
            acc = (score_for_acc.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            # update emdai meters
            if isinstance(emdai_stats, dict):
                interv_meter.update(float(emdai_stats.get('interv_ratio', 0.0)), 1)
                ent_meter.update(float(emdai_stats.get('avg_entropy', 0.0)), 1)

                # detailed diagnostics (may be absent)
                rgb_ent_norm_meter.update(float(emdai_stats.get('rgb_ent_norm_mean', 0.0)), 1)
                rgb_logits_std_meter.update(float(emdai_stats.get('rgb_logits_std', 0.0)), 1)
                rgb_logits_abs_mean_meter.update(float(emdai_stats.get('rgb_logits_abs_mean', 0.0)), 1)
                rgb_feat_l2_meter.update(float(emdai_stats.get('rgb_feat_l2_mean', 0.0)), 1)
            else:
                interv_meter.update(0.0, 1)
                ent_meter.update(0.0, 1)
                rgb_ent_norm_meter.update(0.0, 1)
                rgb_logits_std_meter.update(0.0, 1)
                rgb_logits_abs_mean_meter.update(0.0, 1)
                rgb_feat_l2_meter.update(0.0, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, EMDAI_Ratio: {:.3f}, EMDAI_Ent: {:.3f}, "
                    "RGB_Hnorm: {:.3f}, RGB_logit_std: {:.3f}, RGB_logit_abs_mean: {:.3f}, RGB_feat_l2: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, acc_meter.avg,
                            interv_meter.avg, ent_meter.avg,
                            rgb_ent_norm_meter.avg,
                            rgb_logits_std_meter.avg,
                            rgb_logits_abs_mean_meter.avg,
                            rgb_feat_l2_meter.avg,
                            scheduler.get_lr()[0])
                )

                # also occasionally dump a full stat dict (if present)
                if isinstance(emdai_stats, dict) and (dist.get_rank() == 0 if cfg.MODEL.DIST_TRAIN else True):
                    # keep it short: only keys relevant to uniform-logits diagnosis
                    dbg = {
                        'thr': emdai_stats.get('threshold', None),
                        'H': emdai_stats.get('rgb_ent_mean', None),
                        'Hn': emdai_stats.get('rgb_ent_norm_mean', None),
                        'logit_mean': emdai_stats.get('rgb_logits_mean', None),
                        'logit_std': emdai_stats.get('rgb_logits_std', None),
                        'row_max': emdai_stats.get('rgb_logits_row_max_mean', None),
                        'row_min': emdai_stats.get('rgb_logits_row_min_mean', None),
                        'feat_l2': emdai_stats.get('rgb_feat_l2_mean', None),
                        'ratio': emdai_stats.get('interv_ratio', None),
                    }
                    logger.info(f"[EMDAI-DIAG] {dbg}")

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if not cfg.MODEL.DIST_TRAIN:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch)
            )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger)
            else:
                if test_sign:
                    _, _ = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern=1)
                    _, _ = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern=2)
                mAP, cmc = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern=3)
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))

                logger.info("~" * 50)
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                logger.info("~" * 50)

    return best_index


def training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern=1):
    from engine.processor import training_neat_eval as eval_func
    return eval_func(cfg, model, val_loader, device, evaluator, epoch, logger, return_pattern)
