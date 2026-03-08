import logging
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda import amp

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP


def do_train_cmiei(cfg,
                      model,
                      center_criterion,
                      train_loader,
                      val_loader,
                      optimizer,
                      optimizer_center,
                      scheduler,
                      loss_fn,
                      num_query, local_rank):
    """Copy of engine/processor.py training loop with extra C-MIEI logging.

    IMPORTANT:
    - Training logic is kept identical to original.
    - We do not change model outputs; model remains tuple/list as original.
    - C-MIEI stats are read from model.module._cmiei_stats or model._cmiei_stats if present.
    """

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("DeMo.train")
    logger.info('start training (C-MIEI)')

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    # C-MIEI meters
    cmiei_ratio_meter = AverageMeter()  # sample-level: ratio of intervened samples (avg over logs)
    cmiei_ci_r_meter = AverageMeter()
    cmiei_ci_n_meter = AverageMeter()
    cmiei_ci_t_meter = AverageMeter()
    cmiei_hist_r_meter = AverageMeter()
    cmiei_hist_n_meter = AverageMeter()
    cmiei_hist_t_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    scaler = amp.GradScaler()
    test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM

    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}

    for epoch in range(1, epochs + 1):

        # set epoch for C-MIEI warmup (if plugin exists)
        m = model.module if hasattr(model, 'module') else model
        if hasattr(m, '_cmiei') and getattr(m, '_cmiei') is not None:
            try:
                m._cmiei.set_epoch(epoch)
            except Exception:
                pass

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        cmiei_ratio_meter.reset()
        cmiei_ci_r_meter.reset()
        cmiei_ci_n_meter.reset()
        cmiei_ci_t_meter.reset()
        cmiei_hist_r_meter.reset()
        cmiei_hist_n_meter.reset()
        cmiei_hist_t_meter.reset()

        scheduler.step(epoch)
        model.train()

        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img = {'RGB': img['RGB'].to(device), 'NI': img['NI'].to(device), 'TI': img['TI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)

            with amp.autocast(enabled=True):
                output = model(img, label=target, cam_label=target_cam, view_label=target_view)
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

            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            # fetch cmiei stats if available
            stats = None
            m = model.module if hasattr(model, 'module') else model
            if hasattr(m, '_cmiei_stats'):
                stats = getattr(m, '_cmiei_stats')

            if isinstance(stats, dict):
                # new sample-level stats keys
                cmiei_ratio_meter.update(float(stats.get('cmiei_ratio', 0.0)), 1)
                cmiei_ci_r_meter.update(float(stats.get('cmiei_ted_r', 0.0)), 1)
                cmiei_ci_n_meter.update(float(stats.get('cmiei_ted_n', 0.0)), 1)
                cmiei_ci_t_meter.update(float(stats.get('cmiei_ted_t', 0.0)), 1)
                cmiei_hist_r_meter.update(float(stats.get('cmiei_hist_r', 0.0)), 1)
                cmiei_hist_n_meter.update(float(stats.get('cmiei_hist_n', 0.0)), 1)
                cmiei_hist_t_meter.update(float(stats.get('cmiei_hist_t', 0.0)), 1)
            else:
                cmiei_ratio_meter.update(0.0, 1)
                cmiei_ci_r_meter.update(0.0, 1)
                cmiei_ci_n_meter.update(0.0, 1)
                cmiei_ci_t_meter.update(0.0, 1)
                cmiei_hist_r_meter.update(0.0, 1)
                cmiei_hist_n_meter.update(0.0, 1)
                cmiei_hist_t_meter.update(0.0, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, "
                    "C_MIEI_Ratio: {:.3f}, TED_r: {:.4f}, TED_n: {:.4f}, TED_t: {:.4f}, "
                    "Hist[r,n,t]=[{:.2f},{:.2f},{:.2f}], Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, acc_meter.avg,
                            cmiei_ratio_meter.avg,
                            cmiei_ci_r_meter.avg, cmiei_ci_n_meter.avg, cmiei_ci_t_meter.avg,
                            cmiei_hist_r_meter.avg, cmiei_hist_n_meter.avg, cmiei_hist_t_meter.avg,
                            scheduler._get_lr(epoch)[0])
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if not cfg.MODEL.DIST_TRAIN:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                epoch, time_per_batch, train_loader.batch_size / time_per_batch))

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


# reuse original eval
from engine.processor import training_neat_eval
