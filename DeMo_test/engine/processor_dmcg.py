"""
DMCG Training Processor

Modified training processor for DeMo + DMCG integration
支持DMCG的训练流程，包括warmup机制和DMCG损失计算
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


def do_train_dmcg(cfg,
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
    """
    DMCG训练主循环

    与原始do_train的区别:
    1. 支持字典形式的输出（DMCG启用时）
    2. 添加DMCG损失项（loss_gate, loss_balance）
    3. 添加epoch tracking以支持warmup机制
    """
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("DeMo.train")
    logger.info('start training with DMCG')

    # DMCG配置参数
    use_dmcg = cfg.MODEL.DMCG.ENABLED
    dmcg_warmup_epochs = cfg.MODEL.DMCG.WARMUP_EPOCHS if use_dmcg else 0
    lambda_gate = cfg.MODEL.DMCG.LAMBDA_GATE if use_dmcg else 0.0
    lambda_balance = cfg.MODEL.DMCG.LAMBDA_BALANCE if use_dmcg else 0.0

    if use_dmcg:
        logger.info(f"DMCG Enabled: warmup={dmcg_warmup_epochs}, λ_gate={lambda_gate}, λ_balance={lambda_balance}")

    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    loss_gate_meter = AverageMeter()
    loss_balance_meter = AverageMeter()
    acc_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    test_sign = cfg.MODEL.HDM or cfg.MODEL.ATM

    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_gate_meter.reset()
        loss_balance_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()

        # 设置当前epoch（用于DMCG warmup控制）
        if hasattr(model, 'module'):
            model.module.set_epoch(epoch)
        else:
            model.set_epoch(epoch)

        dmcg_active = use_dmcg and epoch > dmcg_warmup_epochs
        if dmcg_active and epoch == dmcg_warmup_epochs + 1:
            logger.info("=" * 50)
            logger.info(f"DMCG activation starts from epoch {epoch}")
            logger.info("=" * 50)

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

                # 处理输出：支持原始tuple输出和DMCG的dict输出
                if isinstance(output, dict):
                    # DMCG启用时的字典输出
                    loss = compute_dmcg_loss(
                        output, loss_fn, target, target_cam,
                        lambda_gate, lambda_balance, cfg
                    )
                    loss_total = loss['total']
                    loss_gate_val = loss.get('gate', 0.0)
                    loss_balance_val = loss.get('balance', 0.0)
                else:
                    # 原始输出处理（与processor.py相同）
                    loss_total = 0
                    if len(output) % 2 == 1:
                        index = len(output) - 1
                        for i in range(0, index, 2):
                            loss_tmp = loss_fn(score=output[i], feat=output[i + 1],
                                             target=target, target_cam=target_cam)
                            loss_total = loss_total + loss_tmp
                        loss_total = loss_total + output[-1]
                    else:
                        for i in range(0, len(output), 2):
                            loss_tmp = loss_fn(score=output[i], feat=output[i + 1],
                                             target=target, target_cam=target_cam)
                            loss_total = loss_total + loss_tmp
                    loss_gate_val = 0.0
                    loss_balance_val = 0.0

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            # 计算准确率
            if isinstance(output, dict):
                if 'dmcg_score' in output and dmcg_active:
                    acc = (output['dmcg_score'].max(1)[1] == target).float().mean()
                elif 'ori_score' in output:
                    acc = (output['ori_score'].max(1)[1] == target).float().mean()
                else:
                    acc = (output['moe_score'].max(1)[1] == target).float().mean()
            elif isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss_total.item(), img['RGB'].shape[0])
            if dmcg_active:
                loss_gate_meter.update(loss_gate_val if isinstance(loss_gate_val, float)
                                      else loss_gate_val.item(), img['RGB'].shape[0])
                loss_balance_meter.update(loss_balance_val if isinstance(loss_balance_val, float)
                                        else loss_balance_val.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                if dmcg_active:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (Gate: {:.3f}, Balance: {:.3f}), "
                              "Acc: {:.3f}, Base Lr: {:.2e}"
                              .format(epoch, (n_iter + 1), len(train_loader),
                                     loss_meter.avg, loss_gate_meter.avg, loss_balance_meter.avg,
                                     acc_meter.avg, scheduler._get_lr(epoch)[0]))
                else:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                              .format(epoch, (n_iter + 1), len(train_loader),
                                     loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                       .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
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
                if test_sign:
                    _, _ = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,
                                            return_pattern=1)
                    _, _ = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,
                                            return_pattern=2)
                mAP, cmc = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger,
                                            return_pattern=3)
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(),
                             os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))
                logger.info("~" * 50)
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                logger.info("~" * 50)


def compute_dmcg_loss(output_dict, loss_fn, target, target_cam, lambda_gate, lambda_balance, cfg):
    """
    计算DMCG的综合损失

    Args:
        output_dict: 模型输出字典，包含各种score和feature
        loss_fn: 损失函数
        target: 目标标签
        target_cam: 相机标签
        lambda_gate: 门控正则化损失权重
        lambda_balance: 平衡促进损失权重
        cfg: 配置对象

    Returns:
        loss_dict: 包含total, gate, balance的损失字典
    """
    loss_total = 0
    loss_gate = 0
    loss_balance = 0

    # 1. 原始DeMo损失
    if cfg.MODEL.DIRECT:
        # Direct模式
        if 'moe_score' in output_dict and 'moe_feat' in output_dict:
            loss_moe = loss_fn(score=output_dict['moe_score'],
                             feat=output_dict['moe_feat'],
                             target=target, target_cam=target_cam)
            loss_total += loss_moe

        if 'ori_score' in output_dict and 'ori_feat' in output_dict:
            loss_ori = loss_fn(score=output_dict['ori_score'],
                             feat=output_dict['ori_feat'],
                             target=target, target_cam=target_cam)
            loss_total += loss_ori
    else:
        # Separate模式
        if 'moe_score' in output_dict and 'moe_feat' in output_dict:
            loss_moe = loss_fn(score=output_dict['moe_score'],
                             feat=output_dict['moe_feat'],
                             target=target, target_cam=target_cam)
            loss_total += loss_moe

        for mod in ['RGB', 'NI', 'TI']:
            score_key = f'{mod}_ori_score'
            feat_key = f'{mod}_global'
            if score_key in output_dict and feat_key in output_dict:
                loss_mod = loss_fn(score=output_dict[score_key],
                                 feat=output_dict[feat_key],
                                 target=target, target_cam=target_cam)
                loss_total += loss_mod

    # 2. MoE损失（如果有）
    if 'loss_moe' in output_dict:
        loss_total += output_dict['loss_moe']

    # 3. DMCG损失
    if 'dmcg_score' in output_dict and 'dmcg_feat' in output_dict:
        loss_dmcg = loss_fn(score=output_dict['dmcg_score'],
                          feat=output_dict['dmcg_feat'],
                          target=target, target_cam=target_cam)
        loss_total += loss_dmcg

    # 4. 门控正则化损失
    if 'loss_gate' in output_dict:
        loss_gate = output_dict['loss_gate']
        loss_total += lambda_gate * loss_gate

    # 5. 平衡促进损失
    if 'loss_balance' in output_dict:
        loss_balance = output_dict['loss_balance']
        loss_total += lambda_balance * loss_balance

    return {
        'total': loss_total,
        'gate': loss_gate if isinstance(loss_gate, torch.Tensor) else torch.tensor(loss_gate),
        'balance': loss_balance if isinstance(loss_balance, torch.Tensor) else torch.tensor(loss_balance)
    }


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query, return_pattern=1):
    """
    推理函数（与原始processor.py相同）
    """
    device = "cuda"
    logger = logging.getLogger("DeMo.test")
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    logger.info("~" * 50)
    if return_pattern == 1:
        logger.info("Current is the ori feature testing!")
    elif return_pattern == 2:
        logger.info("Current is the moe feature testing!")
    else:
        logger.info("Current is the [moe,ori] feature testing!")
    logger.info("~" * 50)
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            print(imgpath)
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view, return_pattern=return_pattern, img_path=imgpath)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def training_neat_eval(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator, epoch, logger, return_pattern=1):
    """
    训练中评估函数（与原始processor.py相同）
    """
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
