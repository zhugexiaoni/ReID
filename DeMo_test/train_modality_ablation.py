#!/usr/bin/env python3
"""
多模态缺失实验训练脚本

功能：
1. Warm-up阶段使用所有模态正常训练
2. Warm-up后每轮使用不同模态缺失模式训练和评估
3. 记录各模态对性能的影响
4. 分析主导模态

用法：
python train_modality_ablation.py --config_file configs/RGBNT201/vit_demo.yml --warmup_epochs 10
"""

from utils.logger import setup_logger
from data import make_dataloader
from modeling import make_model
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from layers.make_loss import make_loss
from engine.processor_modality_ablation import do_train_with_modality_ablation
import random
import torch
import numpy as np
import os
import argparse
from config import cfg


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DeMo Modality Ablation Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument(
        "--warmup_epochs", default=10, help="Number of warmup epochs", type=int
    )
    parser.add_argument(
        "--fea_cft", default=0, help="Feature choose to be tested", type=int
    )
    parser.add_argument(
        "opts", help="Modify config options using the command-line",
        default=None, nargs=argparse.REMAINDER
    )
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TEST.FEAT = args.fea_cft
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("DeMo_Modality_Ablation", output_dir, if_train=True)
    logger.info("="*80)
    logger.info("多模态缺失实验")
    logger.info("="*80)
    logger.info(f"保存模型至: {cfg.OUTPUT_DIR}")
    logger.info(f"Warm-up轮数: {args.warmup_epochs}")
    logger.info(args)

    if args.config_file != "":
        logger.info(f"配置文件: {args.config_file}")
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)

    logger.info("运行配置:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 加载数据
    logger.info("加载数据...")
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    logger.info("数据加载完成")

    # 创建模型
    logger.info("创建模型...")
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    if hasattr(model, 'flops'):
        logger.info(str(model))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"参数数量: {n_parameters / 1e6:.2f}M")
        flops = model.flops()
        logger.info(f"FLOPs: {flops / 1e9:.2f}G")
    else:
        logger.info("模型无法计算FLOPs")

    # 创建损失函数
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # 创建优化器
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    # 创建学习率调度器
    scheduler = create_scheduler(cfg, optimizer)

    # 开始训练
    logger.info("="*80)
    logger.info("开始模态消融训练...")
    logger.info("="*80)

    results = do_train_with_modality_ablation(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query,
        args.local_rank,
        warmup_epochs=args.warmup_epochs
    )

    logger.info("="*80)
    logger.info("训练完成!")
    logger.info("="*80)
    logger.info(f"结果已保存至: {cfg.OUTPUT_DIR}")
