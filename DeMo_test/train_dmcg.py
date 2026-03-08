"""
DMCG Training Script

DeMo + DMCG模型的训练脚本
使用方法:
    python train_dmcg.py --config_file configs/RGBNT201/DeMo_DMCG.yml

消融实验:
    # 仅使用基础DeMo（baseline）
    python train_dmcg.py --config_file configs/RGBNT201/DeMo.yml

    # 使用DMCG（完整方法）
    python train_dmcg.py --config_file configs/RGBNT201/DeMo_DMCG.yml

    # 调整DMCG超参数
    python train_dmcg.py --config_file configs/RGBNT201/DeMo_DMCG.yml \
        MODEL.DMCG.LAMBDA_GATE 0.2 MODEL.DMCG.LAMBDA_BALANCE 0.1
"""

from utils.logger import setup_logger
from data import make_dataloader
from modeling.make_model_dmcg import make_model_dmcg  # 使用DMCG版本的make_model
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from layers.make_loss import make_loss
from engine.processor_dmcg import do_train_dmcg  # 使用DMCG版本的processor
import random
import torch
import numpy as np
import os
import argparse
from config import cfg


def set_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 必须为False以确保完全可重复性


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="DeMo + DMCG Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("--fea_cft", default=0, help="Feature choose to be tested", type=int)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    # 加载配置
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

    logger = setup_logger("DeMo", output_dir, if_train=True)
    logger.info("=" * 80)
    logger.info("DeMo + DMCG Training")
    logger.info("=" * 80)
    logger.info("Saving model in the path: {}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # DMCG配置信息
    if hasattr(cfg.MODEL, 'DMCG') and cfg.MODEL.DMCG.ENABLED:
        logger.info("=" * 80)
        logger.info("DMCG Configuration:")
        logger.info("  Enabled: {}".format(cfg.MODEL.DMCG.ENABLED))
        logger.info("  Warmup Epochs: {}".format(cfg.MODEL.DMCG.WARMUP_EPOCHS))
        logger.info("  Hidden Dimension: {}".format(cfg.MODEL.DMCG.HIDDEN_DIM))
        logger.info("  Lambda Gate: {}".format(cfg.MODEL.DMCG.LAMBDA_GATE))
        logger.info("  Lambda Balance: {}".format(cfg.MODEL.DMCG.LAMBDA_BALANCE))
        logger.info("  MIEI Beta: {}".format(cfg.MODEL.DMCG.BETA))
        logger.info("  MIEI Alpha: {}".format(cfg.MODEL.DMCG.ALPHA))
        logger.info("=" * 80)
    else:
        logger.info("DMCG is not enabled, training with baseline DeMo")

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 数据加载
    logger.info("Loading data...")
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    logger.info("Data is ready - {} classes, {} queries".format(num_classes, num_query))

    # 模型构建（使用DMCG版本）
    logger.info("Building model...")
    model = make_model_dmcg(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    if hasattr(model, 'flops'):
        logger.info(str(model))
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of parameters: {n_parameters / 1e6:.2f}M")
        try:
            flops = model.flops()
            logger.info(f"Number of GFLOPs: {flops / 1e9:.2f}")
        except:
            logger.info("Cannot compute FLOPs for this model")
    else:
        logger.info("Model summary not available")

    # 损失函数
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # 优化器
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    # 学习率调度器
    scheduler = create_scheduler(cfg, optimizer)

    # 开始训练（使用DMCG版本的训练函数）
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    do_train_dmcg(
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
        args.local_rank
    )

    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
