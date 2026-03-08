import os
import sys
import argparse
import torch

# 添加项目根目录到 path
sys.path.append(os.path.join(os.getcwd(), 'DeMo_test'))

from config import cfg
from engine.processor_emdai import do_train_emdai
from modeling import make_model
from utils.logger import setup_logger
from solver.make_optimizer import make_optimizer
from solver.lr_scheduler import WarmupMultiStepLR
from layers.make_loss import make_loss
from data import make_dataloader
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(cfg):
    # 准备环境
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("DeMo.train", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    set_seed(cfg.SOLVER.SEED)

    # 1. 构建 DataLoader
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 2. 构建模型
    # 注意：我们的 make_model 已经被修改为在 DIRECT=1 时也输出独立 logits
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # 3. 构建 Loss 和 优化器
    loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                  cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    # 4. 开始训练 (使用 E-MDAI Processor)
    # 这将调用包含正交梯度干预的训练循环
    do_train_emdai(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        local_rank
    )

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training with E-MDAI Intervention")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # 确保 DIRECT=1 模式 (这是 E-MDAI 典型使用场景)
    # cfg.MODEL.DIRECT = True
    
    cfg.freeze()

    train(cfg)

if __name__ == '__main__':
    main()
