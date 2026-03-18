"""
Training script for IADD Plugin (Instance-Aware Dynamic Distillation)
Based on train_net.py

Added:
- Grid search over IADD loss weights (alpha=LambdaDistill, beta=LambdaHybrid)
- Per-run log file naming: train_alpha_x_beta_y.txt
- Option to disable checkpoint/model saving during grid search to avoid disk usage
- Export summary of each run (best metrics) to CSV/JSON

Example:
  python train_iadd.py --config_file configs/RGBNT201/DeMo_IADD.yml \
    --grid_alpha 0.2,0.35,0.5 --grid_beta 0.5,0.75,1.0 --no_save

Notes:
- This runs sequentially in a single process (recommended for grid search).
- If you use DDP (MODEL.DIST_TRAIN=True), grid search is not recommended here.
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import torch

from config import cfg
from data import make_dataloader
from engine.processor_iadd import do_train_iadd
from layers.make_loss import make_loss
from modeling.make_model_iadd import make_model_iadd as make_model
from solver.make_optimizer import make_optimizer
from solver.scheduler_factory import create_scheduler
from utils.logger import setup_logger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def _parse_float_list(s: str):
    if s is None or s == "":
        return []
    return [float(x.strip()) for x in s.split(',') if x.strip() != ""]


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _run_one(cfg_in, args, alpha: float, beta: float):
    """Run one training job with given alpha/beta.

    Returns:
        dict with fields: alpha, beta, output_dir, best_mAP, best_R1, best_R5, best_R10
    """
    # Clone cfg for this run
    cfg_run = cfg_in.clone()
    cfg_run.defrost()

    # Set alpha/beta
    cfg_run.MODEL.IADD.LAMBDA_DISTILL = float(alpha)
    cfg_run.MODEL.IADD.LAMBDA_HYBRID = float(beta)

    # When grid searching, disable saving to avoid disk usage
    if args.no_save:
        cfg_run.SOLVER.NO_SAVE = True
        cfg_run.SOLVER.CHECKPOINT_PERIOD = 10 ** 9  # effectively never

    # Build a dedicated output/log dir per setting (avoid overwriting logs)
    base_out = cfg_run.OUTPUT_DIR
    tag = f"_alpha_{alpha:g}_beta_{beta:g}"
    out_dir = base_out + tag
    cfg_run.OUTPUT_DIR = out_dir


    cfg_run.freeze()

    set_seed(cfg_run.SOLVER.SEED)

    if cfg_run.MODEL.DIST_TRAIN:
        # Grid search is typically single process; keep behavior but warn
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg_run.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use per-run log file name
    log_filename = f"train{tag}.txt"
    logger = setup_logger("DeMo.train", output_dir, if_train=True, filename=log_filename)
    logger.info("Grid search run: alpha(LAMBDA_DISTILL)=%s beta(LAMBDA_HYBRID)=%s", alpha, beta)
    logger.info("Saving model in the path :{}".format(cfg_run.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg_run))

    if cfg_run.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg_run)

    model = make_model(cfg_run, num_classes, camera_num, view_num)

    loss_func, center_criterion = make_loss(cfg_run, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg_run, model, center_criterion)

    scheduler = create_scheduler(cfg_run, optimizer)

    best_index = do_train_iadd(
        cfg_run,
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

    # Normalize result
    result = {
        'alpha': float(alpha),
        'beta': float(beta),
        'output_dir': output_dir,

        # fused([ori,moe]) main
        'best_mAP': _safe_float(best_index.get('mAP', 0.0)),
        'best_R1': _safe_float(best_index.get('Rank-1', 0.0)),
        'best_R5': _safe_float(best_index.get('Rank-5', 0.0)),
        'best_R10': _safe_float(best_index.get('Rank-10', 0.0)),

        # ori branch
        'ori_best_mAP': _safe_float(best_index.get('ori_mAP', 0.0)),
        'ori_best_R1': _safe_float(best_index.get('ori_Rank-1', 0.0)),
        'ori_best_R5': _safe_float(best_index.get('ori_Rank-5', 0.0)),
        'ori_best_R10': _safe_float(best_index.get('ori_Rank-10', 0.0)),

        # moe branch
        'moe_best_mAP': _safe_float(best_index.get('moe_mAP', 0.0)),
        'moe_best_R1': _safe_float(best_index.get('moe_Rank-1', 0.0)),
        'moe_best_R5': _safe_float(best_index.get('moe_Rank-5', 0.0)),
        'moe_best_R10': _safe_float(best_index.get('moe_Rank-10', 0.0)),

        # whether moe branch exists in this run
        'has_moe': bool(getattr(cfg_run.MODEL, 'HDM', False) or getattr(cfg_run.MODEL, 'ATM', False)),
    }

    # Also dump per-run summary inside its folder
    try:
        with open(os.path.join(output_dir, f"summary{tag}.json"), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Failed to write per-run summary json: %s", e)

    logger.info("[GRID-SUMMARY] %s", result)
    return result


def _write_grid_summary(results, save_dir, filename_prefix='grid_summary'):
    os.makedirs(save_dir, exist_ok=True)

    # Sort by best_mAP desc, then best_R1 desc
    results_sorted = sorted(results, key=lambda r: (r.get('best_mAP', 0.0), r.get('best_R1', 0.0)), reverse=True)

    json_path = os.path.join(save_dir, f"{filename_prefix}.json")
    csv_path = os.path.join(save_dir, f"{filename_prefix}.csv")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    fieldnames = [
        'alpha', 'beta',
        'best_mAP', 'best_R1', 'best_R5', 'best_R10',
        'ori_best_mAP', 'ori_best_R1', 'ori_best_R5', 'ori_best_R10',
        'moe_best_mAP', 'moe_best_R1', 'moe_best_R5', 'moe_best_R10',
        'has_moe',
        'output_dir'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_sorted:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="ReID IADD Training")
    parser.add_argument("--config_file", default="", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)

    # Grid search arguments
    parser.add_argument("--grid_alpha", default="0.01,0.1,0.3,0.5,0.7,1.0", type=str,
                        help="comma-separated list for alpha = MODEL.IADD.LAMBDA_DISTILL")
    parser.add_argument("--grid_beta", default="0.01,0.1,0.3,0.5,0.7,1.0", type=str,
                        help="comma-separated list for beta = MODEL.IADD.LAMBDA_HYBRID")
    # parser.add_argument("--grid_alpha", default=None, type=str,
    #                     help="comma-separated list for alpha = MODEL.IADD.LAMBDA_DISTILL")
    # parser.add_argument("--grid_beta", default=None, type=str,
    #                     help="comma-separated list for beta = MODEL.IADD.LAMBDA_HYBRID")
    parser.add_argument("--no_save", action='store_true',
                        help="disable saving checkpoints/best model (recommended for grid search)")

    # Summary export
    parser.add_argument("--summary_dir", default="", type=str,
                        help="where to save overall grid summary (default: cfg.OUTPUT_DIR)")
    parser.add_argument("--summary_prefix", default="grid_summary", type=str,
                        help="filename prefix for summary .json/.csv")

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    alphas = _parse_float_list(args.grid_alpha)
    betas = _parse_float_list(args.grid_beta)

    if (len(alphas) > 0) or (len(betas) > 0):
        if len(alphas) == 0:
            alphas = [float(cfg.MODEL.IADD.LAMBDA_DISTILL)]
        if len(betas) == 0:
            betas = [float(cfg.MODEL.IADD.LAMBDA_HYBRID)]

        if bool(cfg.MODEL.DIST_TRAIN):
            print("[WARN] MODEL.DIST_TRAIN=True detected. Grid search is recommended in single-process mode.")

        results = []
        for a in alphas:
            for b in betas:
                results.append(_run_one(cfg, args, a, b))

        # Save overall summary
        summary_dir = args.summary_dir if args.summary_dir else cfg.OUTPUT_DIR
        json_path, csv_path = _write_grid_summary(results, summary_dir, filename_prefix=args.summary_prefix)
        print(f"[GRID] Summary saved: {json_path}")
        print(f"[GRID] Summary saved: {csv_path}")

    else:
        # Original single-run behavior

        cfg.freeze()

        set_seed(cfg.SOLVER.SEED)

        if cfg.MODEL.DIST_TRAIN:
            torch.cuda.set_device(args.local_rank)

        output_dir = cfg.OUTPUT_DIR
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger = setup_logger("DeMo.train", output_dir, if_train=True)
        logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
        logger.info(args)

        if args.config_file != "":
            logger.info("Loaded configuration file {}".format(args.config_file))
            with open(args.config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

        if cfg.MODEL.DIST_TRAIN:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

        train_loader, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

        model = make_model(cfg, num_classes, camera_num, view_num)

        loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

        optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

        scheduler = create_scheduler(cfg, optimizer)

        do_train_iadd(
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


if __name__ == '__main__':
    main()
