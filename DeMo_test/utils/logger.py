import logging
import os
import sys
import os.path as osp


def setup_logger(name, save_dir, if_train, filename=None, mode='w'):
    """Create a logger with both stdout and (optional) file handler.

    Args:
        name: logger name
        save_dir: directory to save log file (if provided)
        if_train: whether this is training log (affects default filename)
        filename: override log filename (e.g. "train_alpha_0.5_beta_0.5.txt")
        mode: file open mode, default 'w'
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicated handlers when calling setup_logger multiple times (e.g. grid search)
    if len(logger.handlers) > 0:
        logger.handlers = []

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        if filename is None:
            filename = "train_log.txt" if if_train else "test_log.txt"
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
