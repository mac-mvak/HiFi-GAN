import argparse
import collections
import warnings

import numpy as np
import torch

import hw_hifi.loss as module_loss
import hw_hifi.metric as module_metric
import hw_hifi.model as module_arch
from hw_hifi.trainer import Trainer
from hw_hifi.utils import prepare_device
from hw_hifi.utils.object_loading import get_dataloaders
from hw_hifi.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 1233
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")


    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    generator = config.init_obj(config["generator_model"], module_arch)
    discriminator = config.init_obj(config["discriminator_model"], module_arch)
    logger.info(generator)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    generator = generator.to(device)
    discriminator = discriminator.to(device)


    # get function handles of loss and metrics
    loss_generator = config.init_obj(config["loss_generator"], module_loss).to(device)
    loss_discriminator = config.init_obj(config["loss_discriminator"], module_loss).to(device)
    metrics = [
        config.init_obj(metric_dict, module_metric)
        for metric_dict in config["metrics"]
    ]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params_generator = filter(lambda p: p.requires_grad, generator.parameters())
    trainable_params_discriminator = filter(lambda p: p.requires_grad, discriminator.parameters())
    optimizer_generator = config.init_obj(config["optimizer_generator"], torch.optim, trainable_params_generator)
    optimizer_discriminator = config.init_obj(config["optimizer_discriminator"], torch.optim, trainable_params_discriminator)
    lr_scheduler_discriminator = config.init_obj(config["lr_scheduler_discriminator"], torch.optim.lr_scheduler, optimizer_discriminator)
    lr_scheduler_generator = config.init_obj(config["lr_scheduler_generator"], torch.optim.lr_scheduler, optimizer_generator)

    trainer = Trainer(
        gen=generator,
        dis=discriminator,
        loss_gen=loss_generator,
        loss_dis=loss_discriminator,
        metrics=metrics,
        optimizer_gen=optimizer_generator,
        optimizer_dis=optimizer_discriminator,
        config=config,
        device=device,
        dataloaders=dataloaders,
        lr_scheduler_gen=lr_scheduler_generator,
        lr_scheduler_dis=lr_scheduler_discriminator,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="/home/mac-mvak/code_disk/HiFi-GAN/final_data/config_gan_v1.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
