from abc import abstractmethod

import torch
from numpy import inf

from hw_hifi.base import BaseModel
from hw_hifi.logger import get_visualizer


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, gen, dis, loss_gen, loss_dis, metrics, optimizer_gen, optimizer_dis, config, device):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.gen = gen
        self.dis = dis
        self.loss_gen = loss_gen
        self.loss_dis = loss_dis
        self.metrics = metrics
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = get_visualizer(
            config, self.logger, cfg_trainer["visualize"]
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            if epoch % self.save_period == 0 :
                self._save_checkpoint(epoch)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch_gen = type(self.gen).__name__
        arch_dis = type(self.dis).__name__
        state = {
            "arch_gen": arch_gen,
            "arch_dis": arch_dis,
            "epoch": epoch,
            "state_dict_gen": self.gen.state_dict(),
            "state_dict_dis": self.dis.state_dict(),
            "optimizer_gen": self.optimizer_gen.state_dict(),
            "optimizer_dis": self.optimizer_dis.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["generator_model"] != self.config["generator_model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.gen.load_state_dict(checkpoint["state_dict_gen"])

        if checkpoint["config"]["discriminator_model"] != self.config["discriminator_model"]:
            self.logger.warning(
                "Warning: Architecture of a discriminator in a"
                "  configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.dis.load_state_dict(checkpoint["state_dict_dis"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["optimizer_generator"] != self.config["optimizer_generator"] or
                checkpoint["config"]["lr_scheduler_generator"] != self.config["lr_scheduler_generator"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler of a generator given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.optimizer_gen.load_state_dict(checkpoint["optimizer_gen"])

        if (
                checkpoint["config"]["optimizer_discriminator"] != self.config["optimizer_discriminator"] or
                checkpoint["config"]["lr_scheduler_discriminator"] != self.config["lr_scheduler_discriminator"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler of a discriminator given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.optimizer_dis.load_state_dict(checkpoint["optimizer_dis"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
