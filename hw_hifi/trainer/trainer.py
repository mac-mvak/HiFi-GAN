import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from hw_hifi.base import BaseTrainer
from hw_hifi.base.base_text_encoder import BaseTextEncoder
from hw_hifi.logger.utils import plot_spectrogram_to_buf
from hw_hifi.metric.utils import calc_cer, calc_wer
from hw_hifi.model import Generator, Discriminator
from hw_hifi.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            gen: Generator,
            dis: Discriminator,
            loss_gen,
            loss_dis,
            metrics,
            optimizer_gen,
            optimizer_dis,
            config,
            device,
            dataloaders,
            lr_scheduler_gen=None,
            lr_scheduler_dis=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(gen, dis, loss_gen, loss_dis, metrics, 
                         optimizer_gen, optimizer_dis, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler_gen = lr_scheduler_gen
        self.lr_scheduler_dis = lr_scheduler_dis
        self.log_step = 50

        self.loss_names = ["mel_loss", "fm_loss", "gan_loss",  "loss", "discriminator loss"]

        self.train_metrics = MetricTracker(
            *self.loss_names, "grad norm dis", "grad norm gen",
            writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["audios", "mel_spectrogram"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm_gen(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.gen.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
    
    def _clip_grad_norm_dis(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.dis.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.gen.train()
        self.dis.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.gen.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    for p in self.dis.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            #self.train_metrics.update("grad norm dis", self.get_grad_norm_dis())
            #self.train_metrics.update("grad norm gen", self.get_grad_norm_gen())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} G_Loss: {:.6f} D_Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item(), 
                        batch['discriminator loss']
                    )
                )
                lr_epoch_generator = self.lr_scheduler_gen.get_last_lr()[0]
                self.writer.add_scalar(
                   "learning rate generator", lr_epoch_generator
                )
                lr_epoch_discriminator = self.lr_scheduler_dis.get_last_lr()[0]
                self.writer.add_scalar(
                   "learning rate discriminator", lr_epoch_discriminator
                )
                self._log_predictions(**batch)
                #self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        self.lr_scheduler_gen.step()
        self.lr_scheduler_dis.step()

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        gen_out = self.gen(**batch)
        batch.update(gen_out)

        if is_train:
            #optimize discriminator
            self.optimizer_dis.zero_grad()
            discr_out = self.dis(audios=batch['audios'],
                             predicted_audios=batch['predicted_audios'].detach())
        
            discr_loss = self.loss_dis(**discr_out)
            discr_loss.backward()
            batch['discriminator loss'] = discr_loss
            self._clip_grad_norm_dis()
            self.optimizer_dis.step()
            metrics.update("grad norm dis", self.get_grad_norm_dis())

            #optimize generator
            self.optimizer_gen.zero_grad()
            discr_out = self.dis(**batch)
            batch.update(discr_out)
            gen_loss = self.loss_gen(**batch)
            batch.update(gen_loss)
            batch['loss'].backward()
            self._clip_grad_norm_gen()
            self.optimizer_gen.step()
            metrics.update("grad norm gen", self.get_grad_norm_gen())


            for name in self.loss_names:
                metrics.update(name, batch[name].detach().cpu().item())
        return batch



    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            audios,
            predicted_audios,
            examples_to_log=2,
            *args,
            **kwargs,
    ):
        # TODO: implement logging of beam search results
        if self.writer is None:
            return
        rows = {}
        i = 0
        for audio, pred_audio in zip(audios[:examples_to_log], predicted_audios[:examples_to_log]):
            rows[i] = {
                "orig_audio" : self.writer.wandb.Audio(audio.detach().squeeze().cpu().numpy(), sample_rate=22050),
                "pred_audio" : self.writer.wandb.Audio(pred_audio.detach().squeeze().cpu().numpy(), sample_rate=22050),
            }
            i+=1
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm_gen(self, norm_type=2):
        parameters = self.gen.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def get_grad_norm_dis(self, norm_type=2):
        parameters = self.dis.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
