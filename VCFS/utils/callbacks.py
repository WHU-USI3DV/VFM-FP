import os
import sys
import shutil
import json
from pathlib import Path

import matplotlib
import numpy as np
import scipy.signal
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .utils import cvtColor, preprocess_input, resize_img
from .utils_metrics import compute_mIoU, mean_metric


TRUE_VALUES = {"1", "true", "yes", "on"}


class LossHistory:
    def __init__(self, log_dir, model, input_shape):
        self.log_dir = Path(log_dir)
        self.losses = []
        self.val_loss = []

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        if os.environ.get("VCFS_ADD_TENSORBOARD_GRAPH", "0").lower() in TRUE_VALUES:
            self._try_add_graph(model, input_shape)

    def _try_add_graph(self, model, input_shape):
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except Exception as exc:
            print(f"TensorBoard graph export skipped: {exc}")

    def append_loss(self, epoch, loss, val_loss):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        self._append_scalar_file("epoch_loss.txt", loss)
        self._append_scalar_file("epoch_val_loss.txt", val_loss)
        self.writer.add_scalar("loss", loss, epoch)
        self.writer.add_scalar("val_loss", val_loss, epoch)
        self.loss_plot()

    def _append_scalar_file(self, filename, value):
        with (self.log_dir / filename).open("a", encoding="utf-8") as file:
            file.write(f"{value}\n")

    def loss_plot(self):
        epochs = range(len(self.losses))
        plt.figure()
        plt.plot(epochs, self.losses, "red", linewidth=2, label="train loss")
        plt.plot(epochs, self.val_loss, "coral", linewidth=2, label="val loss")

        smooth_window = 5 if len(self.losses) < 25 else 15
        try:
            plt.plot(
                epochs,
                scipy.signal.savgol_filter(self.losses, smooth_window, 3),
                "green",
                linestyle="--",
                linewidth=2,
                label="smooth train loss",
            )
            plt.plot(
                epochs,
                scipy.signal.savgol_filter(self.val_loss, smooth_window, 3),
                "#8B4513",
                linestyle="--",
                linewidth=2,
                label="smooth val loss",
            )
        except Exception:
            pass

        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(self.log_dir / "epoch_loss.png")
        plt.close("all")


class EvalCallback:
    def __init__(
        self,
        net,
        input_shape,
        num_classes,
        image_ids,
        dataset_path,
        log_dir,
        cuda,
        miou_out_path=None,
        eval_flag=True,
        period=1,
        ignore_class_ids=None,
    ):
        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.dataset_path = Path(dataset_path)
        self.log_dir = Path(log_dir)
        self.cuda = cuda
        self.miou_out_path = Path(miou_out_path) if miou_out_path else self.log_dir / "miou_tmp"
        self.eval_flag = eval_flag
        self.period = period
        self.ignore_class_ids = list(ignore_class_ids or [])
        self.mious = [0]
        self.epoches = [0]

        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "evaluation_config.json").write_text(
            json.dumps(
                {
                    "metric": "mean_iou",
                    "num_classes": self.num_classes,
                    "ignore_class_ids": self.ignore_class_ids,
                    "includes_background": 0 not in self.ignore_class_ids,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if self.eval_flag:
            self._append_miou(0)

    def _append_miou(self, value):
        with (self.log_dir / "epoch_miou.txt").open("a", encoding="utf-8") as file:
            file.write(f"{value}\n")

    def get_miou_png(self, image, label):
        image = cvtColor(image)
        image_data, label = resize_img(image, (self.input_shape[1], self.input_shape[0]), label)
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)),
            0,
        )

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            logits = self.net(images)[0]
            probs = F.softmax(logits.permute(1, 2, 0), dim=-1).cpu().numpy()
            pred = probs.argmax(axis=-1)

        return Image.fromarray(np.uint8(pred)), label

    def on_epoch_end(self, epoch, model_eval):
        if not self.eval_flag or epoch % self.period != 0:
            return

        self.net = model_eval
        pred_dir = self.miou_out_path / "detection-results"
        gt_crop_dir = self.miou_out_path / "Ground_True_Crop"
        pred_dir.mkdir(parents=True, exist_ok=True)
        gt_crop_dir.mkdir(parents=True, exist_ok=True)

        print("Get miou.")
        for image_id in tqdm(self.image_ids, disable=not sys.stderr.isatty()):
            image = Image.open(self.dataset_path / "JPEGImages" / f"{image_id}.jpg")
            gt = Image.open(self.dataset_path / "SegmentationClass" / f"{image_id}.png")
            pred, gt = self.get_miou_png(image, gt)
            pred.save(pred_dir / f"{image_id}.png")
            gt.save(gt_crop_dir / f"{image_id}.png")

        print("Calculate miou.")
        _, ious, _, _ = compute_mIoU(
            str(gt_crop_dir),
            str(pred_dir),
            self.image_ids,
            self.num_classes,
            None,
            ignore_class_ids=self.ignore_class_ids,
        )
        miou = mean_metric(ious, self.ignore_class_ids) * 100
        self.mious.append(miou)
        self.epoches.append(epoch)
        self._append_miou(miou)
        self._plot_miou()

        print("Get miou done.")
        shutil.rmtree(self.miou_out_path, ignore_errors=True)

    def _plot_miou(self):
        plt.figure()
        plt.plot(self.epoches, self.mious, "red", linewidth=2, label="train miou")
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Miou")
        plt.title("A Miou Curve")
        plt.legend(loc="upper right")
        plt.savefig(self.log_dir / "epoch_miou.png")
        plt.close("all")
