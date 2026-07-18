"""Segmentation metrics used by VCFS training and evaluation."""

import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(
        inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),
        -1,
    )
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta**2) * tp + smooth) / (
        (1 + beta**2) * tp + beta**2 * fn + fp + smooth
    )
    return torch.mean(score)


def fast_hist(a, b, n):
    """Build a confusion matrix from flattened label and prediction arrays."""
    valid = (a >= 0) & (a < n)
    return np.bincount(n * a[valid].astype(int) + b[valid], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def mean_metric(values, ignore_class_ids=None):
    ignored = set(ignore_class_ids or [])
    selected = [value for index, value in enumerate(values) if index not in ignored]
    if not selected:
        raise ValueError("All metric classes were ignored")
    return np.nanmean(selected)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None, ignore_class_ids=None):
    print("Num classes", num_classes)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    for index, (gt_path, pred_path) in enumerate(zip(gt_imgs, pred_imgs)):
        pred = np.array(Image.open(pred_path))
        label = np.array(Image.open(gt_path))

        if len(label.flatten()) != len(pred.flatten()):
            print(
                "Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}".format(
                    len(label.flatten()),
                    len(pred.flatten()),
                    gt_path,
                    pred_path,
                )
            )
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if name_classes is not None and index > 0 and index % 10 == 0:
            print(
                "{:d} / {:d}: mIou-{:0.4f}%; mPA-{:0.4f}%; Accuracy-{:0.4f}%".format(
                    index,
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist),
                )
            )

    ious = per_class_iu(hist)
    pa_recall = per_class_PA_Recall(hist)
    precision = per_class_Precision(hist)

    if name_classes is not None:
        for class_index in range(num_classes):
            print(
                "===>{}:\tIou-{}; Recall (equal to the PA)-{}; Precision-{}".format(
                    name_classes[class_index],
                    round(ious[class_index] * 100, 2),
                    round(pa_recall[class_index] * 100, 2),
                    round(precision[class_index] * 100, 2),
                )
            )

    print(
        "===> mIoU: {}; mPA: {}; Accuracy: {}".format(
            round(mean_metric(ious, ignore_class_ids) * 100, 2),
            round(mean_metric(pa_recall, ignore_class_ids) * 100, 2),
            round(per_Accuracy(hist) * 100, 2),
        )
    )
    return np.array(hist, int), ious, pa_recall, precision


def adjust_axes(renderer, text, fig, axes):
    bbox = text.get_window_extent(renderer=renderer)
    text_width_inches = bbox.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    proportion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * proportion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color="royalblue")
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    renderer = fig.canvas.get_renderer()
    for index, value in enumerate(values):
        label = " " + str(value)
        if value < 1.0:
            label = " {0:.4f}".format(value)
        text = plt.text(value, index, label, color="royalblue", va="center", fontweight="bold")
        if index == len(values) - 1:
            adjust_axes(renderer, text, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12, ignore_class_ids=None):
    draw_plot_func(
        IoUs,
        name_classes,
        "mIoU = {0:.4f}%".format(mean_metric(IoUs, ignore_class_ids) * 100),
        "Intersection over Union",
        os.path.join(miou_out_path, "mIoU.png"),
        tick_font_size=tick_font_size,
        plt_show=True,
    )
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(
        PA_Recall,
        name_classes,
        "mPA = {0:.4f}%".format(mean_metric(PA_Recall, ignore_class_ids) * 100),
        "Pixel Accuracy",
        os.path.join(miou_out_path, "mPA.png"),
        tick_font_size=tick_font_size,
        plt_show=False,
    )
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(
        PA_Recall,
        name_classes,
        "mRecall = {0:.4f}%".format(mean_metric(PA_Recall, ignore_class_ids) * 100),
        "Recall",
        os.path.join(miou_out_path, "Recall.png"),
        tick_font_size=tick_font_size,
        plt_show=False,
    )
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(
        Precision,
        name_classes,
        "mPrecision = {0:.4f}%".format(mean_metric(Precision, ignore_class_ids) * 100),
        "Precision",
        os.path.join(miou_out_path, "Precision.png"),
        tick_font_size=tick_font_size,
        plt_show=False,
    )
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        rows = [[" "] + [str(class_name) for class_name in name_classes]]
        for index in range(len(hist)):
            rows.append([name_classes[index]] + [str(value) for value in hist[index]])
        writer.writerows(rows)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
