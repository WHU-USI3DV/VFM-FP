import os
import sys

import torch
from nets.deeplabv3_training import CE_Loss, Dice_loss, Focal_Loss, InverFreq_loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


FACADEWHU_TOTAL_CLASS_FREQUENCY = torch.tensor(
    [1617279596, 88907134, 49734707, 471044165, 35841020, 26988849, 75566705]
)


def _to_device(images, masks, labels, weights, cuda, local_rank):
    if not cuda:
        return images, masks, labels, weights
    return (
        images.cuda(local_rank),
        masks.cuda(local_rank),
        labels.cuda(local_rank),
        weights.cuda(local_rank),
    )


def _segmentation_loss(outputs, masks, labels, weights, num_classes, dice_loss, focal_loss, inverfreq_loss):
    if focal_loss:
        loss = Focal_Loss(outputs, masks, weights, num_classes=num_classes)
    elif inverfreq_loss:
        loss = InverFreq_loss(
            outputs,
            masks,
            FACADEWHU_TOTAL_CLASS_FREQUENCY.to(masks.device),
            num_classes=num_classes,
            alpha=0.4,
        )
    else:
        loss = CE_Loss(outputs, masks, weights, num_classes=num_classes)

    if dice_loss:
        loss = loss + Dice_loss(outputs, labels)
    return loss


def _forward_loss_score(model, images, masks, labels, weights, num_classes, dice_loss, focal_loss, inverfreq_loss):
    outputs = model(images)
    loss = _segmentation_loss(
        outputs,
        masks,
        labels,
        weights,
        num_classes,
        dice_loss,
        focal_loss,
        inverfreq_loss,
    )
    with torch.no_grad():
        score = f_score(outputs, labels)
    return loss, score


def fit_one_epoch(
    model_train,
    model,
    loss_history,
    eval_callback,
    optimizer,
    epoch,
    epoch_step,
    epoch_step_val,
    gen,
    gen_val,
    Epoch,
    cuda,
    dice_loss,
    focal_loss,
    inverfreq_loss,
    cls_weights,
    num_classes,
    fp16,
    scaler,
    save_period,
    save_dir,
    local_rank=0,
):
    total_loss = 0
    total_f_score = 0
    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print("Start Train")
        pbar = tqdm(
            total=epoch_step,
            desc=f"Epoch {epoch + 1}/{Epoch}",
            postfix={},
            mininterval=0.3,
            disable=not sys.stderr.isatty(),
        )
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            imgs, pngs, labels, weights = _to_device(imgs, pngs, labels, weights, cuda, local_rank)

        optimizer.zero_grad()
        if not fp16:
            loss, _f_score = _forward_loss_score(
                model_train, imgs, pngs, labels, weights, num_classes, dice_loss, focal_loss, inverfreq_loss
            )
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                loss, _f_score = _forward_loss_score(
                    model_train, imgs, pngs, labels, weights, num_classes, dice_loss, focal_loss, inverfreq_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(
                **{
                    "total_loss": total_loss / (iteration + 1),
                    "f_score": total_f_score / (iteration + 1),
                    "lr": get_lr(optimizer),
                }
            )
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print("Finish Train")
        print("Start Validation")
        pbar = tqdm(
            total=epoch_step_val,
            desc=f"Epoch {epoch + 1}/{Epoch}",
            postfix={},
            mininterval=0.3,
            disable=not sys.stderr.isatty(),
        )

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            imgs, pngs, labels, weights = _to_device(imgs, pngs, labels, weights, cuda, local_rank)
            loss, _f_score = _forward_loss_score(
                model_train, imgs, pngs, labels, weights, num_classes, dice_loss, focal_loss, inverfreq_loss
            )

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(
                    **{
                        "val_loss": val_loss / (iteration + 1),
                        "f_score": val_f_score / (iteration + 1),
                        "lr": get_lr(optimizer),
                    }
                )
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print("Finish Validation")
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print(f"Epoch: {epoch + 1}/{Epoch}")
        print("Total Loss: %.3f || Val Loss: %.3f " % (total_loss / epoch_step, val_loss / epoch_step_val))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            checkpoint_name = "ep%03d-loss%.3f-val_loss%.3f.pth" % (
                epoch + 1,
                total_loss / epoch_step,
                val_loss / epoch_step_val,
            )
            torch.save(model.state_dict(), os.path.join(save_dir, checkpoint_name))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print("Save best model to best_epoch_weights.pth")
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
