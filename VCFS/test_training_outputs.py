import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VCFS"))

from train import scaled_lr, sha256_file, validate_split_contract, write_training_manifest
from utils.callbacks import EvalCallback
from utils.split_utils import load_split_lines
from utils.fea_upscale import DINOFeatureProjection
from utils.train_config import TrainConfig
from utils.utils_metrics import mean_metric


class TrainingOutputTests(unittest.TestCase):
    def test_dino_projection_is_registered_stable_and_trainable(self):
        projection = DINOFeatureProjection(8, 4)
        self.assertIn("channel_transform.weight", dict(projection.named_parameters()))
        self.assertIn("batch_norm.running_mean", projection.state_dict())

        dino = torch.randn(2, 4, 8)
        cnn = torch.randn(2, 4, 3, 5)
        projection.eval()
        first = projection(dino, cnn, token_hw=(2, 2))
        second = projection(dino, cnn, token_hw=(2, 2))
        torch.testing.assert_close(first, second)

        projection.train()
        optimizer = torch.optim.SGD(projection.parameters(), lr=0.1)
        before = projection.channel_transform.weight.detach().clone()
        projection(dino, cnn, token_hw=(2, 2)).sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, projection.channel_transform.weight.detach()))

    def test_dinov2_loader_is_pinned_frozen_and_in_eval_mode(self):
        from nets import deeplabv3_plus

        fake_dino = torch.nn.Linear(2, 2)
        previous = deeplabv3_plus._DINOV2_VITS14
        deeplabv3_plus._DINOV2_VITS14 = None
        try:
            with mock.patch.object(torch.hub, "load", return_value=fake_dino) as load:
                loaded = deeplabv3_plus._get_dinov2_vits14(torch.device("cpu"))
            load.assert_called_once_with(
                deeplabv3_plus.DINOV2_REPO,
                deeplabv3_plus.DINOV2_MODEL,
                skip_validation=True,
            )
            self.assertIs(loaded, fake_dino)
            self.assertFalse(loaded.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
        finally:
            deeplabv3_plus._DINOV2_VITS14 = previous

    def test_dinov2_input_uses_imagenet_normalization(self):
        from nets.deeplabv3_plus import DINOV2_MEAN, DINOV2_STD, _normalize_for_dinov2

        image = torch.tensor(DINOV2_MEAN).view(1, 3, 1, 1)
        torch.testing.assert_close(_normalize_for_dinov2(image), torch.zeros_like(image))

        unit_offset = image + torch.tensor(DINOV2_STD).view(1, 3, 1, 1)
        torch.testing.assert_close(
            _normalize_for_dinov2(unit_offset),
            torch.ones_like(unit_offset),
        )

    def test_paper_metric_ignores_background(self):
        self.assertEqual(mean_metric([0.9, 0.4, 0.6], [0]), 0.5)

    def test_sda_split_contract_allows_variable_final_size(self):
        cfg = TrainConfig(
            require_unique_split_ids=True,
            expected_original_train_samples=2,
            expected_val_samples=1,
            minimum_synthetic_train_samples=1,
        )
        counts = validate_split_contract(cfg, ["a\n", "b\n", "syn_a_0\n", "syn_b_0\n"], ["c\n"])
        self.assertEqual(counts, {"original_train": 2, "synthetic_train": 2, "val": 1})

    def test_sda_split_contract_rejects_missing_synthetic_data(self):
        cfg = TrainConfig(
            expected_original_train_samples=2,
            expected_val_samples=1,
            minimum_synthetic_train_samples=1,
        )
        with self.assertRaisesRegex(ValueError, "too few synthetic"):
            validate_split_contract(cfg, ["a\n", "b\n"], ["c\n"])

    def test_sda_split_contract_rejects_duplicate_evaluation_ids(self):
        cfg = TrainConfig(require_unique_split_ids=True)
        with self.assertRaisesRegex(ValueError, "unique ids"):
            validate_split_contract(cfg, ["a\n"], ["c\n", "c\n"])

    def test_empty_fallback_name_does_not_load_original_train_split(self):
        with tempfile.TemporaryDirectory() as directory:
            split_dir = Path(directory) / "txt"
            split_dir.mkdir()
            (split_dir / "train.txt").write_text("a\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                load_split_lines(directory, "train_sda.txt", fallback_name="")

    def test_fixed_lr_uses_paper_value_without_batch_scaling(self):
        cfg = TrainConfig(
            init_lr=2e-3,
            min_lr_ratio=0.01,
            lr_scaling_mode="fixed",
        )
        self.assertEqual(scaled_lr(cfg, batch_size=2), (2e-3, 2e-5))

    def test_legacy_lr_scaling_remains_available(self):
        cfg = TrainConfig(init_lr=2e-4, lr_scaling_mode="legacy_batch")
        init_lr, min_lr = scaled_lr(cfg, batch_size=4)
        self.assertAlmostEqual(init_lr, 3e-4)
        self.assertAlmostEqual(min_lr, 3e-6)

    def test_eval_temporary_output_stays_below_log_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory) / "results" / "loss_test"
            log_dir.mkdir(parents=True)
            callback = EvalCallback(
                None,
                [512, 512],
                7,
                [],
                directory,
                log_dir,
                False,
                eval_flag=False,
                ignore_class_ids=[0],
            )
            self.assertEqual(callback.miou_out_path, log_dir / "miou_tmp")
            self.assertEqual(callback.ignore_class_ids, [0])
            evaluation_config = json.loads(
                (log_dir / "evaluation_config.json").read_text(encoding="utf-8")
            )
            self.assertFalse(evaluation_config["includes_background"])

    def test_training_manifest_hashes_effective_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "train.json"
            class_path = root / "classes.json"
            checkpoint_path = root / "initial.pth"
            train_path = root / "train.txt"
            val_path = root / "val.txt"
            for path, content in (
                (config_path, b"{}"),
                (class_path, b"{}"),
                (checkpoint_path, b"weights"),
                (train_path, b"a\nb\n"),
                (val_path, b"c\n"),
            ):
                path.write_bytes(content)

            cfg = TrainConfig(
                class_config=str(class_path),
                dataset_path=str(root / "dataset"),
                model_path=str(checkpoint_path),
                save_dir=str(root / "results"),
            )
            cfg.train_config_path = str(config_path)
            class_config = {
                "path": str(class_path),
                "classes": ["background", "facade"],
            }
            write_training_manifest(cfg, class_config, train_path, val_path, 2, 1)

            manifest_path = Path(cfg.save_dir) / "training_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["num_train"], 2)
            self.assertEqual(manifest["num_val"], 1)
            self.assertEqual(manifest["eval_ignore_class_ids"], [])
            self.assertEqual(
                manifest["files"]["initial_checkpoint"]["sha256"],
                sha256_file(checkpoint_path),
            )
            self.assertEqual(manifest["files"]["train_split"]["sha256"], sha256_file(train_path))


if __name__ == "__main__":
    unittest.main()
