'''
Regression tests for reusable settings and consistent workflow artifacts.
'''

import inspect
import json
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from bicbioseg import (
    Segmenter,
    SegmenterConfig,
    ExperimentConfig,
    ExperimentRunConfig,
    TrainingConfig,
    SegmentationExperiment,
    ImageOps,
)
from bicbioseg.exceptions import ModelError, DatasetError, InferenceError
from bicbioseg.models.registry import MODEL_SPECS
from bicbioseg.utils.augmentation import AugmentImages, AugmentationPipeline, apply_and_save_augmentations
from bicbioseg.utils.checkpoints import atomic_torch_save, read_checkpoint
from bicbioseg.utils.evaluation import MetricAccumulator


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def tiny(**kwargs):
    return Segmenter(
        image_size=(16, 24),
        device="cpu",
        model_kwargs={"base_channels": 2, "num_decoder_blocks": 1},
        **kwargs,
    )


def dataset(root, count=3):
    root = Path(root)
    for folder in ["images", "masks"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(13)
    for i in range(count):
        h, w = 19 + i, 27 + i
        cv2.imwrite(str(root / "images" / f"{i}.png"), rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
        mask = np.zeros((h, w), np.uint8)
        mask[2:8, 3:9] = 1
        cv2.imwrite(str(root / "masks" / f"{i}.png"), mask)
    return root


SMALL = {
    "unet": {"base_channels": 2, "num_decoder_blocks": 2},
    "cattention_unet": {"base_channels": 2, "num_decoder_blocks": 2},
    "double_unet": {"backbone": "resnet50", "pretrained": False},
    "transunet": {
        "out_channels": 8,
        "embedding_dim": 32,
        "head_num": 4,
        "mlp_dim": 32,
        "block_num": 1,
        "decoder_channels": [16, 8, 4, 2],
    },
    "segformer": {"dims": [4, 8, 8, 16], "heads": [1, 1, 1, 1], "num_layers": [1, 1, 1, 1], "decoder_dim": 8},
    "deit": {"decoder_channels": [16, 8, 4, 2]},
    "swin_unet": {"decoder_channels": 8},
    "pvt_unet": {"decoder_channels": 8},
    "swin_unet_full": {"decoder_depths": [1, 1, 1]},
    "pvtformer_full": {"variant": "b0", "decoder_channels": 8},
    "resunetplusplus_full": {"base_channels": 2, "aspp_rates": [1, 2]},
    "unext_full": {"widths": [4, 8, 8, 16, 16]},
}


@pytest.mark.parametrize("name", MODEL_SPECS)
def test_every_model_json_roundtrip_builds_and_forwards(tmp_path, name):
    config = SegmenterConfig(architecture=name, device="cpu", image_size=(32, 48), model_kwargs=SMALL[name])
    path = tmp_path / "model.json"
    config.save(path)
    restored = SegmenterConfig.load(path)
    assert restored == config
    model = Segmenter.from_config(restored)
    model.model.eval()
    with torch.no_grad():
        assert model._forward_logits(torch.zeros(1, 3, 32, 48)).shape == (1, 1, 32, 48)
    nested = ExperimentRunConfig(segmenter=restored, training=TrainingConfig(epochs=2))
    nested.save(tmp_path / "run.json")
    assert ExperimentRunConfig.load(tmp_path / "run.json").segmenter == config
    workflow = ExperimentConfig(
        images="images", masks="masks", model=name, image_size=(32, 48), model_kwargs=SMALL[name]
    )
    workflow.save(tmp_path / "workflow.json")
    assert ExperimentConfig.load(tmp_path / "workflow.json") == workflow


def test_model_options_cover_constructor_defaults_without_download():
    from importlib import import_module

    for name, spec in MODEL_SPECS.items():
        schema = Segmenter.model_options(name)
        constructor = getattr(import_module(f"bicbioseg.models.{spec.module}"), spec.class_name)
        params = inspect.signature(constructor).parameters
        shared = {spec.channel_arg, spec.class_arg, spec.size_arg}
        assert set(schema) == set(params) - shared
        for key, rule in schema.items():
            if name == "double_unet" and key == "pretrained":
                continue
            assert rule["default"] == params[key].default
        schema.clear()
        assert Segmenter.model_options(name)
    assert Segmenter.model_options("pvt") == Segmenter.model_options("pvt_unet")


@pytest.mark.parametrize(
    "name,options,match",
    [
        ("segformer", {"heads": [3, 2, 5, 8]}, "divisible"),
        ("transunet", {"embedding_dim": 31}, "divisible"),
        ("unet", {"typo": True}, "Unsupported"),
        ("unext_full", {"widths": [1, 2]}, "widths"),
        ("deit", {"variant": "large"}, "variant"),
    ],
)
def test_invalid_config_fails_before_model_construction(name, options, match):
    with pytest.raises(ModelError, match=match):
        SegmenterConfig(architecture=name, model_kwargs=options)


def test_color_map_discovers_later_colors_and_is_reusable(tmp_path):
    masks = tmp_path / "colors"
    masks.mkdir()
    red = np.zeros((4, 4, 3), np.uint8)
    red[:, 2:] = (255, 0, 0)
    green = np.zeros((4, 4, 3), np.uint8)
    green[:, 2:] = (0, 255, 0)
    for name, rgb in [("a", red), ("b", green)]:
        cv2.imwrite(str(masks / f"{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    out = Path(ImageOps.normalize_masks(masks, tmp_path / "labels", mode="color"))
    color_map = ImageOps.load_color_map(out / "class_map.json")
    assert set(color_map) == {(0, 0, 0), (255, 0, 0), (0, 255, 0)}
    assert cv2.imread(str(out / "b.png"), -1)[0, 3] == color_map[(0, 255, 0)]
    expected, _ = ImageOps.convert_color_mask_to_labels(green, out / "class_map.json")
    np.testing.assert_array_equal(expected, cv2.imread(str(out / "b.png"), -1))
    with pytest.raises(DatasetError, match="Unknown RGB"):
        ImageOps.convert_color_mask_to_labels(green, {(0, 0, 0): 0})
    with pytest.warns(UserWarning, match="Unknown RGB"):
        ignored, _ = ImageOps.convert_color_mask_to_labels(
            green, {(0, 0, 0): 0}, unknown_color="ignore", ignore_index=255
        )
    assert ignored[0, 3] == 255


def test_qc_label_ranges_ignore_and_split_coverage(tmp_path):
    root = dataset(tmp_path / "train", 1)
    dataset(tmp_path / "validate", 1)
    mask = np.full((19, 27), 255, np.uint8)
    mask[0, 0] = 4
    cv2.imwrite(str(root / "masks" / "0.png"), mask)
    qc = ImageOps.dataset_split_qc_report(
        tmp_path, num_classes=3, ignore_index=255, save_to=tmp_path / "qc.json"
    )
    assert qc["splits"]["train"]["ignored_pixels"] == 512
    assert qc["splits"]["train"]["invalid_labels"][0]["labels"] == [4]
    assert 2 in qc["missing_classes"]["validate"]
    mask[:] = 255
    cv2.imwrite(str(root / "masks" / "0.png"), mask)
    report = ImageOps.dataset_qc_report(root / "images", root / "masks", num_classes=3, ignore_index=255)
    assert report["ignored_only_masks"] == ["0.png"]
    assert report["num_empty_masks"] == 0


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_augmentation_preserves_dtype_range_and_mask_ids(tmp_path, dtype):
    maximum = 1 if np.issubdtype(dtype, np.floating) else np.iinfo(dtype).max
    image = (np.random.default_rng(4).random((24, 32, 3)) * maximum).astype(dtype)
    mask = np.zeros((24, 32), np.uint16)
    mask[5:20, 6:25] = 300
    for transform in [
        AugmentImages.rotate,
        AugmentImages.flip,
        AugmentImages.adjust_brightness,
        AugmentImages.hist_equalize,
        partial(AugmentImages.random_crop, crop_size=(16, 16)),
    ]:
        result, labels = transform(image, mask)
        assert result.dtype == dtype and 0 <= result.min() <= result.max() <= maximum
        assert labels.dtype == mask.dtype and set(np.unique(labels)).issubset({0, 300})
    pipeline = AugmentationPipeline([partial(AugmentImages.rotate, angle_range=(10, 10))])
    report = pipeline.preview(image, mask, save_to=tmp_path / "preview.png")
    assert report["labels_preserved"] and (tmp_path / "preview.png").exists()


def test_saved_augmentation_uses_rgb_and_preserves_uint16(tmp_path):
    root = dataset(tmp_path / "raw", 1)
    image = np.zeros((19, 27, 3), np.uint16)
    image[..., 0] = 50000
    image[..., 1] = 10000
    cv2.imwrite(str(root / "images" / "0.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    seen = []

    def transform(rgb, mask):
        seen.append(rgb.copy())
        return AugmentImages.hist_equalize(rgb, mask)

    files = apply_and_save_augmentations(
        root / "images", root / "masks", tmp_path / "aug", augmentations=[transform]
    )
    np.testing.assert_array_equal(seen[0], image)
    expected, _ = AugmentImages.hist_equalize(image)
    actual = cv2.cvtColor(cv2.imread(files[0], -1), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(expected, actual)
    assert actual.dtype == np.uint16


def test_metrics_exclude_empty_and_ignored_and_offer_dataset_aggregation():
    acc = MetricAccumulator(1, ("dice", "iou"), ignore_index=255)
    assert acc.add(np.zeros((2, 2)), np.full((2, 2), 255))["dice"] is None
    assert acc.add(np.zeros((2, 2)), np.zeros((2, 2)))["dice"] is None
    acc.add(np.array([[1, 0]]), np.array([[1, 1]]))
    acc.add(np.ones((2, 2)), np.ones((2, 2)))
    report = acc.summary()
    assert report["mean"]["dice"] == pytest.approx((2 / 3 + 1) / 2)
    assert report["dataset"]["dice"] == pytest.approx(10 / 11)
    assert report["ignored_only_samples"] == 1 and report["class_support"]["1"]["pixels"] == 6
    absent = MetricAccumulator(3, ("dice",))
    row = absent.add(np.ones((2, 2)), np.ones((2, 2)))
    assert row["dice_class_2"] is None and row["macro_dice"] == 1
    only_ignored = MetricAccumulator(1, ignore_index=255, empty_policy="one")
    assert only_ignored.add(np.zeros((2, 2)), np.full((2, 2), 255))["dice"] is None


def test_training_metric_options_use_same_policy():
    model = tiny(num_classes=3, include_background=True, metric_aggregation="dataset")
    logits = torch.full((1, 3, 2, 2), -10.0)
    logits[:, 0] = 10
    scores = model._metric_values(logits, torch.zeros((1, 2, 2), dtype=torch.long))
    assert scores["dice"] == 1
    model.include_background = False
    assert model._metric_values(logits, torch.zeros((1, 2, 2), dtype=torch.long))["dice"] is None


def test_checkpoints_atomic_versioned_and_resumable(tmp_path, monkeypatch):
    model = tiny(loss="bce")
    data = (np.zeros((2, 16, 24, 3), np.uint8), np.ones((2, 16, 24), np.uint8))
    model.train(
        data=data,
        epochs=1,
        num_workers=0,
        experiment_dir=tmp_path,
        run_name="run",
        checkpoint_interval=1,
        verbose=False,
        progress_bar=False,
    )
    path = tmp_path / "run" / "last_model.pt"
    checkpoint = read_checkpoint(path)
    assert checkpoint["checkpoint_format_version"] == 1 and checkpoint["completed_epoch"] == 1
    assert checkpoint["dataset_identity"]["content_verified"]
    assert "torch" in checkpoint["package_versions"]
    restored = Segmenter.load(path)
    restored.train(data=data, epochs=1, num_workers=0, resume_from=path, verbose=False, progress_bar=False)
    assert restored.completed_epoch == 2
    changed = (data[0] + 1, data[1])
    with pytest.raises(ValueError, match="dataset identity"):
        restored.train(data=changed, num_workers=0, resume_from=path)
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_torch_save({"value": 1}, path)
    assert path.read_bytes() == original and not list(path.parent.glob("*.tmp"))


def test_future_checkpoint_rejected(tmp_path):
    path = tmp_path / "future.pt"
    torch.save({"checkpoint_format_version": 999}, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        Segmenter.load(path)


def test_batched_inference_matches_serial_with_mixed_sizes(tmp_path):
    root = dataset(tmp_path / "raw")
    model = tiny(class_map={0: "background", 1: "cell"})
    serial = model.inference(
        root / "images", save_to=tmp_path / "serial", return_arrays=True, structured=True
    )
    batched = model.inference(
        root / "images",
        save_to=tmp_path / "batch",
        batch_size=2,
        return_arrays=True,
        structured=True,
        save_probability=True,
    )
    for a, b in zip(serial, batched):
        np.testing.assert_array_equal(a["mask"], b["mask"])
        np.testing.assert_allclose(a["probability"], b["probability"], atol=1e-6)
    metadata = json.loads((tmp_path / "batch" / "predictions.json").read_text())["predictions"][0]["metadata"]
    assert metadata["threshold"] == 0.5 and metadata["class_map"]["1"] == "cell"
    single = model.predict_one(root / "images" / "0.png", structured=True, save_to=tmp_path / "one.png")
    tiled = model.inference_large_image(
        root / "images" / "0.png", patch_size=(16, 24), overlap=4, structured=True, save_to=tmp_path / "tiles"
    )
    model.save(tmp_path / "model.pt")
    ensemble = Segmenter.ensemble_predict(
        [tmp_path / "model.pt"],
        root / "images",
        batch_size=2,
        structured=True,
        save_to=tmp_path / "ensemble",
        return_arrays=True,
    )
    assert set(single) == set(tiled) == set(batched[0]) == set(ensemble[0])
    for a, b in zip(batched, ensemble):
        np.testing.assert_array_equal(a["mask"], b["mask"])


def test_workflow_checkpoint_selection_and_report_external_artifacts(tmp_path):
    raw = dataset(tmp_path / "raw", 1)
    exp = SegmentationExperiment(
        raw / "images",
        raw / "masks",
        work_dir=tmp_path / "work",
        image_size=(16, 24),
        device="cpu",
        model_kwargs={"base_channels": 2, "num_decoder_blocks": 1},
    )
    with torch.no_grad():
        for parameter in exp.segmenter.model.parameters():
            parameter.zero_()
        exp.segmenter.model.outc.conv.bias.fill_(10)
    exp.segmenter.save(exp.run_dir / "best_model.pt")
    with torch.no_grad():
        exp.segmenter.model.outc.conv.bias.fill_(-10)
    exp.segmenter.save(exp.run_dir / "final_model.pt")
    current = exp.segmenter.model.outc.conv.bias.detach().clone()
    best = exp.predict(
        raw / "images", checkpoint="best", save_to=tmp_path / "best", return_arrays=True, structured=True
    )
    final = exp.predict(
        raw / "images", checkpoint="final", save_to=tmp_path / "final", return_arrays=True, structured=True
    )
    assert np.all(best[0]["mask"] == 255) and np.all(final[0]["mask"] == 0)
    torch.testing.assert_close(current, exp.segmenter.model.outc.conv.bias)
    scores = exp.evaluate(
        images=raw / "images",
        masks=raw / "masks",
        checkpoint="best",
        save_to=tmp_path / "external",
        threshold=0.7,
    )
    assert scores["provenance"]["split"] == "custom" and scores["provenance"]["threshold"] == 0.7
    report = Path(exp.report()).read_text()
    assert "best_model.pt" in report and "Dataset split: custom" in report and "Threshold: 0.7" in report
    assert "Example predictions" in report and "Class support" in report


def test_recovery_checkpoint_survives_interrupted_next_epoch(tmp_path, monkeypatch):
    model = tiny(loss="bce")
    data = (np.zeros((2, 16, 24, 3), np.uint8), np.ones((2, 16, 24), np.uint8))
    original = model._run_epoch
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "_run_epoch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        model.train(
            data=data,
            epochs=3,
            num_workers=0,
            experiment_dir=tmp_path,
            run_name="run",
            checkpoint_interval=1,
            verbose=False,
            progress_bar=False,
        )
    last = tmp_path / "run" / "last_model.pt"
    checkpoint = read_checkpoint(last)
    assert checkpoint["completed_epoch"] == 1 and len(checkpoint["history"]["train_loss"]) == 1
    restored = Segmenter.load(last)
    restored.train(data=data, resume_from=last, epochs=1, num_workers=0, verbose=False, progress_bar=False)
    assert restored.completed_epoch == 2 and len(restored.history["train_loss"]) == 2


def test_equalization_interprets_rgb_channels(monkeypatch):
    captured = []

    class IdentityCLAHE:
        def apply(self, luminance):
            captured.append(luminance.copy())
            return luminance

    monkeypatch.setattr(cv2, "createCLAHE", lambda **kwargs: IdentityCLAHE())
    image = np.full((8, 8, 3), (200, 20, 10), np.uint8)
    result, _ = AugmentImages.hist_equalize(image)
    expected_luminance = round(0.299 * 200 + 0.587 * 20 + 0.114 * 10)
    assert abs(int(captured[0][0, 0]) - expected_luminance) <= 1
    np.testing.assert_allclose(result, image, atol=2)


def test_preview_detects_interpolated_or_new_labels(tmp_path):
    image = np.zeros((4, 4, 3), np.uint8)
    mask = np.ones((4, 4), np.uint8)
    bad = AugmentationPipeline([lambda image, mask: (image, mask.astype(float) + 0.5)])
    report = bad.preview(image, mask, save_to=tmp_path / "bad.png")
    assert not report["labels_preserved"] and report["unexpected_labels"] == [1.5]


def test_float_files_remain_float_and_unscaled(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    image = np.random.default_rng(7).random((8, 9, 3)).astype(np.float32)
    cv2.imwrite(
        str(source / "image.tiff"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_TIFF_COMPRESSION, 1]
    )
    paths = apply_and_save_augmentations(
        source, output_dir=tmp_path / "out", augmentations=[partial(AugmentImages.flip, mode="horizontal")]
    )
    output = cv2.cvtColor(cv2.imread(paths[0], -1), cv2.COLOR_BGR2RGB)
    assert output.dtype == np.float32 and Path(paths[0]).suffix == ".tiff"
    np.testing.assert_array_equal(output, image[:, ::-1])


def test_qc_reports_unmatched_without_aborting(tmp_path):
    root = dataset(tmp_path / "raw", 1)
    (root / "masks" / "0.png").unlink()
    qc = ImageOps.dataset_qc_report(root / "images", root / "masks", num_classes=2)
    assert qc["num_pairs"] == 0 and qc["unmatched"]["images_without_masks"]


def test_config_roundtrip_restores_training_sequences_and_class_map(tmp_path):
    config = ExperimentRunConfig(
        segmenter=SegmenterConfig(class_map={0: "background", 1: "cell"}),
        training=TrainingConfig(crop_size=(20, 30), aux_loss_weights=(0.4, 0.2)),
    )
    config.save(tmp_path / "config.json")
    assert ExperimentRunConfig.load(tmp_path / "config.json") == config


def test_evaluation_all_ignored_serializes_nulls_and_no_inflated_score(tmp_path):
    predictions = tmp_path / "predictions"
    masks = tmp_path / "masks"
    predictions.mkdir()
    masks.mkdir()
    cv2.imwrite(str(predictions / "a_mask.png"), np.ones((4, 4), np.uint8))
    cv2.imwrite(str(masks / "a.png"), np.full((4, 4), 255, np.uint8))
    result = Segmenter.evaluate_predictions(
        predictions, masks, num_classes=1, ignore_index=255, save_to=tmp_path / "eval"
    )
    assert result["scores"]["dice"] is None and result["dataset"]["dice"] is None
    assert result["ignored_only_samples"] == 1
    text = (tmp_path / "eval" / "evaluation_summary.json").read_text()
    assert "null" in text and "NaN" not in text


def test_metric_aggregation_is_independent_of_batch_partition():
    model = tiny(num_classes=3, metric_aggregation="dataset")
    predictions = torch.tensor([[[0, 1], [1, 1]], [[0, 0], [0, 0]], [[1, 2], [1, 2]]])
    targets = torch.tensor([[[0, 1], [0, 1]], [[0, 0], [0, 0]], [[1, 1], [1, 2]]])
    logits = torch.nn.functional.one_hot(predictions, 3).movedim(-1, 1).float() * 10
    whole = model._metric_accumulator()
    model._add_metric_batch(whole, logits, targets)
    split = model._metric_accumulator()
    for i in range(3):
        model._add_metric_batch(split, logits[i : i + 1], targets[i : i + 1])
    assert whole.summary() == split.summary()
