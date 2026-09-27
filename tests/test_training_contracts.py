import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from bicbioseg import Segmenter, SegmentationExperiment, TrainingConfig
from bicbioseg.exceptions import DatasetError, ModelError
from bicbioseg.utils import losses
from bicbioseg.utils.load_data import BiosegDataset, DataLoader


def tiny_segmenter(**kwargs):
    defaults = dict(image_size=(16, 24), device="cpu", model_kwargs={"base_channels": 2, "num_decoder_blocks": 1})
    defaults.update(kwargs)
    return Segmenter(**defaults)


@pytest.mark.parametrize("loss_cls", [losses.DiceLoss, losses.BCELoss, losses.DiceBCELoss, losses.FocalLoss,
    losses.LogCoshDiceLoss, losses.JaccardLoss, losses.TverskyLoss, losses.FocalTverskyLoss,
    losses.SensitivitySpecificityLoss])
def test_losses_reward_correct_predictions_and_have_gradients(loss_cls):
    targets = torch.tensor([[[[0., 1.], [1., 0.]]]])
    good = (targets * 2 - 1) * 8
    bad = -good
    loss_fn = loss_cls()
    assert loss_fn(good, targets) < loss_fn(bad, targets)
    assert loss_fn(good, targets) < 0.01
    logits = torch.zeros_like(targets, requires_grad=True)
    loss = loss_fn(logits, targets)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 1e-6
    assert (logits.grad[targets == 1] < 0).all()
    assert (logits.grad[targets == 0] > 0).all()
    for target in (torch.zeros_like(targets), torch.ones_like(targets)):
        assert torch.isfinite(loss_fn(torch.full_like(target, -100), target))


def test_combo_and_log_cosh_use_dice_loss_not_dice_score():
    inputs = torch.zeros(1, 1, 2, 2)
    targets = torch.tensor([[[[0., 1.], [1., 0.]]]])
    dice = losses.DiceLoss()(inputs, targets)
    torch.testing.assert_close(losses.DiceBCELoss()(inputs, targets), dice + losses.BCELoss()(inputs, targets))
    torch.testing.assert_close(losses.LogCoshDiceLoss()(inputs, targets), torch.log(torch.cosh(dice)))


def test_tversky_alpha_and_beta_have_documented_meaning():
    target = torch.tensor([[[[1., 0.]]]])
    false_negative = torch.tensor([[[[-8., -8.]]]])
    false_positive = torch.tensor([[[[8., 8.]]]])
    loss_fn = losses.TverskyLoss(alpha=0.1, beta=0.9)
    assert loss_fn(false_negative, target) > loss_fn(false_positive, target)


@pytest.mark.parametrize("loss", ["dice", "jaccard", "cross_entropy"])
def test_multiclass_grayscale_rectangular_training_and_checkpoint(tmp_path, loss):
    images = np.random.default_rng(42).integers(0, 256, (2, 16, 24), dtype=np.uint8)
    masks = np.zeros((2, 16, 24), dtype=np.uint8)
    masks[:, :8] = 1
    masks[:, 8:, :12] = 2
    segmenter = tiny_segmenter(num_classes=3, in_channels=1, loss=loss)
    before = next(segmenter.model.parameters()).detach().clone()
    history = segmenter.train(data=(images, masks), batch_size=2, num_workers=0, verbose=False, progress_bar=False,
                              experiment_dir=tmp_path, run_name=loss)
    assert np.isfinite(history["train_loss"]).all()
    assert not torch.equal(before, next(segmenter.model.parameters()).detach())
    assert (tmp_path / loss / "best_model.pt").exists()
    restored = Segmenter.load(tmp_path / loss / "best_model.pt", device="auto")
    assert restored.in_channels == 1
    pred, probability, logits = restored._predict_array(np.zeros((19, 29, 3), np.uint8), return_raw=True)
    assert pred.shape == (19, 29)
    assert probability.shape == (3, 19, 29)
    np.testing.assert_allclose(probability.sum(axis=0), 1, atol=1e-6)
    np.testing.assert_array_equal(pred, probability.argmax(axis=0))
    assert not restored.model.training


def test_multiclass_metrics_use_class_membership_and_match_evaluation():
    segmenter = tiny_segmenter(num_classes=3, metrics=["dice", "iou", "precision", "recall"])
    truth = torch.tensor([[[0, 1], [2, 2]]])
    prediction = torch.tensor([[[0, 2], [1, 2]]])
    logits = torch.nn.functional.one_hot(prediction, 3).movedim(-1, 1).float() * 20
    actual = segmenter._metric_values(logits, truth)
    expected = Segmenter._score_prediction_arrays(prediction.numpy()[0], truth.numpy()[0], segmenter.metrics, num_classes=3)
    for name in segmenter.metrics:
        assert actual[name] == pytest.approx(expected[f"macro_{name}"])
        assert 0 <= actual[name] <= 1
    perfect = torch.nn.functional.one_hot(truth, 3).movedim(-1, 1).float() * 20
    assert segmenter._metric_values(perfect, truth)["dice"] == 1


def test_preprocessing_matches_between_training_and_inference():
    image = np.random.default_rng(42).integers(0, 256, (19, 31, 3), dtype=np.uint8)
    dataset = BiosegDataset([image], [np.zeros((19, 31), np.uint8)], image_size=(16, 24), in_channels=1)
    expected, mask = dataset[0]
    segmenter = tiny_segmenter(in_channels=1)
    seen = []
    handle = segmenter.model.register_forward_pre_hook(lambda model, args: seen.append(args[0].detach().cpu()))
    segmenter._predict_array(image)
    handle.remove()
    torch.testing.assert_close(seen[0][0], expected, atol=1e-6, rtol=1e-6)
    assert expected.shape == (1, 16, 24)
    assert mask.shape == (16, 24)


def test_loader_accepts_path_and_tiff_and_rejects_ambiguous_masks(tmp_path):
    for name in ("images", "masks"):
        (tmp_path / name).mkdir()
    cv2.imwrite(str(tmp_path / "images" / "sample.TIFF"), np.zeros((8, 12, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "masks" / "sample.png"), np.zeros((8, 12), np.uint8))
    loader = DataLoader.load_data(tmp_path, image_size=(8, 12), num_workers=0)
    assert next(iter(loader))[0].shape == (1, 3, 8, 12)
    cv2.imwrite(str(tmp_path / "masks" / "sample.bmp"), np.zeros((8, 12), np.uint8))
    with pytest.raises(DatasetError, match="Duplicate"):
        DataLoader.load_data(tmp_path, num_workers=0)


@pytest.mark.parametrize("kwargs, message", [
    ({"num_classes": 3, "loss": "bce"}, "binary-only"),
    ({"loss": "cross_entropy"}, "requires"),
    ({"architecture": "double_unet", "num_classes": 2}, "only RGB binary"),
    ({"architecture": "double_unet", "in_channels": 1}, "only RGB binary"),
    ({"model_kwargs": {"n_classes": 3}}, "conflicts"),
    ({"metrics": ["typo"]}, "Unsupported metrics"),
])
def test_incompatible_configuration_fails_early(kwargs, message):
    with pytest.raises(ModelError, match=message):
        tiny_segmenter(**kwargs)


def test_segformer_odd_rectangular_output_can_train():
    segmenter = Segmenter(architecture="segformer", image_size=(35, 49), device="cpu",
        model_kwargs={"dims": (4, 8, 8, 16), "heads": (1, 1, 1, 1), "num_layers": 1, "decoder_dim": 8})
    images = torch.rand(2, 3, 35, 49)
    logits = segmenter._forward_logits(images)
    assert logits.shape == (2, 1, 35, 49)
    loss = segmenter.loss_fn(logits, torch.ones_like(logits))
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in segmenter.model.parameters())


def test_data_argument_honors_explicit_validation_and_monitor_errors():
    data = (np.zeros((2, 16, 24, 3), np.uint8), np.zeros((2, 16, 24), np.uint8))
    segmenter = tiny_segmenter()
    train, val = segmenter._resolve_loaders(data=data, val_data=data, num_workers=0)
    assert val is not None
    with pytest.raises(ValueError, match="unavailable"):
        segmenter.train(data=data, monitor="val_dice", num_workers=0)
    with pytest.raises(ValueError, match="no image/mask"):
        DataLoader.load_data(([], []), num_workers=0)


def test_resume_preserves_previous_best(tmp_path, monkeypatch):
    segmenter = tiny_segmenter()
    data = (np.zeros((1, 16, 24, 3), np.uint8), np.zeros((1, 16, 24), np.uint8))
    values = iter([0.2, 0.7])
    monkeypatch.setattr(segmenter, "_run_epoch", lambda *args, **kwargs: {"loss": next(values)})
    segmenter.train(data=data, num_workers=0, experiment_dir=tmp_path, run_name="run", verbose=False, progress_bar=False)
    segmenter.train(data=data, num_workers=0, experiment_dir=tmp_path, run_name="run", verbose=False, progress_bar=False,
                    resume_from=tmp_path / "run" / "final_model.pt")
    best = torch.load(tmp_path / "run" / "best_model.pt", map_location="cpu")
    assert best["history"]["train_loss"] == [0.2]
    assert segmenter.history["train_loss"] == [0.2, 0.7]


def test_workflow_honors_output_overrides_and_config_run_dir(tmp_path, monkeypatch):
    experiment = SegmentationExperiment("unused", "unused", work_dir=tmp_path, image_size=(16, 24), device="cpu",
                                         model_kwargs={"base_channels": 2, "num_decoder_blocks": 1})
    monkeypatch.setattr(experiment.segmenter, "train", lambda **kwargs: {})
    monkeypatch.setattr(experiment.segmenter, "inference", lambda **kwargs: kwargs["save_to"])
    monkeypatch.setattr(experiment.segmenter, "evaluate", lambda **kwargs: kwargs["save_to"])
    experiment.train(config=TrainingConfig(experiment_dir=str(tmp_path / "custom"), run_name="configured"))
    assert experiment.run_dir == tmp_path / "custom" / "configured"
    assert experiment.predict("unused", save_to="custom_predictions") == "custom_predictions"
    assert experiment.evaluate(save_to="custom_evaluation") == "custom_evaluation"


def test_empty_validation_split_uses_training_monitor(tmp_path):
    for split in ("train", "validate"):
        for name in ("images", "masks"):
            (tmp_path / split / name).mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "train" / "images" / "cell.png"), np.zeros((16, 24, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "train" / "masks" / "cell.png"), np.zeros((16, 24), np.uint8))
    _, validation = tiny_segmenter()._resolve_loaders(data=tmp_path, num_workers=0)
    assert validation is None


def test_existing_run_requires_resume(tmp_path):
    run = tmp_path / "existing"
    run.mkdir()
    (run / "history.json").write_text("{}")
    with pytest.raises(FileExistsError, match="Choose a new run_name"):
        Segmenter._prepare_run_dir(tmp_path, "existing")
    assert Segmenter._prepare_run_dir(tmp_path, "existing", run / "final_model.pt") == run


def test_evaluation_rejects_partial_pairs(tmp_path):
    from bicbioseg.exceptions import InferenceError
    (tmp_path / "masks").mkdir()
    with pytest.raises(InferenceError, match="No ground-truth"):
        Segmenter.evaluate_predictions([tmp_path / "cell_mask.png"], tmp_path / "masks")


def test_dataset_output_cannot_delete_inputs_and_resize_is_height_width(tmp_path):
    from bicbioseg import create_dataset_split, ImageOps
    for folder in ("images", "masks"):
        (tmp_path / folder).mkdir()
        cv2.imwrite(str(tmp_path / folder / "sample.png"), np.zeros((16, 24), np.uint8))
    with pytest.raises(DatasetError, match="must not contain source"):
        create_dataset_split(tmp_path / "images", tmp_path / "masks", save_to=tmp_path, overwrite=True, progress=False)
    assert (tmp_path / "images" / "sample.png").exists()
    assert ImageOps.resize_image(np.zeros((16, 24), np.uint8), (10, 15)).shape == (10, 15)


@pytest.mark.parametrize("width", [32, 64, 128])
def test_transunet_configurable_width_matches_decoder(width):
    segmenter = Segmenter(architecture="transunet", image_size=(32, 32), device="cpu",
                          model_kwargs={"out_channels": width, "block_num": 1, "mlp_dim": 32})
    segmenter.model.eval()
    with torch.no_grad():
        assert segmenter._forward_logits(torch.zeros(1, 3, 32, 32)).shape == (1, 1, 32, 32)
