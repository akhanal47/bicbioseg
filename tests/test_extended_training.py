import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
from bicbioseg import Segmenter, SegmenterConfig, TrainingConfig, ExperimentRunConfig
from bicbioseg.exceptions import ModelError
from bicbioseg.utils.load_data import BiosegDataset
from bicbioseg.utils.preprocessing import image_tensor


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny(**kwargs):
    return Segmenter(device="cpu", image_size=(16, 24), model_kwargs={"base_channels": 2, "num_decoder_blocks": 1}, **kwargs)


def pixel_segmenter(**kwargs):
    model = tiny(loss="bce", metrics=[], **kwargs)
    model.model = torch.nn.Conv2d(3, 1, 1)
    return model


def test_device_defaults_and_index_validation(monkeypatch):
    import bicbioseg.trainer as trainer
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(trainer.platform, "system", lambda: "Darwin")
    assert str(Segmenter.resolve_device()) == "mps"
    monkeypatch.setattr(trainer.platform, "system", lambda: "Linux")
    assert str(Segmenter.resolve_device()) == "cuda"
    assert str(Segmenter.resolve_device("cuda:1")) == "cuda:1"
    with pytest.raises(ModelError, match="index"):
        Segmenter.resolve_device("cuda:2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for system in ("Windows", "Linux"):
        monkeypatch.setattr(trainer.platform, "system", lambda: system)
        assert str(Segmenter.resolve_device()) == "cpu"
    with pytest.raises(ModelError, match="CUDA"):
        Segmenter.resolve_device("cuda:0")


def test_discovery_and_preflight_preserve_buffers_gradients_and_modes():
    metadata = Segmenter.available_models(detailed=True)
    assert set(metadata) == set(Segmenter.available_models())
    assert metadata["swin_unet_full"]["install_extra"] == "bicbioseg[transformers]"
    model = tiny()
    before = copy.deepcopy(model.model.state_dict())
    rng = torch.get_rng_state()
    result = model.validate_setup((torch.ones(2, 3, 16, 24), torch.zeros(2, 16, 24)), backward=True)
    assert result["ok"] and result["backward_checked"]
    assert model.model.training
    assert all(p.grad is None for p in model.model.parameters())
    for key, tensor in model.model.state_dict().items():
        torch.testing.assert_close(tensor, before[key])
    torch.testing.assert_close(torch.get_rng_state(), rng)
    with pytest.raises(ModelError, match="NCHW"):
        model.validate_setup(torch.zeros(1, 2, 16, 24))
    with pytest.raises(ValueError, match="CUDA"):
        model.validate_setup(precision="fp16")


def test_accumulation_matches_large_batches_including_partial_group():
    torch.manual_seed(1)
    images, targets = torch.rand(5, 3, 8, 8), torch.randint(2, (5, 8, 8))
    accumulated, direct = pixel_segmenter(), pixel_segmenter()
    direct.model.load_state_dict(accumulated.model.state_dict())
    for model, batch, steps in ((accumulated, 1, 3), (direct, 3, 1)):
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, targets), batch_size=batch)
        model.train(data=loader, optimizer="sgd", lr=.1, accumulation_steps=steps,
                    max_grad_norm=1., verbose=False, progress_bar=False)
    for a, b in zip(accumulated.model.parameters(), direct.model.parameters()):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)


def test_scheduler_resume_matches_uninterrupted_training(tmp_path):
    torch.manual_seed(4)
    images, targets = torch.rand(3, 3, 8, 8), torch.randint(2, (3, 8, 8))
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, targets), batch_size=2)
    whole, split = pixel_segmenter(), pixel_segmenter()
    split.model.load_state_dict(whole.model.state_dict())
    options = dict(data=loader, optimizer="adam", lr=.01, accumulation_steps=2,
                   scheduler="cosine", scheduler_kwargs={"T_max": 4}, verbose=False, progress_bar=False)
    whole.train(epochs=4, **options)
    split.train(epochs=2, save_to=tmp_path / "resume.pt", **options)
    resumed = pixel_segmenter()
    resumed.train(data=loader, epochs=2, resume_from=tmp_path / "resume.pt", verbose=False, progress_bar=False)
    assert resumed.history["lr"] == whole.history["lr"]
    for a, b in zip(whole.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b)
    saved = torch.load(tmp_path / "resume.pt", weights_only=True)
    assert saved["scheduler_state_dict"]["last_epoch"] == 2
    assert saved["training_options"]["accumulation_steps"] == 2
    assert "scaler_state_dict" in saved


@pytest.mark.parametrize("loss,classes", [("bce",1),("dice",1),("dice_bce",1),("focal",1),("tversky",1),
                                         ("sensitivity_specificity",1),("cross_entropy",3),("dice",3),("iou",3)])
def test_ignored_pixels_have_no_loss_gradient_or_metric_effect(loss, classes):
    model = tiny(loss=loss, num_classes=classes, ignore_index=255)
    raw = torch.tensor([[[0, 1], [255, 255]]])
    targets = model._prepare_targets(raw)
    x = torch.randn(1, classes, 2, 2, requires_grad=True)
    value = model.loss_fn(x, targets)
    value.backward()
    assert (x.grad[..., 1, :] == 0).all()
    changed = x.detach().clone()
    changed[..., 1, :] = 100
    torch.testing.assert_close(value.detach(), model.loss_fn(changed, targets))
    assert model._metric_values(x.detach(), targets) == model._metric_values(changed, targets)
    all_ignored = model._prepare_targets(torch.full((1, 2, 2), 255))
    zero = model.loss_fn(x, all_ignored)
    assert zero == 0 and torch.isfinite(zero)
    prediction = (changed[:, 0] > 0).long() if classes == 1 else changed.argmax(1)
    score = model._score_prediction_arrays(prediction[0].numpy(), raw[0].numpy(), model.metrics,
                                            num_classes=classes, ignore_index=255)
    expected = model._metric_values(changed, targets)
    for name in model.metrics:
        assert score[name if classes == 1 else 'macro_' + name] == pytest.approx(expected[name])


def test_uint16_normalization_crop_and_checkpoint(tmp_path):
    image = np.arange(32*48, dtype=np.uint16).reshape(32, 48) * 30
    mask = np.zeros((32, 48), np.int32)
    mask[30, 46] = 1
    policy = {"mode": "range", "min": 0, "max": 65535}
    dataset = BiosegDataset([image], [mask], image_size=(16, 24), in_channels=1, normalization=policy)
    expected, _ = dataset[0]
    torch.testing.assert_close(expected, image_tensor(image, (16, 24), 1, policy))
    crop = BiosegDataset([image], [mask], image_size=(8, 8), in_channels=1, normalization=policy,
                          crop_size=(8, 8), foreground_probability=1)
    assert crop[0][1].sum() == 1
    model = tiny(in_channels=1, normalization=policy, ignore_index=255)
    model.save(tmp_path / "model.pt")
    restored = Segmenter.load(tmp_path / "model.pt", device="cpu")
    assert restored.normalization == policy and restored.ignore_index == 255
    restored.model.eval()
    with torch.no_grad():
        logits = restored.model(expected[None])
    _, _, actual = restored._predict_array(cv2.resize(image, (24,16)), return_raw=True)
    assert logits.shape == (1,1,16,24) and np.isfinite(actual).all()


@pytest.mark.parametrize("classes", [1,3])
@pytest.mark.parametrize("weighting", ["uniform", "gaussian"])
def test_tiled_batching_covers_edges_and_matches_serial(tmp_path, classes, weighting):
    image = np.random.default_rng(4).integers(0, 65536, (19,37,3), dtype=np.uint16)
    path = tmp_path / "sample.tiff"
    cv2.imwrite(str(path), image)
    model = tiny(num_classes=classes, normalization={"mode":"dtype"})
    outputs = []
    batches = []
    handle = model.model.register_forward_pre_hook(lambda m, args: batches.append(args[0].shape[0]))
    for batch in (1,3):
        result = model.inference_large_image(path, save_to=tmp_path / str(batch), patch_size=(24,24), overlap=8,
                                             tile_batch_size=batch, weighting=weighting, save_probability=True, save_overlay=False)
        outputs.append(np.load(result['probability']))
    handle.remove()
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-6)
    assert np.isfinite(outputs[1]).all() and outputs[1].shape[-2:] == (19,37)
    assert max(batches) > 1
    if classes > 1:
        np.testing.assert_allclose(outputs[1].sum(0), 1, atol=1e-6)


def test_named_experiments_save_independent_configurations(tmp_path):
    runs = {name: ExperimentRunConfig(
        segmenter=SegmenterConfig(device="cpu", image_size=(16,24), model_kwargs={"base_channels": width, "num_decoder_blocks":1}),
        training=TrainingConfig(num_workers=0, batch_size=1, verbose=False, progress_bar=False), seed=number)
        for number, (name, width) in enumerate((("narrow",2),("wide",4)))}
    runs['narrow'].save(tmp_path / 'config.json')
    restored = ExperimentRunConfig.load(tmp_path / 'config.json')
    assert restored.segmenter.model_kwargs['base_channels'] == 2
    data = (np.zeros((1,16,24,3), np.uint8), np.zeros((1,16,24), np.uint8))
    result = Segmenter.run_experiment(data, runs=runs, output_dir=tmp_path / 'runs')
    assert set(result) == set(runs)
    for name, width in (("narrow",2),("wide",4)):
        record = json.loads((tmp_path / 'runs' / name / 'experiment.json').read_text())
        assert record['segmenter']['model_kwargs']['base_channels'] == width
        assert record['performance']['training_samples'] == 1
        assert record['performance']['elapsed_seconds'] > 0


def test_bfloat16_training_is_finite():
    model = pixel_segmenter()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.rand(2,3,8,8), torch.zeros(2,8,8)), batch_size=1)
    model.train(data=loader, precision='bf16', verbose=False, progress_bar=False)
    assert np.isfinite(model.history['train_loss']).all()


def test_auxiliary_loss_weighting_and_gradients():
    model = pixel_segmenter()
    main, aux1, aux2 = [torch.randn(2,1,8,8,requires_grad=True) for _ in range(3)]
    target = torch.zeros_like(main)
    model._training_options = {'aux_loss_weights':[0., .25]}
    loss = model._prediction_loss(main,[aux1,aux2],target)
    torch.testing.assert_close(loss, model.loss_fn(main,target) + .25 * model.loss_fn(aux2,target))
    loss.backward()
    assert aux1.grad.abs().sum() == 0 and aux2.grad.abs().sum() > 0
    with pytest.raises(ModelError, match='one weight'):
        model._prediction_loss(main,[],target)


@pytest.mark.parametrize('backend', ['cuda','mps'])
def test_accelerator_training_when_available(backend, tmp_path):
    if backend not in Segmenter.available_devices():
        pytest.skip(f'{backend} hardware unavailable')
    model = Segmenter(architecture='unext_full', device=backend, image_size=(35,49),
                      model_kwargs={'variant':'small','deep_supervision':True})
    precision = 'fp16' if backend == 'cuda' else 'fp32'
    batch = (torch.rand(2,3,35,49), torch.randint(2,(2,35,49)))
    assert model.validate_setup(batch, backward=True, precision=precision)['ok']
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*batch),batch_size=1)
    model.train(data=loader, precision=precision, accumulation_steps=2, max_grad_norm=1.,
                scheduler='step', scheduler_kwargs={'step_size':1}, save_to=tmp_path/'accelerator.pt',
                verbose=False,progress_bar=False)
    model.train(data=loader, resume_from=tmp_path/'accelerator.pt',verbose=False,progress_bar=False)
    assert np.isfinite(model.history['train_loss']).all()
