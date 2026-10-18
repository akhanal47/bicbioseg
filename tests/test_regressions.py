from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from bicbioseg import Segmenter
from bicbioseg.exceptions import InferenceError
from bicbioseg.utils.load_data import BiosegDataset


@pytest.fixture
def model():
    return Segmenter(
        image_size=(16, 24), device="cpu",
        model_kwargs={"base_channels": 2, "num_decoder_blocks": 1},
    )


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "cell.png"
    assert cv2.imwrite(str(path), np.full((16, 24, 3), 128, np.uint8))
    return path


@pytest.mark.parametrize("method", ["inference", "predict_one", "inference_large_image", "ensemble_predict"])
@pytest.mark.parametrize("failure", ["false", "error"])
def test_prediction_write_failure_reports_path(model, image_path, tmp_path, monkeypatch, method, failure):
    checkpoint = model.save(tmp_path / "model.pt")
    output = tmp_path / "predictions"
    mask_path = output / "cell_mask.png"

    def fail_write(path, array):
        if failure == "error":
            raise cv2.error("Test write failure")
        return False

    monkeypatch.setattr(cv2, "imwrite", fail_write)
    with pytest.raises(InferenceError, match="cell_mask.png"):
        if method == "predict_one":
            model.predict_one(image_path, save_to=mask_path)
        elif method == "inference_large_image":
            model.inference_large_image(image_path, save_to=output, patch_size=(16, 24), overlap=0)
        elif method == "ensemble_predict":
            Segmenter.ensemble_predict([checkpoint], image_path, save_to=output, device="cpu")
        else:
            model.inference(image_path, save_to=output)
    assert not list(output.rglob("*.json"))


@pytest.mark.parametrize("kind", ["overlay", "contours"])
def test_optional_prediction_write_failure_stops_manifest(model, image_path, tmp_path, monkeypatch, kind):
    output = tmp_path / "predictions"
    real_write = cv2.imwrite

    def write(path, array):
        return False if Path(path).stem.endswith(f"_{kind}") else real_write(path, array)

    monkeypatch.setattr(cv2, "imwrite", write)
    with pytest.raises(InferenceError, match=f"cell_{kind}.png"):
        model.inference(image_path, save_to=output, **{f"save_{kind}": True})
    assert (output / "cell_mask.png").exists()
    assert not (output / "predictions.json").exists()


@pytest.mark.parametrize("source", ["folder", "list"])
def test_empty_inference_source_fails_before_output(model, tmp_path, source):
    empty = tmp_path / "empty"
    empty.mkdir()
    output = tmp_path / "predictions"
    with pytest.raises(InferenceError, match="No images"):
        model.inference(empty if source == "folder" else [], save_to=output)
    assert not output.exists()


def test_predict_one_without_overlay_skips_overlay_work(model, image_path, monkeypatch):
    def unexpected_overlay(*args, **kwargs):
        pytest.fail("A mask-only request must not create an overlay.")

    monkeypatch.setattr(model, "_overlay_mask_on_image", unexpected_overlay)
    result = model.predict_one(image_path, structured=True)
    assert result["mask"].shape == (16, 24)
    assert result["probability"].shape == (16, 24)
    assert result["paths"] == {}


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("dtype,normalization", [(np.uint8, None), (np.uint16, {"mode": "dtype"})])
def test_dataset_resizes_image_once_and_matches_prediction(model, monkeypatch, channels, dtype, normalization):
    image = np.random.default_rng(12).integers(0, 256, (19, 31, channels), dtype=dtype)
    original = image.copy()
    dataset = BiosegDataset(
        [image], [np.zeros((19, 31), np.uint8)], image_size=model.image_size,
        normalization=normalization,
    )
    real_resize = cv2.resize
    image_resizes = []

    def resize(array, size, *, interpolation):
        if interpolation == cv2.INTER_LINEAR:
            image_resizes.append(size)
        return real_resize(array, size, interpolation=interpolation)

    monkeypatch.setattr(cv2, "resize", resize)
    tensor, mask = dataset[0]
    assert image_resizes == [(24, 16)]
    assert mask.shape == (16, 24)
    model.normalization = dataset.normalization
    seen = []
    handle = model.model.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().cpu()))
    try:
        model._predict_array(image)
    finally:
        handle.remove()
    torch.testing.assert_close(tensor, seen[0][0], rtol=0, atol=0)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("backbone", ["vgg19", "resnet50", "resnet101"])
@pytest.mark.parametrize("pretrained", [False, True])
def test_doubleunet_preserves_weight_selection_without_legacy_api(monkeypatch, backbone, pretrained):
    from torchvision import models
    from bicbioseg.models.doubleunet import BackboneEncoder

    weight_types = {
        "vgg19": models.VGG19_Weights,
        "resnet50": models.ResNet50_Weights,
        "resnet101": models.ResNet101_Weights,
    }
    requested = []

    def factory(*, weights):
        requested.append(weights)
        if backbone == "vgg19":
            return SimpleNamespace(features=torch.nn.Sequential(*[torch.nn.Identity() for _ in range(36)]))
        return SimpleNamespace(**{
            name: torch.nn.Identity()
            for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")
        })

    monkeypatch.setitem(BackboneEncoder.SUPPORTED_BACKBONES[backbone], "model_fn", factory)
    encoder = BackboneEncoder(backbone, pretrained=pretrained)
    assert requested == [weight_types[backbone].IMAGENET1K_V1 if pretrained else None]
    assert encoder.get_output_channels() == (512 if backbone == "vgg19" else 2048)
