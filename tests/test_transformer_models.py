import builtins

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from bicbioseg import Segmenter, SegmenterConfig
from bicbioseg.models.transunet import TransUNet, MultiHeadAttention


@pytest.fixture(autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def small_transunet(**kwargs):
    config = dict(out_channels=8, embedding_dim=24, head_num=3, block_num=1,
                  mlp_dim=32, decoder_channels=(16, 8, 4, 2), dropout=0)
    config.update(kwargs)
    return config


@pytest.mark.parametrize("encoder", ["custom", "resnet50"])
@pytest.mark.parametrize("n_skip", [0, 1, 2, 3])
def test_transunet_rectangular_gradients_and_dynamic_positions(encoder, n_skip):
    model = TransUNet(img_dim=(32, 48), in_channels=1, n_classes=3,
                     **small_transunet(encoder_name=encoder, n_skip=n_skip))
    # Different token grid from construction, including a larger height.
    logits = model(torch.rand(2, 1, 48, 64))
    assert logits.shape == (2, 3, 48, 64)
    torch.nn.functional.cross_entropy(logits, torch.randint(3, (2, 48, 64))).backward()
    for parameter in (model.encoder.vit.embedding, model.encoder.vit.projection.weight,
                      model.decoder.conv1.weight):
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("kwargs, message", [
    ({"out_channels": 0}, "positive integer"),
    ({"block_num": 0}, "positive integer"),
    ({"head_num": 0}, "positive integer"),
    ({"embedding_dim": 25}, "divisible"),
    ({"patch_dim": 8}, "fixed stride"),
    ({"img_dim": (32, 47)}, "divisible by 16"),
    ({"decoder_channels": [8, 4]}, "four"),
    ({"decoder_channels": [8, 4, 2, 0]}, "positive integer"),
    ({"dropout": 1}, "dropout"),
    ({"n_skip": 4}, "n_skip"),
    ({"encoder_name": "resnet18"}, "encoder_name"),
    ({"encoder_weights": "DEFAULT"}, "requires"),
    ({"encoder_weights": "missing.npz"}, "encoder_weights"),
])
def test_transunet_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TransUNet(**small_transunet(**kwargs))


def test_official_vit_weights_remain_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="Pretrained ViT"):
        TransUNet(pretrained_vit=True)


def test_transunet_attention_uses_inverse_sqrt_scale():
    attention = MultiHeadAttention(8, 2)
    x = torch.randn(2, 5, 8)
    qkv = attention.qkv_layer(x).reshape(2, 5, 4, 3, 2).permute(3, 0, 4, 1, 2)
    q, k, v = qkv.unbind(0)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    expected = attention.out_attention(expected.transpose(1, 2).reshape(2, 5, 8))
    torch.testing.assert_close(attention(x), expected)


@pytest.mark.parametrize("weights", ["DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2"])
def test_resnet_weights_are_requested_and_transferred(monkeypatch, weights):
    import torchvision.models as vision
    real_factory = vision.resnet50
    requested = []

    def factory(*, weights):
        requested.append(weights)
        backbone = real_factory(weights=None)
        with torch.no_grad():
            backbone.conv1.weight.fill_(0.125)
            backbone.layer3[0].conv1.weight.fill_(0.25)
        return backbone

    monkeypatch.setattr(vision, "resnet50", factory)
    model = TransUNet(**small_transunet(encoder_name="resnet50", encoder_weights=weights))
    assert requested == [vision.ResNet50_Weights[weights]]
    assert (model.encoder.stem[0].weight == 0.125).all()
    assert (model.encoder.layer3[0].conv1.weight == 0.25).all()


def test_grayscale_normalization_and_opt_out():
    from bicbioseg.models._common import ImageNetInput
    x = torch.rand(2, 1, 32, 48)
    normalize = ImageNetInput(1)
    torch.testing.assert_close(normalize(x), (x.repeat(1, 3, 1, 1) - normalize.mean) / normalize.std)
    torch.testing.assert_close(ImageNetInput(1, False)(x), x.repeat(1, 3, 1, 1))
    model = TransUNet(img_dim=(32, 48), in_channels=1, **small_transunet(normalize_input=True))
    assert model(x).shape == (2, 1, 32, 48)


@pytest.mark.parametrize("encoder", ["custom", "resnet50"])
def test_transunet_config_and_checkpoint_restore_offline(tmp_path, monkeypatch, encoder):
    config = SegmenterConfig(architecture="transunet", image_size=(32, 48), in_channels=1,
                             device="cpu", model_kwargs=small_transunet(encoder_name=encoder))
    config.save(tmp_path / "config.json")
    segmenter = Segmenter.from_config(SegmenterConfig.load(tmp_path / "config.json"))
    segmenter.model.eval()
    image = torch.rand(1, 1, 32, 48)
    with torch.no_grad():
        expected = segmenter.model(image)
    if encoder == "resnet50":
        # Simulate provenance from a previously pretrained run, without networking.
        segmenter.model_kwargs["encoder_weights"] = "IMAGENET1K_V2"
        import torchvision.models as vision
        factory = vision.resnet50

        def offline_factory(*, weights):
            assert weights is None, "Checkpoint restore must not download weights"
            return factory(weights=None)

        monkeypatch.setattr(vision, "resnet50", offline_factory)
    checkpoint = segmenter.save(tmp_path / "model.pt")
    restored = Segmenter.load(checkpoint, device="cpu")
    assert restored.model_kwargs == segmenter.model_kwargs
    with torch.no_grad():
        torch.testing.assert_close(restored.model(image), expected)


MODEL_CASES = [("deit", {"decoder_channels": (16, 8, 4, 2)}),
               ("deit", {"decoder_channels": (16, 8, 4, 2), "distilled": True}),
               ("swin_unet", {"decoder_channels": 8}),
               ("pvt_unet", {"decoder_channels": 8})]


@pytest.mark.parametrize("name, kwargs", MODEL_CASES)
@pytest.mark.parametrize("channels, classes", [(1, 3), (3, 1)])
def test_transformer_models_exact_shapes_and_gradients(name, kwargs, channels, classes):
    pytest.importorskip("timm")
    segmenter = Segmenter(architecture=name, image_size=(35, 49), in_channels=channels,
                          num_classes=classes, device="cpu", model_kwargs=kwargs)
    logits = segmenter.model(torch.rand(2, channels, 35, 49))
    assert logits.shape == (2, classes, 35, 49)
    assert torch.isfinite(logits).all()
    targets = torch.randint(classes, (2, 35, 49)) if classes > 1 else torch.rand_like(logits)
    loss = segmenter.loss_fn(logits, targets)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in segmenter.model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in segmenter.model.decoder.parameters())


@pytest.mark.parametrize("name, kwargs", MODEL_CASES)
def test_transformer_pretraining_dispatch_freezing_and_checkpoint(tmp_path, monkeypatch, name, kwargs):
    timm = pytest.importorskip("timm")
    factory = timm.create_model
    calls = []

    def offline_factory(model_name, *, pretrained, **options):
        calls.append(pretrained)
        return factory(model_name, pretrained=False, **options)

    monkeypatch.setattr(timm, "create_model", offline_factory)
    segmenter = Segmenter(architecture=name, image_size=(35, 49), in_channels=1, device="cpu",
                          model_kwargs={**kwargs, "pretrained": True, "freeze_encoder": True})
    model = segmenter.model
    model.train()
    assert not model.encoder.training
    assert all(not p.requires_grad for p in model.encoder.parameters())
    model(torch.rand(1, 1, 35, 49)).mean().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters())
    model.eval()
    image = torch.rand(1, 1, 35, 49)
    with torch.no_grad():
        expected = model(image)
    segmenter.save(tmp_path / "model.pt")
    restored = Segmenter.load(tmp_path / "model.pt", device="cpu")
    assert calls == [True, False]
    assert restored.model_kwargs == segmenter.model_kwargs
    with torch.no_grad():
        torch.testing.assert_close(restored.model(image), expected)


def test_optional_dependency_error_is_actionable(monkeypatch):
    original_import = builtins.__import__

    def without_timm(name, *args, **kwargs):
        if name == "timm":
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_timm)
    with pytest.raises(ImportError, match=r"bicbioseg\[transformers\]"):
        Segmenter(architecture="deit", device="cpu")
    assert {"deit", "swin_unet", "pvt_unet"} <= set(Segmenter.available_models())


@pytest.mark.parametrize("name", ["deit", "swin_unet", "pvt_unet"])
def test_new_models_reject_unsupported_variant_without_download(name):
    with pytest.raises(ValueError, match="variant"):
        Segmenter(architecture=name, device="cpu", model_kwargs={"variant": "typo", "pretrained": True})
