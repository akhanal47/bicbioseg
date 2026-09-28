import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
from bicbioseg import Segmenter


CASES = [("swin_unet_full", {}), ("pvtformer_full", {"variant":"b0", "decoder_channels":8}),
         ("resunetplusplus_full", {"base_channels":4}), ("unext_full", {"variant":"small"})]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name,kwargs", CASES)
@pytest.mark.parametrize("channels,classes", [(1,3),(3,1)])
def test_full_architectures_shape_auxiliary_gradients_and_roundtrip(tmp_path, name, kwargs, channels, classes):
    if name in {"swin_unet_full", "pvtformer_full"}:
        pytest.importorskip("timm")
    model = Segmenter(architecture=name, device="cpu", image_size=(35,49), in_channels=channels,
                       num_classes=classes, model_kwargs={**kwargs,"deep_supervision":True})
    x = torch.rand(2,channels,35,49)
    labels = torch.randint(classes, (2,35,49)) if classes > 1 else torch.randint(2,(2,35,49))
    output, auxiliary = model._forward_predictions(x)
    assert output.shape == (2,classes,35,49) and len(auxiliary) == 2
    model._training_options = {"aux_loss_weights":[.3,.2]}
    model._prediction_loss(output, auxiliary, model._prepare_targets(labels)).backward()
    for module in (model.model.encoder, model.model.decoder, model.model.aux_heads):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)
    model.model.eval()
    with torch.no_grad():
        expected = model.model(x)
    assert isinstance(expected, torch.Tensor)
    model.save(tmp_path / "full.pt")
    restored = Segmenter.load(tmp_path / "full.pt", device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(restored.model(x), expected)
        assert restored.model(torch.rand(1,channels,64,96)).shape == (1,classes,64,96)


@pytest.mark.parametrize("name,kwargs", CASES[:2])
def test_full_transformer_pretraining_freezing_offline_restore(tmp_path, monkeypatch, name, kwargs):
    timm = pytest.importorskip("timm")
    create = timm.create_model
    requested = []
    def offline(model_name, *, pretrained, **kwargs):
        requested.append(pretrained)
        return create(model_name, pretrained=False, **kwargs)
    monkeypatch.setattr(timm, 'create_model', offline)
    model = Segmenter(architecture=name, device='cpu', image_size=(32,32),
        model_kwargs={**kwargs, 'pretrained':True, 'freeze_encoder':True})
    model.model.train()
    assert not model.model.encoder.training
    model.model(torch.rand(2,3,32,32)).mean().backward()
    assert all(p.grad is None for p in model.model.encoder.parameters())
    assert any(p.grad is not None for p in model.model.decoder.parameters())
    model.save(tmp_path / 'pretrained.pt')
    restored = Segmenter.load(tmp_path / 'pretrained.pt', device='cpu')
    assert requested == [True,False]
    assert restored.model_kwargs['pretrained']


def test_patch_expansion_spatial_order():
    from bicbioseg.models.swin_unet_full import PatchExpand
    layer = PatchExpand(1,1,2)
    layer.norm = torch.nn.Identity()
    with torch.no_grad():
        layer.projection.weight.copy_(torch.tensor([[1.],[2.],[3.],[4.]]))
    output = layer(torch.tensor([[[[1.],[10.]]]]))
    torch.testing.assert_close(output[0,:,:,0],torch.tensor([[1.,2.,10.,20.],[3.,4.,30.,40.]]))


def test_shifted_mlp_does_not_wrap_across_image_edges():
    from bicbioseg.models.unext_full import ShiftedMLP
    block = ShiftedMLP(5)
    image = torch.zeros(1,5,5,5)
    image[:,:,0,0] = 1
    shifted = block.shift(image, 2)
    assert not shifted[..., -1, :].any()
    assert shifted[0,4,2,0] == 1


@pytest.mark.parametrize("name,kwargs", [("swin_unet_full",{"decoder_depths":[0,2,2]}),
                                          ("pvtformer_full",{"variant":"no"}),
                                          ("resunetplusplus_full",{"base_channels":0}),
                                          ("unext_full",{"widths":[1,2]})])
def test_invalid_new_model_settings(name, kwargs):
    with pytest.raises(ValueError):
        Segmenter(architecture=name, device='cpu', model_kwargs=kwargs)


@pytest.mark.parametrize('backend', ['cuda','mps'])
@pytest.mark.parametrize('name,kwargs', CASES)
def test_full_model_native_accelerator_backward(backend, name, kwargs):
    if backend not in Segmenter.available_devices():
        pytest.skip(f'{backend} hardware unavailable')
    if name in {'swin_unet_full','pvtformer_full'}:
        pytest.importorskip('timm')
    segmenter = Segmenter(architecture=name, device=backend, image_size=(35,49), in_channels=1,
                         num_classes=3, ignore_index=255, model_kwargs={**kwargs,'deep_supervision':True})
    batch = (torch.rand(2,1,35,49),torch.randint(3,(2,35,49)))
    batch[1][..., :3] = 255
    assert segmenter.validate_setup(batch, backward=True)['ok']
    segmenter.model.eval()
    with torch.no_grad():
        output = segmenter.model(batch[0].to(backend)).cpu()
    cpu = Segmenter(architecture=name, device='cpu', image_size=(35,49), in_channels=1,
                    num_classes=3, ignore_index=255, model_kwargs={**kwargs,'deep_supervision':True})
    cpu.model.load_state_dict({key: value.cpu() for key, value in segmenter.model.state_dict().items()})
    cpu.model.eval()
    with torch.no_grad():
        expected = cpu.model(batch[0])
    torch.testing.assert_close(output,expected,atol=5e-4,rtol=5e-3)
