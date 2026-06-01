import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from bicbioseg import Segmenter


def test_segmenter_train_logs_and_loads_checkpoint(tmp_path):
    images = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    images[:, 4:12, 4:12, :] = 180
    masks = np.zeros((2, 16, 16), dtype=np.uint8)
    masks[:, 4:12, 4:12] = 255

    segmenter = Segmenter(
        architecture="unet",
        loss="dice",
        metrics=["dice", "iou"],
        image_size=(16, 16),
        num_classes=1,
        device="cpu",
        model_kwargs={"base_channels": 2, "num_decoder_blocks": 1},
    )

    history = segmenter.train(
        data=(images, masks),
        epochs=1,
        batch_size=1,
        num_workers=0,
        experiment_dir=tmp_path,
        run_name="smoke",
        verbose=False,
    )

    run_dir = tmp_path / "smoke"
    assert "train_loss" in history
    assert (run_dir / "config.json").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "final_model.pt").exists()

    config = json.loads((run_dir / "config.json").read_text())
    assert config["architecture"] == "unet"
    assert config["loss"] == "dice"

    restored = Segmenter.load(run_dir / "final_model.pt", device="cpu")
    assert restored.architecture == "unet"
    assert restored.history["train_loss"]

    history_plot = run_dir / "history.png"
    fig = segmenter.plot_history(save_to=history_plot, show=False)
    assert fig is not None
    assert history_plot.exists()

    history_file_plot = run_dir / "history_from_file.png"
    fig = Segmenter.plot_history_file(
        run_dir / "history.json",
        metrics=["train_loss"],
        save_to=history_file_plot,
        show=False,
    )
    assert fig is not None
    assert history_file_plot.exists()

    training_plot = run_dir / "training_samples.png"
    fig = segmenter.plot_training_samples(
        data=(images, masks),
        batch_size=1,
        num_samples=1,
        num_workers=0,
        save_to=training_plot,
        show=False,
    )
    assert fig is not None
    assert training_plot.exists()

    inference_dir = tmp_path / "inference_images"
    inference_dir.mkdir()
    image_path = inference_dir / "cell.png"
    cv2.imwrite(str(image_path), images[0])

    prediction_paths = segmenter.inference(
        images=inference_dir,
        save_to=run_dir / "predictions",
        save_overlay=True,
        save_probability=True,
        save_logits=True,
        save_contours=True,
    )
    assert prediction_paths
    assert (run_dir / "predictions" / "overlays" / "cell_overlay.png").exists()
    assert (run_dir / "predictions" / "probabilities" / "cell_probability.npy").exists()
    assert (run_dir / "predictions" / "logits" / "cell_logits.npy").exists()
    assert (run_dir / "predictions" / "contours" / "cell_contours.png").exists()

    inference_plot = run_dir / "inference_results.png"
    fig = segmenter.plot_inference_results(
        images=inference_dir,
        predictions=prediction_paths,
        num_samples=1,
        save_to=inference_plot,
        show=False,
    )
    assert fig is not None
    assert inference_plot.exists()

    tiled_outputs = segmenter.inference_large_image(
        image=image_path,
        save_to=run_dir / "large_prediction",
        patch_size=(8, 8),
        overlap=2,
        save_probability=True,
    )
    assert "mask" in tiled_outputs
    assert "overlay" in tiled_outputs
    assert "probability" in tiled_outputs

    mask_dir = tmp_path / "eval_masks"
    mask_dir.mkdir()
    cv2.imwrite(str(mask_dir / "cell.png"), masks[0])
    evaluation = segmenter.evaluate(
        images=inference_dir,
        masks=mask_dir,
        save_to=run_dir / "evaluation",
    )
    assert evaluation["num_samples"] == 1
    assert (run_dir / "evaluation" / "evaluation.csv").exists()
    assert (run_dir / "evaluation" / "evaluation_summary.json").exists()


def test_run_experiment_writes_summary(tmp_path):
    images = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    masks = np.zeros((2, 16, 16), dtype=np.uint8)
    masks[:, 4:12, 4:12] = 255

    results = Segmenter.run_experiment(
        dataset=(images, masks),
        architectures=["unet"],
        losses=["dice"],
        epochs=1,
        batch_size=1,
        image_size=(16, 16),
        model_kwargs={"base_channels": 2, "num_decoder_blocks": 1},
        output_dir=tmp_path,
        num_workers=0,
        verbose=False,
    )

    assert "unet_dice" in results
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()

    comparison_plot = tmp_path / "comparison.png"
    fig = Segmenter.compare_experiments(
        tmp_path,
        metric="train_loss",
        save_to=comparison_plot,
        show=False,
    )
    assert fig is not None
    assert comparison_plot.exists()


def test_early_stopping(tmp_path):
    images = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    masks = np.zeros((2, 16, 16), dtype=np.uint8)

    segmenter = Segmenter(
        architecture="unet",
        loss="dice",
        image_size=(16, 16),
        device="cpu",
        model_kwargs={"base_channels": 2, "num_decoder_blocks": 1},
    )

    history = segmenter.train(
        data=(images, masks),
        epochs=3,
        batch_size=1,
        num_workers=0,
        early_stopping=True,
        patience=0,
        monitor="train_loss",
        verbose=False,
    )

    assert len(history["train_loss"]) <= 3
