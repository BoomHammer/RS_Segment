"""Check default confidence output through the streaming prediction pipeline."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from rasterio.windows import Window


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize(
    "constraint_mode", ["disabled", "nodata", "exclude_restricted", "all_restricted"]
)
@pytest.mark.parametrize("row_block", [1, 2])
def test_default_confidence(
    tmp_path, monkeypatch, override, constraint_mode, row_block
):
    spec = importlib.util.spec_from_file_location(
        "confidence_predict", Path(__file__).parents[1] / "predict.py"
    )
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    for name in ("sample_index.json", "raster_stats.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "label_mapping.json").write_text(
        json.dumps(
            {
                "classes": [
                    {
                        "alliance_code": 1,
                        "formation": "Grassland",
                        "alliance": "Meadow",
                        "alliance_zh": "高寒草甸",
                    },
                    {"alliance_code": 2, "formation": "Grassland", "alliance": "Other"},
                ]
            }
        ),
        encoding="utf-8",
    )
    grid = SimpleNamespace(
        width=3, height=2, crs="EPSG:3857", transform=from_origin(100, 500, 250, 250)
    )

    class Dataset:
        def __len__(self):
            return 2

    dataset = Dataset()
    dataset.grid = grid
    dataset.ground_truth_pixels = {(0, 0): 2, (0, 1): 1, (1, 0): 1, (1, 2): 1}
    dem_path = tmp_path / "dem.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="float32",
        crs=grid.crs,
        transform=grid.transform,
        nodata=-9999,
    ) as dem:
        dem.write(
            np.array([[2799, 2800, 2801], [-9999, 3000, 2700]], dtype=np.float32), 1
        )
    constraint_config = {
        "enabled": constraint_mode != "disabled",
        "dem_path": str(dem_path),
        "classes": ["Grassland"]
        if constraint_mode == "all_restricted"
        else ["高寒草甸"],
        "min_elevation_m": 2800,
        "missing_dem": "exclude_restricted"
        if constraint_mode == "exclude_restricted"
        else "nodata",
    }
    monkeypatch.setattr(predict, "_load_run", lambda _: tmp_path)
    monkeypatch.setattr(
        predict,
        "load_config",
        lambda _: SimpleNamespace(
            data=SimpleNamespace(
                label_columns={}, raster={}, stage2={}, output_nodata=-1
            )
        ),
    )
    monkeypatch.setattr(predict, "WindowedSampleDataset", lambda *a, **kw: dataset)
    monkeypatch.setattr(predict, "Subset", lambda *a: dataset)
    monkeypatch.setattr(predict, "SequentialSampler", lambda _: None)
    # Two overlapping windows disagree. The shared column must use blended
    # probabilities, not the average of each window's maximum confidence.
    batches = []
    for column, probability in ((0, 0.9), (1, 0.2)):
        window = Window(column, 0, 2, 2)
        valid = torch.ones((1, 2, 2), dtype=torch.bool)
        if column == 1:
            valid[0, 1, 1] = False
        batches.append(
            {
                "window": [window],
                "input_window": [window],
                "valid_mask": valid,
                "probability": probability,
            }
        )
    monkeypatch.setattr(predict, "DataLoader", lambda *a, **kw: batches)
    monkeypatch.setattr(
        predict.torch,
        "load",
        lambda *a, **kw: {"contract": {"derived": {"num_classes": 2}}, "model": {}},
    )
    monkeypatch.setattr("models.config.validate_checkpoint_mapping", lambda *a: None)

    class Model(torch.nn.Module):
        def forward(self, batch):
            p = batch["probability"]
            return {
                "fine_probability": torch.tensor([p, 1 - p])
                .reshape(1, 2, 1, 1)
                .expand(1, 2, 2, 2)
            }

    monkeypatch.setattr(predict.SegFormerUtae, "from_contract", lambda _: Model())
    output = tmp_path / "vegetation.tif"
    mapping = tmp_path / "vegetation.csv"
    assert predict.predict(
        checkpoint,
        output,
        mapping,
        window_size=(2, 2),
        stride=(1, 1),
        device="cpu",
        amp_enabled=False,
        write_row_block=row_block,
        override_ground_truth=override,
        elevation_constraint=constraint_config,
    ) == (output, mapping)
    confidence = tmp_path / "vegetation_confidence.tif"
    with rasterio.open(output) as labels, rasterio.open(confidence) as scores:
        assert scores.crs == labels.crs
        assert scores.transform == labels.transform
        assert scores.shape == labels.shape
        assert scores.dtypes == ("float32",)
        assert scores.nodata == -9999
        expected_scores = [[0.1 if override else 0.9, 0.55, 0.8], [0.9, 0.55, -9999]]
        expected_labels = [[2 if override else 1, 1, 2], [1, 1, 1 if override else -1]]
        if constraint_mode in {"nodata", "exclude_restricted"}:
            exclude = constraint_mode == "exclude_restricted"
            expected_scores = [
                [0.1, 0.45, 0.8],
                [0.1 if exclude else -9999, 0.55, -9999],
            ]
            expected_labels = [[2, 2, 2], [2 if exclude else -1, 1, -1]]
        elif constraint_mode == "all_restricted":
            expected_scores = [[-9999, -9999, 0.8], [-9999, 0.55, -9999]]
            expected_labels = [[-1, -1, 2], [-1, 1, -1]]
        np.testing.assert_allclose(scores.read(1), expected_scores, atol=1e-6)
        np.testing.assert_array_equal(labels.read(1), expected_labels)
        if constraint_mode != "disabled":
            expected_skips = 4 if constraint_mode == "all_restricted" else 3
            assert int(labels.tags()["skipped_ground_truth_overrides"]) == (
                expected_skips if override else 0
            )
    # A leftover confidence sidecar must receive the same overwrite protection.
    output.unlink()
    mapping.unlink()
    with pytest.raises(FileExistsError):
        predict.predict(checkpoint, output, mapping)
