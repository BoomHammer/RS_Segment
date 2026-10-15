import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from data.labels import LabelRecord
from data.raster_alignment import target_grid_from_raster
from data.weak_labels import (
    WeakLabelGenerationConfig,
    evaluate_label_quality_from_raster,
    generate_weak_labels,
)
from inference.pointsam import PointSAMPrediction


class FakeInferencer:
    def predict(self, request):
        height, width = request.image.shape[-2:]
        mask = np.zeros((height, width), dtype=bool)
        point = request.positive_points[0]
        mask[point.row, point.column] = True
        return PointSAMPrediction(mask=mask, confidence=np.ones((height, width)))


class FailingInferencer(FakeInferencer):
    def __init__(self, fail_after: int) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def predict(self, request):
        self.calls += 1
        if self.calls > self.fail_after:
            raise RuntimeError("simulated interruption")
        return super().predict(request)


class CountingInferencer(FakeInferencer):
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, request):
        self.calls += 1
        return super().predict(request)


def _write_band(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 4, 1, 1),
        nodata=-9999,
    ) as dataset:
        dataset.write(values.astype(np.float32), 1)


def test_streaming_weak_labels_write_geo_tiff_and_quality(tmp_path: Path) -> None:
    bands = []
    for index in range(3):
        path = tmp_path / f"band_{index}.tif"
        _write_band(path, np.ones((4, 4), dtype=np.float32))
        bands.append(path)
    grid = target_grid_from_raster(bands[0], target_crs="EPSG:4326", resolution=1)
    records = [LabelRecord("1", 0.5, 3.5, "F", "A", "F", "A", 1, 7)]
    output = tmp_path / "weak_labels.tif"
    report = generate_weak_labels(
        records,
        grid=grid,
        image_paths=bands,
        inferencer=FakeInferencer(),
        output_path=output,
        config=WeakLabelGenerationConfig(window_size=(4, 4)),
        alliance_names={7: "Test alliance"},
        provenance_path=tmp_path / "seed_ids.tif",
        sample_outcomes_path=tmp_path / "outcomes.json",
    )
    with rasterio.open(output) as dataset:
        assert dataset.crs.to_string() == "EPSG:4326"
        assert dataset.nodata == -9999
        assert dataset.read(1)[0, 0] == 7
    assert report["labeled_pixels"] == 1
    assert report["sample_quality"]["invalid_samples"] == 0
    assert report["class_distribution"]["7"]["alliance"] == "Test alliance"
    with rasterio.open(tmp_path / "seed_ids.tif") as seeds:
        assert seeds.read(1)[0, 0] == 1
        assert (seeds.read(1) > 0).sum() == 1
    outcomes = json.loads((tmp_path / "outcomes.json").read_text())
    assert outcomes[0]["record_index"] == "1"
    assert outcomes[0]["seed_id"] == 1


def test_quality_report_detects_conflicting_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.tif"
    _write_band(path, np.array([[1, -1], [0, -9999]], dtype=np.float32))
    report = evaluate_label_quality_from_raster(path)
    assert report["invalid_pixel_count"] == 1
    assert report["valid_pixels"] == 3


def test_weak_label_generation_resumes_after_interruption(tmp_path: Path) -> None:
    bands = []
    for index in range(3):
        path = tmp_path / f"resume_band_{index}.tif"
        _write_band(path, np.ones((4, 4), dtype=np.float32))
        bands.append(path)
    grid = target_grid_from_raster(bands[0], target_crs="EPSG:4326", resolution=1)
    records = [
        LabelRecord("1", 0.5, 3.5, "F", "A", "F", "A", 1, 7),
        LabelRecord("2", 3.5, 0.5, "F", "A", "F", "A", 1, 7),
    ]
    output = tmp_path / "resumed.tif"
    checkpoint = output.with_name(f"{output.name}.checkpoint")
    config = WeakLabelGenerationConfig(window_size=(4, 4))

    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate_weak_labels(
            records,
            grid=grid,
            image_paths=bands,
            inferencer=FailingInferencer(fail_after=1),
            output_path=output,
            config=config,
        )

    assert checkpoint.is_dir()
    assert not output.exists()
    with (checkpoint / "completed.jsonl").open("ab") as stream:
        stream.write(b'{"interrupted"')
    inferencer = CountingInferencer()
    generate_weak_labels(
        records,
        grid=grid,
        image_paths=bands,
        inferencer=inferencer,
        output_path=output,
        config=config,
        resume=True,
    )

    assert inferencer.calls == 1
    assert not checkpoint.exists()
    with rasterio.open(output) as dataset:
        labels = dataset.read(1)
    assert labels[0, 0] == 7
    assert labels[3, 3] == 7


def test_weak_label_resume_rejects_changed_contract(tmp_path: Path) -> None:
    bands = []
    for index in range(3):
        path = tmp_path / f"contract_band_{index}.tif"
        _write_band(path, np.ones((4, 4), dtype=np.float32))
        bands.append(path)
    grid = target_grid_from_raster(bands[0], target_crs="EPSG:4326", resolution=1)
    records = [LabelRecord("1", 0.5, 3.5, "F", "A", "F", "A", 1, 7)]
    output = tmp_path / "contract.tif"

    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate_weak_labels(
            records,
            grid=grid,
            image_paths=bands,
            inferencer=FailingInferencer(fail_after=0),
            output_path=output,
            config=WeakLabelGenerationConfig(window_size=(4, 4)),
        )

    with pytest.raises(ValueError, match="不匹配"):
        generate_weak_labels(
            records,
            grid=grid,
            image_paths=bands,
            inferencer=FakeInferencer(),
            output_path=output,
            config=WeakLabelGenerationConfig(window_size=(3, 3)),
            resume=True,
        )

    manifest = output.with_name(f"{output.name}.checkpoint") / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["version"] = 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="旧版"):
        generate_weak_labels(
            records,
            grid=grid,
            image_paths=bands,
            inferencer=FakeInferencer(),
            output_path=output,
            config=WeakLabelGenerationConfig(window_size=(4, 4)),
            resume=True,
        )


@pytest.fixture
def conflict_inputs(tmp_path):
    bands = [tmp_path / f"conflict_band_{index}.tif" for index in range(3)]
    for band in bands:
        _write_band(band, np.ones((4, 4), dtype=np.float32))
    return {
        "grid": target_grid_from_raster(bands[0], target_crs="EPSG:4326", resolution=1),
        "image_paths": bands,
        "config": WeakLabelGenerationConfig(window_size=(4, 4)),
    }


def _conflicting_records(codes):
    return [
        LabelRecord(str(index), 1.5, 2.5, "F", str(code), "F", str(code), 1, code)
        for index, code in enumerate(codes)
    ]


@pytest.mark.parametrize("codes", [(1, 2, 3), (1, 2, 1)])
def test_generated_conflicts_survive_order_and_duplicates(
    tmp_path, conflict_inputs, codes
):
    for case, order in enumerate(sorted(set(permutations(codes)))):
        output = tmp_path / f"conflict_{case}.tif"
        provenance = tmp_path / f"provenance_{case}.tif"
        score = tmp_path / f"score_{case}.tif"
        candidate_paths = (
            tmp_path / f"candidates_{case}.tif",
            tmp_path / f"candidate_scores_{case}.tif",
        )
        generate_weak_labels(
            _conflicting_records(order),
            **conflict_inputs,
            inferencer=FakeInferencer(),
            output_path=output,
            provenance_path=provenance,
            score_output_path=score,
            candidate_output_paths=candidate_paths,
        )
        for path in (output, provenance, score):
            with rasterio.open(path) as source:
                assert source.read(1)[1, 1] == source.nodata
        with rasterio.open(candidate_paths[0]) as source:
            assert source.read()[:, 1, 1].tolist() == [1, 2]
        with rasterio.open(candidate_paths[1]) as source:
            assert source.read()[:, 1, 1].tolist() == [1.0, 1.0]


def test_conflicting_evidence_survives_checkpoint_resume(tmp_path, conflict_inputs):
    records = _conflicting_records((1, 2, 3, 1))
    output = tmp_path / "conflict_resume.tif"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate_weak_labels(
            records,
            **conflict_inputs,
            inferencer=FailingInferencer(fail_after=2),
            output_path=output,
        )
    checkpoint = output.with_name(f"{output.name}.checkpoint")
    assert (checkpoint / "second_scores.bin").is_file()
    inferencer = CountingInferencer()
    generate_weak_labels(
        records,
        **conflict_inputs,
        inferencer=inferencer,
        output_path=output,
        resume=True,
    )
    assert inferencer.calls == 2
    assert not checkpoint.exists()
    clean = tmp_path / "conflict_clean.tif"
    generate_weak_labels(
        records, **conflict_inputs, inferencer=FakeInferencer(), output_path=clean
    )
    with rasterio.open(output) as resumed, rasterio.open(clean) as fresh:
        np.testing.assert_array_equal(resumed.read(), fresh.read())
        assert resumed.read(1)[1, 1] == resumed.nodata


class ScoredInferencer(FakeInferencer):
    def __init__(self, scores):
        self.scores = iter(scores)

    def predict(self, request):
        prediction = super().predict(request)
        # Without mask logits the generator multiplies confidence by itself.
        confidence = np.sqrt(next(self.scores))
        return PointSAMPrediction(
            mask=prediction.mask,
            confidence=np.full(prediction.mask.shape, confidence, dtype=np.float32),
        )


@pytest.mark.parametrize(
    "codes, scores, expected",
    [
        ((1, 2, 3, 1), (0.9, 0.92, 0.75, 0.96), -9999),
        ((1, 2, 3), (0.9, 0.9, 0.99), 3),
    ],
)
def test_generated_shards_and_recovery_match_single_run(
    tmp_path, conflict_inputs, monkeypatch, codes, scores, expected
):
    import scripts.recover_weak_labels as recovery
    from scripts.weak_label import _merge_weak_label_shards

    records = _conflicting_records(codes)
    single = tmp_path / "single.tif"
    generate_weak_labels(
        records,
        **conflict_inputs,
        inferencer=ScoredInferencer(scores),
        output_path=single,
    )
    shards = []
    # A locally ambiguous shard, another shard with new evidence, and an empty shard.
    for rank, indices in enumerate((range(2), range(2, len(records)), ())):
        shard = {
            "labels": tmp_path / f".weak_labels.tif.rank{rank}",
            "candidate_labels": tmp_path
            / f".weak_labels.tif.rank{rank}.candidates.tif",
            "candidate_scores": tmp_path
            / f".weak_labels.tif.rank{rank}.candidate_scores.tif",
            "outcomes": tmp_path / f".weak_labels.tif.rank{rank}.outcomes.json",
        }
        generate_weak_labels(
            [records[index] for index in indices],
            **conflict_inputs,
            inferencer=ScoredInferencer([scores[index] for index in indices]),
            output_path=shard["labels"],
            candidate_output_paths=(
                shard["candidate_labels"],
                shard["candidate_scores"],
            ),
            sample_outcomes_path=shard["outcomes"],
        )
        shards.append(shard)
    merged = tmp_path / "weak_labels.tif"
    for ordered in (shards, list(reversed(shards))):
        _merge_weak_label_shards(ordered, merged, nodata=-9999, conflict_margin=0.05)
        with rasterio.open(single) as reference, rasterio.open(merged) as actual:
            np.testing.assert_array_equal(actual.read(), reference.read())
            assert actual.read(1)[1, 1] == expected
    mapping = {"classes": [{"alliance_code": code} for code in sorted(set(codes))]}
    (tmp_path / "label_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    monkeypatch.setattr(
        recovery, "write_label_quality_visualization", lambda *a, **kw: None
    )
    report = recovery.recover(tmp_path, len(shards), 0.05)
    assert report["recovery_audit"]["merge_mismatch_pixels"] == 0
    with rasterio.open(merged, "r+") as raster:
        raster.write(np.full((4, 4), 1, dtype=np.int32), 1)
    with pytest.raises(ValueError, match="differs from rank merge"):
        recovery.recover(tmp_path, len(shards), 0.05)
    with rasterio.open(shards[0]["candidate_scores"], "r+") as raster:
        raster.write(np.full((4, 4), np.nan, dtype=np.float32), 1)
    with pytest.raises(ValueError, match="无效分数"):
        _merge_weak_label_shards(shards, merged, nodata=-9999, conflict_margin=0.05)
