# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas
import pytest

import frame_selection_core
from frame_selection_core import FrameSelectionError
from frame_selection_core import select_frames
from frame_selection_core.engine import compute_max_neighbor_change


def test_plain_feature_table_retains_source_ids():
    features = pandas.DataFrame(
        {"activity": [100, 100, 200, 200], "generic_activity": [1, 1, 4, 4]},
        index=[101, 103, 105, 107],
    )
    result = select_frames(features.iloc[::-1], clusters=2)
    assert set(result.frames) == {101, 105}
    assert result.counters.loc[list(result.frames), "fixed_rate_weight"].tolist() == [
        0.5,
        0.5,
    ]


@pytest.mark.parametrize("start", [2**63, 2**64 - 4])
def test_unsigned_frame_ids_are_preserved(start):
    frame_ids = list(range(start, start + 4))
    features = pandas.DataFrame(
        {"activity": [100, 100, 200, 200]},
        index=pandas.Index(frame_ids, dtype="uint64"),
    )
    result = select_frames(features.iloc[::-1], clusters=2)
    assert set(result.frames) == {frame_ids[0], frame_ids[2]}
    assert result.frame_range == (frame_ids[0], frame_ids[-1])
    assert result.counters.index.tolist() == frame_ids
    assert result.counters["source_frame_number"].tolist() == frame_ids
    assert result.counters["source_frame_number"].dtype == features.index.dtype


@pytest.mark.parametrize("active", [[0, 100], [-1, 100], [np.nan, 100], [np.inf, 100]])
def test_invalid_activity_raises_library_error(active):
    with pytest.raises(FrameSelectionError):
        select_frames(pandas.DataFrame({"activity": active}), clusters=1)


@pytest.mark.parametrize("index", [[1, 1], [1, np.nan], [1, 1.5], ["1", "2"]])
def test_invalid_frame_ids_raise_library_error(index):
    with pytest.raises(FrameSelectionError, match="frame IDs"):
        select_frames(
            pandas.DataFrame({"activity": [100, 100]}, index=index), clusters=1
        )


@pytest.mark.parametrize(
    "options",
    [
        {"clusters": 0},
        {"clusters": True},
        {"clusters": 1.5},
        {"clusters": 3},
        {"percentage": -1},
        {"percentage": np.inf},
        {"percentage": np.nan},
        {"percentage": True},
        {"frame_numbers": [999]},
        {"frame_numbers": []},
        {"frame_numbers": [True]},
        {"frame_numbers": [1.5]},
        {"frame_start": 100},
        {"frame_start": None},
        {"frame_start": True},
        {"frame_end": 1.5},
        {"frame_start": 2, "frame_end": 1},
    ],
)
def test_invalid_selection_options(options):
    features = pandas.DataFrame({"activity": [100, 100]}, index=[1, 2])
    with pytest.raises(FrameSelectionError):
        select_frames(features, **({"clusters": 1} | options))


def test_nonfinite_feature_is_not_silently_filled():
    with pytest.raises(FrameSelectionError, match="infinite"):
        select_frames(
            pandas.DataFrame({"activity": [100, 100], "metric": [1, np.inf]}),
            clusters=1,
        )


def test_neighbor_changes_preserve_missing_values_and_zero_denominators():
    features = pandas.DataFrame(
        {"metric": [1.0, np.nan, 2.0, 0.0, 0.0]}, index=[10, 20, 30, 40, 50]
    )
    expected = pandas.Series([0.5, np.nan, np.inf, 1.0, 1.0], index=features.index)
    pandas.testing.assert_series_equal(compute_max_neighbor_change(features), expected)


def test_missing_feature_fill_and_empty_feature_removal():
    features = pandas.DataFrame(
        {"activity": [100] * 4, "metric": [1, np.nan, 1, 1], "unused": [0] * 4}
    )
    result = select_frames(features, clusters=1)
    assert result.features == ("activity", "metric")
    assert result.counters.loc[1, "metric"] == 0


@pytest.mark.parametrize("dtype", ["Float32", "Float64"])
@pytest.mark.parametrize("missing", [False, True])
def test_nullable_float_features_match_numpy(dtype, missing):
    values = [1, None if missing else 1, 1, 1]
    features = pandas.DataFrame(
        {"activity": [100] * 4, "metric": pandas.Series(values, dtype=dtype)}
    )
    original = features.copy(deep=True)
    expected = select_frames(features.astype(float), clusters=1)
    result = select_frames(features, clusters=1)
    assert result.frames == expected.frames
    pandas.testing.assert_frame_equal(
        result.normalized, expected.normalized, check_dtype=False
    )
    pandas.testing.assert_frame_equal(features, original)


@pytest.mark.parametrize(
    "features",
    [
        [],
        pandas.DataFrame(),
        pandas.DataFrame({"metric": [1, 2]}),
        pandas.DataFrame([[100, 100]], columns=["activity", "activity"]),
        pandas.DataFrame({"activity": [100, 100], 1: [1, 2]}),
        pandas.DataFrame({"activity": [100, 100], "cluster": [1, 2]}),
        pandas.DataFrame({"activity": ["100", "200"]}),
        pandas.DataFrame({"activity": [100, 200], "metric": [1 + 2j, 3 + 4j]}),
    ],
    ids=[
        "not-a-table",
        "empty",
        "missing-activity",
        "duplicate-columns",
        "non-string-column",
        "reserved-column",
        "non-numeric-values",
        "complex-values",
    ],
)
def test_invalid_feature_tables_raise_library_error(features):
    with pytest.raises(FrameSelectionError):
        select_frames(features, clusters=1)


@pytest.mark.parametrize(
    "metric, message",
    [
        ([-1, 0], "nonzero column maxima"),
        ([-1e308, -1e-308], "scaling produced nonfinite"),
    ],
)
def test_invalid_scaling_reports_a_library_error(metric, message):
    features = pandas.DataFrame({"activity": [100, 100], "metric": metric})
    with pytest.raises(FrameSelectionError, match=message):
        select_frames(features, clusters=1)


def test_filtering_keeps_source_ids_and_range_before_frame_list():
    features = pandas.DataFrame({"activity": [100] * 6}, index=[10, 20, 30, 40, 50, 60])
    result = select_frames(
        features,
        clusters=1,
        frame_start=20,
        frame_end=50,
        frame_numbers=[50, 30, 10],
    )
    assert result.frame_range == (20, 50)
    assert result.counters.index.tolist() == [30, 50]
    assert result.frames == (30,)


def test_same_cluster_fallback_preserves_primary_validity_flags():
    features = pandas.DataFrame(
        {"activity": [100, 200] * 4, "metric": [1, 4] * 4},
        index=range(10, 26, 2),
    )
    result = select_frames(features, clusters=2)
    assert set(result.frames) == {10, 12}
    assert not result.normalized["valid"].any()
    assert result.counters.loc[list(result.frames), "fixed_rate_weight"].tolist() == [
        0.5,
        0.5,
    ]


def test_unrepresented_cluster_keeps_its_share_of_the_weights():
    features = pandas.DataFrame({"activity": [100] * 6 + [4000]})
    result = select_frames(features, clusters=2)
    assert result.frames == (0,)
    assert result.counters.loc[0, "fixed_rate_weight"] == pytest.approx(6 / 7)
    assert result.counters.loc[0, "fixed_time_weight"] == pytest.approx(6 / 7)


@pytest.mark.parametrize(
    "dtype", ["int64", "float16", "float32", "float64", "Int64", "Float32", "Float64"]
)
def test_activity_adjusted_weights_are_not_normalized_probabilities(dtype):
    features = pandas.DataFrame({"activity": pandas.Series([100, 200], dtype=dtype)})
    result = select_frames(features, clusters=1, percentage=100)
    assert result.frames == (0,)
    assert result.counters["cluster_size"].dtype == np.dtype("int64")
    assert result.counters["fixed_rate_weight"].tolist() == pytest.approx([1.5, 0.75])
    assert result.counters["fixed_time_weight"].tolist() == pytest.approx([0.75, 1.5])


@pytest.mark.parametrize("dtype, activity", [("float16", 1e-5), ("float32", 1e38)])
def test_weight_calculation_uses_sufficient_precision(dtype, activity):
    features = pandas.DataFrame(
        {"activity": pandas.Series([activity] * 4, dtype=dtype)}
    )
    result = select_frames(features, clusters=1)
    assert result.frames == (0,)
    assert result.counters["fixed_rate_weight"].tolist() == pytest.approx([1] * 4)
    assert result.counters["fixed_time_weight"].tolist() == pytest.approx([1] * 4)
    pandas.testing.assert_series_equal(
        result.counters["activity"], features["activity"], check_names=False
    )


@pytest.mark.parametrize("activity", [1e-310, 1e308])
def test_unrepresentable_weights_raise_library_error(activity):
    features = pandas.DataFrame({"activity": [activity] * 4})
    with pytest.raises(FrameSelectionError, match="nonfinite weights"):
        select_frames(features, clusters=1)


def test_selection_is_repeatable_and_does_not_mutate_the_input():
    features = pandas.DataFrame(
        {"activity": [100] * 4 + [200] * 4, "metric": [1] * 4 + [4] * 4},
        index=range(18, 10, -1),
    )
    original = features.copy(deep=True)
    first = select_frames(features, clusters=2)
    second = select_frames(features, clusters=2)
    assert first.frames == second.frames
    pandas.testing.assert_frame_equal(first.counters, second.counters)
    pandas.testing.assert_frame_equal(first.normalized, second.normalized)
    first.counters.iloc[0, 0] = -1
    first.normalized.iloc[0, 0] = -1
    pandas.testing.assert_frame_equal(features, original)


def test_import_does_not_configure_logging_or_load_plotting(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(frame_selection_core.__file__).parents[1])
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import logging
import sys

before = logging.getLogger().handlers[:]
import frame_selection_core
assert logging.getLogger().handlers == before
assert 'matplotlib' not in sys.modules
""",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
    )
