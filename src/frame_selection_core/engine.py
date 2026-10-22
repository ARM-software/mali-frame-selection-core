# SPDX-FileCopyrightText: Copyright 2015-2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

"""Select representative frames from numerical features."""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from numbers import Real

import numpy as np
import pandas
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

from frame_selection_core.errors import FrameSelectionError

logger = logging.getLogger(__name__)

_DIAGNOSTIC_COLUMNS = {
    "source_frame_number",
    "filtered_frame_idx",
    "SOAD",
    "cluster",
    "valid",
    "cluster_size",
    "raw_weight",
    "activity_mean",
    "fixed_rate_weight",
    "fps",
    "fps_mean",
    "fixed_time_weight",
}


@dataclass(frozen=True)
class SelectionResult:
    """Original frame IDs, weights, and diagnostic tables for one selection.

    ``frames`` contains representatives in selection order. ``frame_range`` is
    the inclusive range before an optional frame list is applied. ``counters``
    holds analyzed features and weights. ``normalized`` holds standardized
    features, distances (``SOAD``), cluster labels and primary validity flags.
    A fallback representative can have ``valid=False``. Both tables retain
    source frame IDs; ``features`` lists the feature columns in their order.

    Fields cannot be reassigned, but the result's DataFrames remain mutable.
    Changing these tables does not change the caller's input.
    """

    frames: tuple[int, ...]
    frame_range: tuple[int, int]
    counters: pandas.DataFrame
    normalized: pandas.DataFrame
    features: tuple[str, ...]


def select_frames(
    features: pandas.DataFrame,
    *,
    clusters: int = 3,
    percentage: float = 10,
    frame_start: int = 0,
    frame_end: int | None = None,
    frame_numbers: Sequence[int] | None = None,
) -> SelectionResult:
    """Select stable representatives from clusters of numerical frame features.

    Args:
        features: Real-valued features indexed by unique integer frame IDs.
            Include a positive, finite ``activity`` column for weighting.
        clusters: Number of clusters; must not exceed the analyzed frame count.
        percentage: Maximum neighbor change, in percent (10 means 10%).
        frame_start: Inclusive lower bound; zero leaves the start unrestricted.
        frame_end: Inclusive upper bound, or None for the last available frame.
        frame_numbers: Optional subset of source IDs to analyze after filtering.

    Returns:
        Original frame IDs, weights and diagnostic tables. Representative order
        follows cluster selection order, with fallback representatives appended.
        Clusters with no stable representative are omitted.

    Raises:
        FrameSelectionError: Invalid options or unsuitable feature data.

    Input data is copied and sorted by source ID. Entirely zero or missing columns
    are removed; other missing values are zero-filled for clustering and weighting.
    """
    _validate_selection_options(clusters, percentage, frame_start, frame_end)
    _validate_feature_table(features)

    features = features.sort_index()
    features.index = pandas.Index(
        [int(frame) for frame in features.index], name="frame"
    )
    features, frame_range = filter_frames(
        features, frame_start, frame_end, frame_numbers, clusters
    )
    features = _prepare_features(features)
    counters = features.fillna(0)
    normalized = _normalize_features(counters)
    labels = KMeans(n_clusters=clusters, random_state=42, n_init=10).fit_predict(
        normalized
    )
    representatives = _select_representatives(
        features, normalized, labels, percentage / 100
    )

    if len(representatives) < clusters:
        logger.warning(
            "Fewer than %s representative frames met the %s%% neighbor tolerance.",
            clusters,
            percentage,
        )
        logger.info(
            "Try a cluster count lower than %s or a tolerance higher than %s.",
            clusters,
            percentage,
        )

    counters["source_frame_number"] = counters.index
    counters["filtered_frame_idx"] = range(len(counters))
    counters["cluster"] = labels
    compute_weights(counters)
    return SelectionResult(
        frames=tuple(int(frame) for frame in representatives),
        frame_range=frame_range,
        counters=counters,
        normalized=normalized,
        features=tuple(features.columns),
    )


def _validate_selection_options(
    clusters: int, percentage: float, frame_start: int, frame_end: int | None
) -> None:
    """Check selection options before inspecting the feature table."""
    if isinstance(clusters, bool) or not isinstance(clusters, Integral) or clusters < 1:
        raise FrameSelectionError("The cluster count must be a positive integer.")
    if (
        isinstance(percentage, bool)
        or not isinstance(percentage, Real)
        or not math.isfinite(percentage)
        or percentage < 0
    ):
        raise FrameSelectionError(
            "The percentage tolerance must be finite and non-negative."
        )
    for name, value in (("frame_start", frame_start), ("frame_end", frame_end)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Integral)
        ):
            raise FrameSelectionError(f"{name} must be an integer.")
    if frame_start is None:
        raise FrameSelectionError("frame_start must be an integer.")


def _validate_feature_table(features: pandas.DataFrame) -> None:
    """Check feature names, types, and source frame IDs before filtering."""
    if not isinstance(features, pandas.DataFrame) or features.empty:
        raise FrameSelectionError("The feature table must contain frames and counters.")
    if not features.columns.is_unique or not all(
        isinstance(c, str) for c in features.columns
    ):
        raise FrameSelectionError("Feature names must be unique strings.")
    if _DIAGNOSTIC_COLUMNS.intersection(features.columns):
        raise FrameSelectionError(
            "Feature names collide with selection diagnostic columns."
        )
    if "activity" not in features:
        raise FrameSelectionError("The feature table must contain an activity column.")
    if not features.index.is_unique or any(
        isinstance(frame, bool)
        or not isinstance(frame, Real)
        or not math.isfinite(frame)
        or int(frame) != frame
        for frame in features.index
    ):
        raise FrameSelectionError("Source frame IDs must be unique, finite integers.")
    if not all(
        pandas.api.types.is_numeric_dtype(dtype)
        and not pandas.api.types.is_complex_dtype(dtype)
        for dtype in features.dtypes
    ):
        raise FrameSelectionError("All features must be real-valued numeric data.")


def _validate_frame_numbers(
    frame_numbers: Sequence[int], available_frames: set[int]
) -> None:
    """Validate requested frames against the original capture population."""
    if any(
        isinstance(frame, bool) or not isinstance(frame, Integral)
        for frame in frame_numbers
    ):
        raise FrameSelectionError("Requested frame IDs must be integers.")
    missing_frames = [frame for frame in frame_numbers if frame not in available_frames]
    if missing_frames:
        raise FrameSelectionError(
            "The frame list contains frames that don't exist in the input data: "
            f"{missing_frames}"
        )


def filter_frames(
    df: pandas.DataFrame,
    frame_start: int,
    frame_end: int | None,
    frame_numbers: Sequence[int] | None,
    clusters: int,
) -> tuple[pandas.DataFrame, tuple[int, int]]:
    """Filter sorted frames by inclusive bounds and an optional list of IDs.

    Args:
        df: Nonempty table indexed by unique frame IDs in ascending order.
        frame_start: Inclusive lower bound; zero leaves the start unrestricted.
            A nonzero value must not precede the first frame in ``df``.
        frame_end: Inclusive upper bound, or None for the last available frame.
            If set, it must be greater than ``frame_start``.
        frame_numbers: Optional subset of IDs present in ``df``, applied after
            the range filter. Duplicates are ignored; input row order is kept.
        clusters: Minimum number of frames required after all filters.

    Returns:
        A copy of the filtered table and the inclusive bounds of the available
        frames after range filtering, before applying ``frame_numbers``.

    Raises:
        FrameSelectionError: Invalid bounds or frame IDs, no remaining frames,
            or fewer remaining frames than the requested cluster count.
    """
    available_frames = set(df.index)
    if frame_start:
        first_frame = int(df.index[0])
        if frame_start < first_frame:
            raise FrameSelectionError(
                f"frame_start ({frame_start}) is smaller than "
                f"the first frame ({first_frame})."
            )
        df = df[df.index >= frame_start]
    if frame_end is not None:
        if frame_start >= frame_end:
            raise FrameSelectionError(
                f"frame_start ({frame_start}) must be smaller than "
                f"frame_end ({frame_end})."
            )
        df = df[df.index <= frame_end]
    if df.empty:
        raise FrameSelectionError(
            "No frames remain after applying the frame range filters."
        )
    chosen_frame_range = (int(df.index[0]), int(df.index[-1]))
    if frame_numbers is not None:
        _validate_frame_numbers(frame_numbers, available_frames)
        df = df[df.index.isin(frame_numbers)]
    if df.empty:
        raise FrameSelectionError("No frames remain after applying the frame filters.")
    if len(df) < clusters:
        raise FrameSelectionError(
            f"The filtered frame count ({len(df)}) is smaller than "
            f"the requested cluster count ({clusters})."
        )
    return df.copy(), chosen_frame_range


def _prepare_features(features: pandas.DataFrame) -> pandas.DataFrame:
    """Validate analyzed values and remove features that carry no information."""
    if np.isinf(features.to_numpy(dtype=float, na_value=np.nan)).any():
        raise FrameSelectionError("Features contain infinite values.")
    activity = features["activity"].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(activity).all() or (activity <= 0).any():
        raise FrameSelectionError(
            "activity must be positive and finite for every analyzed frame."
        )
    empty = [name for name in features.columns if (features[name].fillna(0) == 0).all()]
    return features.drop(columns=empty)


def _normalize_features(features: pandas.DataFrame) -> pandas.DataFrame:
    """Scale by column maxima, then standardize without changing column order."""
    maxima = features.max()
    if (maxima == 0).any():
        raise FrameSelectionError("Nonempty features must have nonzero column maxima.")
    # Keep both stages: changing their order can change clustering through rounding.
    normalized = features / maxima
    if not np.isfinite(normalized.to_numpy(dtype=float)).all():
        raise FrameSelectionError("Feature scaling produced nonfinite values.")
    # Standardize owned arrays so pandas' read-only views are never mutated.
    return normalized.apply(
        lambda column: scale(column.to_numpy(dtype=np.float64, copy=True), copy=False),
        axis=0,
    ).fillna(0)


def _select_representatives(
    features: pandas.DataFrame,
    normalized: pandas.DataFrame,
    labels: NDArray[np.integer],
    threshold: float,
) -> list[int]:
    """Attach selection diagnostics to normalized and return representative IDs."""
    cluster_means = normalized.groupby(labels).transform("mean")
    # KMeans uses Euclidean distance; picking representatives uses Manhattan distance.
    normalized["SOAD"] = (normalized - cluster_means).abs().sum(axis=1)
    normalized["cluster"] = labels
    normalized["valid"] = compute_max_neighbor_change(features) <= threshold
    valid = normalized[normalized["valid"]]
    # idxmin breaks distance ties using the first (lowest) source frame ID.
    representatives = valid.groupby("cluster")["SOAD"].idxmin().to_list()

    missing_clusters = sorted(set(labels) - set(valid["cluster"].unique()))
    if missing_clusters:
        # Fallback uses zero-filled values and neighbors within the same cluster.
        same_cluster_valid = compute_same_cluster_validity(
            features.fillna(0), labels, threshold
        )
        fallback = normalized[
            same_cluster_valid & normalized["cluster"].isin(missing_clusters)
        ]
        representatives.extend(fallback.groupby("cluster")["SOAD"].idxmin().to_list())
    return representatives


def compute_max_neighbor_change(df: pandas.DataFrame) -> pandas.Series:
    """Return the largest relative feature change to two rows on either side.

    Neighbors are adjacent analyzed rows, even when source frame IDs have gaps.
    Missing comparisons are ignored; no comparable values yields NaN.
    """
    changes = [
        ((df / df.shift(offset)) - 1).abs().max(axis=1) for offset in (1, 2, -1, -2)
    ]

    return pandas.concat(changes, axis=1).max(axis=1)


def compute_same_cluster_validity(
    df: pandas.DataFrame, clusters: NDArray[np.integer], threshold: float
) -> pandas.Series:
    """Check same-cluster neighbors using a fractional threshold (0.1 means 10%)."""
    same_cluster_valid = pandas.Series(False, index=df.index)
    for _, cluster_df in df.groupby(clusters):
        cluster_max_change = compute_max_neighbor_change(cluster_df)
        same_cluster_valid.loc[cluster_df.index] = cluster_max_change <= threshold

    return same_cluster_valid


def compute_weights(df: pandas.DataFrame) -> None:
    """Add cluster sizes and activity-adjusted weights to the table in place."""
    # Promote arithmetic without changing the original activity column.
    activity = df["activity"].astype(np.float64)
    by_cluster = activity.groupby(df["cluster"])
    df["cluster_size"] = by_cluster.transform("size")
    df["raw_weight"] = df["cluster_size"] / len(df)
    df["activity_mean"] = by_cluster.transform("mean")
    # Weights cover the full analyzed population, including unrepresented clusters.
    df["fixed_rate_weight"] = (df["activity_mean"] / activity) * df["raw_weight"]

    df["fps"] = 1.0 / activity
    df["fps_mean"] = df["fps"].groupby(df["cluster"]).transform("mean")
    df["fixed_time_weight"] = df["fps_mean"] * activity * df["raw_weight"]
    if not np.isfinite(df[["fixed_rate_weight", "fixed_time_weight"]].to_numpy()).all():
        raise FrameSelectionError("Activity values produce nonfinite weights.")
