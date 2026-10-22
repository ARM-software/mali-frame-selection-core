<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
<open-source-office@arm.com>
SPDX-License-Identifier: MIT
-->

# Mali frame selection core

A Python library for choosing representative frames from a workload. It groups
frames with similar performance measurements and selects stable frames from each
group. It also calculates weights that describe how much of the workload each
selected frame represents.

## Installation

Python 3.12 or newer is required.

To add the package to a uv project:

```sh
uv add mali-frame-selection-core
```

Or install it with pip in a virtual environment:

```sh
python -m pip install mali-frame-selection-core
```

To install from a local checkout instead, run this in the repository directory:

```sh
python -m pip install .
```

## Usage

Provide a pandas DataFrame with one row per frame. The index contains unique
integer frame IDs, and the columns contain numerical measurements. Include an
`activity` column with positive, finite values in consistent units.

```python
import pandas

from frame_selection_core import select_frames

features = pandas.DataFrame(
    {"activity": [100, 100, 200, 200], "metric": [1, 1, 4, 4]},
    index=[101, 103, 105, 107],
)

result = select_frames(features, clusters=2)
print(result.frames)
print(
    result.counters.loc[
        list(result.frames), ["fixed_rate_weight", "fixed_time_weight"]
    ]
)
```

The result contains the selected frame IDs, weights and diagnostic tables. The
input table is not modified. Weights account for each representative's activity;
they are not probabilities and do not necessarily add up to one.

The following optional arguments control selection:

| Argument | Default | Description |
| --- | --- | --- |
| `clusters` | `3` | Number of groups to form. |
| `percentage` | `10` | Maximum change between neighboring analyzed frames, in percent. |
| `frame_start` | `0` | Inclusive lower frame bound; zero leaves the start unrestricted. |
| `frame_end` | `None` | Inclusive upper frame bound; must be greater than the start. |
| `frame_numbers` | `None` | Analyze only the listed frame IDs within the selected range. |

Fewer representatives may be returned if some groups have no stable frame.
Invalid inputs or activity values that produce nonfinite weights raise
`FrameSelectionError`.

## Development

Install the development tools in a virtual environment:

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```

## License

This project is licensed under the [MIT License](LICENSE).
