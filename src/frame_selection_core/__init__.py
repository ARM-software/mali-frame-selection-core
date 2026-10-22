# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates
# <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

"""Numerical frame selection and weighting."""

from frame_selection_core.engine import SelectionResult
from frame_selection_core.engine import select_frames
from frame_selection_core.errors import FrameSelectionError

__all__ = ["FrameSelectionError", "SelectionResult", "select_frames"]
