from __future__ import annotations

import pytest

from evaluation.prepare_local_model import _percentile, _summary


def test_percentile_uses_fixed_nearest_rank_index() -> None:
    values = [5, 1, 3, 2, 4]
    assert _percentile(values, 0.50) == 3
    assert _percentile(values, 0.95) == 4
    assert _percentile(values, 1.00) == 5


def test_summary_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        _summary([])
