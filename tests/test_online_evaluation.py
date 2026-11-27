"""Online benchmarks use the standalone evaluation's data and score contract."""

import pytest

from social_jev.evaluation import OnlineEvaluation


def test_online_evaluation_requires_repeatable_data():
    with pytest.raises(TypeError, match="fresh iterator"):
        OnlineEvaluation(data=iter([{"state": "A", "label": 0}]))
