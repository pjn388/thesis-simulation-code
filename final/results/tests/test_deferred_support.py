import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from deferred_support_diagnostic import admission


def test_followup_requires_vacant_lane_capacity_resources_and_no_pause():
    assert admission(False, 1, 13048, 25, False)
    assert not admission(True, 1, 20000, 100, False)
    assert not admission(False, 2, 20000, 100, False)
    assert not admission(False, 1, 13047, 100, False)
    assert not admission(False, 1, 20000, 24, False)
    assert not admission(False, 1, 20000, 100, True)
