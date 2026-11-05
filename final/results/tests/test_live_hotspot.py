import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from review_live_hotspot import selected_blocks, diagnostic_frames
import pytest


def test_snapshot_without_crossing_keeps_latest_pressure():
    rows = [dict(stage='indent', max_peeq=.9),
            dict(stage='pressurise', max_peeq=.5),
            dict(stage='pressurise', max_peeq=.6)]
    pressure, crossing, selected = diagnostic_frames({'history': rows}, .8)
    assert pressure == rows[1:]
    assert crossing is None
    assert selected == [rows[-1]]


def test_sampled_pressure_crossing_is_selected_without_pressure_comparison():
    rows = [dict(stage='pressurise', max_peeq=.7),
            dict(stage='pressurise', max_peeq=.8),
            dict(stage='pressurise', max_peeq=.9)]
    _, crossing, selected = diagnostic_frames({'history': rows}, .8)
    assert crossing == rows[1]
    assert selected == [rows[1], rows[-1]]
    with pytest.raises(ValueError, match='no accepted pressure'):
        diagnostic_frames({'history': []}, .8)


def test_only_complete_header_terminated_pipe_blocks(tmp_path):
    path = tmp_path/'job.dat'
    path.write_text('stresses for set DEFECT and time 1.0\n'
                    ' 12 1 1D+0 2 3 4 5 6\n 12 2 7 8 9 10 11 12\n'
                    'equivalent plastic strain for set PIPE and time 1.0\n'
                    ' 12 1 .2\n 12 2 .3\n'
                    'stresses for set DEFECT and time 2.0\n'
                    ' 12 1 1 2 3 4 5 6\n'
                    'equivalent plastic strain for set PIPE and time 2.0\n'
                    ' 12 1 .4\n 12 2 .5\n')
    evidence = {}
    blocks = selected_blocks(path, {(12, 1)}, {'S': 2, 'PEEQ': 2}, evidence)
    assert blocks == {(1., 'S'): {(12, 1): [1., 2., 3., 4., 5., 6.]},
                      (1., 'PEEQ'): {(12, 1): [.2]}}
    assert evidence['job.dat']['prefix_bytes'] == path.stat().st_size


def test_other_sets_do_not_supply_pipe_fields(tmp_path):
    path = tmp_path/'job.dat'
    path.write_text('stresses for set TOOL and time 1.0\n'
                    ' 12 1 1 2 3 4 5 6\n'
                    'displacements for set HISTORY and time 1.0\n')
    assert selected_blocks(path, {(12, 1)}, {'S': 1, 'PEEQ': 1}, {}) == {}
