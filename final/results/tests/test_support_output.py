import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from review_support_output import complete_node_blocks


def test_requires_full_unique_coverage_header_termination():
    header = ' displacements (vx,vy,vz) for set SUPPORT_AUDIT and time 0.25\n'
    end = ' forces (fx,fy,fz) for set HISTORY and time 0.25\n'
    expected = {'SUPPORT_AUDIT': {1, 2}}
    rows = [' 1 0 0 0\n', ' 2 0 0 0\n']
    assert len(list(complete_node_blocks([header, *rows, end], expected))) == 1
    assert not list(complete_node_blocks([header, *rows], expected))
    assert not list(complete_node_blocks([header, rows[0], end], expected))
    assert not list(complete_node_blocks([header, *rows, rows[0], end], expected))
    assert not list(complete_node_blocks([header, *rows, ' 3 0 0 0\n', end], expected))


def test_terminal_eof_requires_full_unique_coverage():
    header = ' forces (fx,fy,fz) for set SUPPORT_AUDIT and time 0.25\n'
    expected = {'SUPPORT_AUDIT': {1, 2}}
    rows = [' 1 0 0 0\n', ' 2 0 0 0\n']
    assert len(list(complete_node_blocks([header, *rows], expected, allow_eof=True))) == 1
    assert not list(complete_node_blocks([header, rows[0]], expected, allow_eof=True))
    assert not list(complete_node_blocks([header, *rows, rows[0]], expected, allow_eof=True))
    assert not list(complete_node_blocks([header, *rows, ' 3 0 0 0\n'], expected, allow_eof=True))
