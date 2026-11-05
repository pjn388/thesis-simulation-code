"""Guard removal of alternative backends at the CLI and public entry points."""
import json

import pytest

from pipe_sim.cli import main
from pipe_sim.config import Case
from pipe_sim.decks import write_deck
from pipe_sim.results import extract
from pipe_sim.runner import case_solver_path, collect_external, generate, run, solver_path


@pytest.mark.parametrize('backend', ['unsupported', ''])
def test_backend_rejected_before_outputs_are_created(tmp_path, backend):
    output = tmp_path / 'run'
    for operation in (
        lambda: solver_path(backend),
        lambda: case_solver_path(Case(), backend),
        lambda: generate(Case(), output, backend),
        lambda: write_deck(Case(), None, output, backend),
    ):
        with pytest.raises(ValueError, match='Only CalculiX'):
            operation()
    assert not output.exists()


@pytest.mark.parametrize('operation', [run, collect_external, extract])
def test_foreign_manifest_is_rejected_without_modification(tmp_path, operation):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'solver': 'unsupported', 'files': {}}))
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match='Only CalculiX'):
        operation(tmp_path)
    assert manifest.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['manifest.json']


@pytest.mark.parametrize('command', ['generate', 'run', 'matrix'])
def test_cli_has_no_backend_switch(tmp_path, command):
    with pytest.raises(SystemExit) as error:
        main([command, 'example.json', '--out', str(tmp_path / 'run'), '--solver', 'unsupported'])
    assert error.value.code == 2


def test_reference_data_must_be_supplied():
    with pytest.raises(SystemExit) as error:
        main(['validate', 'example-run', '--specimen', 'example'])
    assert error.value.code == 2
