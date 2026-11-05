"""Audit retained native call flags; no solver or production modification."""
import json
import sys
from pathlib import Path
from audit_plastic_return_map import F, sha256


def main(tag):
    parent_path = F / 'results/output' / ('pipe_solver_' + tag + '.json')
    parent = json.loads(parent_path.read_text())
    assert parent['checks_passed']
    root = F / parent['run']
    for name, digest in parent['hashes'].items():
        assert sha256(root / name) == digest
    data = json.loads((root / 'assembly.json').read_text())
    calls = data['element_calls']
    assert calls and all(c['element'] == 1 for c in calls)
    source = F / 'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    text = source.read_text()
    signature = text.split('subroutine e_c3d(', 1)[1].split(')', 1)[0]
    names = signature.replace('&', '').replace('\n', '').replace(' ', '').split(',')
    expected = {12:'nelem',13:'nmethod',30:'iperturb',49:'mass',50:'stiffness',51:'buckling',52:'rhsi',55:'time',56:'istep',57:'iinc'}
    assert all(names[i-1] == name for i, name in expected.items())
    selected = [c for c in calls if abs(c['time'] - data['time']) < 1e-14]
    assert selected
    for c in selected:
        assert c['nmethod'] == 1 and c['stiffness'] == 1
        assert c['mass'] == 0 and c['rhsi'] == 1
        assert c['iperturb'][1] == 1
        assert 'mafillsm_' in c['stack']
    report = dict(checks_passed=True, parent_report=str(parent_path.relative_to(F)),
                  run=parent['run'], time=data['time'], calls=calls,
                  selected_call_count=len(selected), argument_positions_verified=expected,
                  source_references=['e_c3d.f:1482 force branch rhsi=1',
                                     'e_c3d.f:1625 distributed-load stiffness elseif mass=1',
                                     'nonlingeo.c:118 static defaults mass=0 stiffness=1 rhsi=1'],
                  provenance={str(p.relative_to(F)):sha256(p) for p in
                              (source, Path(__file__), parent_path, root/'assembly.json',
                               F/'sym/vendor/CalculiX/ccx_2.23/src/mafillsm.f',
                               F/'sym/vendor/CalculiX/ccx_2.23/src/nonlingeo.c')},
                  limitations='Element 1 call flags at one early elastic state only. Source branch and native flags explain exclusion of this follower-load stiffness branch, not a pressure-only matrix extraction or convergence causation. No constitutive, production, burst, experimental or Abaqus acceptance.')
    output=F/'results/output'/('pipe_assembly_flags_'+tag+'.json')
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main(sys.argv[1])
