"""Fixed homogeneous F path with nested native grids; no production revision."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from audit_plastic_return_map import F, sha256, tensors, update
from review_plastic_state_capture import CAP, state_audit

TAG = 'prescribed_metric_coupon_20261011T180008Z'
ROOT = F/'sym/runs/investigation'/TAG
XYZ = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
MU = 202500/(2*1.3)
BULK = 202500/(3*.4)


def audit(run, divisions, curve):
    state_gate = state_audit(run)
    assert state_gate['accepted_frames'] == divisions
    records = json.loads((run/'fields.json').read_text())['records']
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    metric = np.broadcast_to(np.eye(3), (1,8,3,3)).copy()
    peeq = np.zeros((1,8))
    old_c = metric.copy(); old_s = np.zeros_like(metric); work = 0.
    maxima = dict(stress_mpa=0., metric=0., peeq=0., native_work_n_mm=0.,
                  prescribed_f=0., reconstructed_c=0.)
    history = []
    for i, rec in enumerate(records, 1):
        assert abs(rec['time']-i/divisions) < 1e-10
        def field(name, width):
            meta = rec['fields'][name]
            values = raw[meta['offset']//8:meta['offset']//8+meta['count']]
            return values.reshape((-1,width)) if name == 'u' else values.reshape((1,8,width))
        u = field('u',4)[:,1:]
        coef = np.linalg.lstsq(np.c_[np.ones(8),XYZ],u,rcond=None)[0]
        deformation = np.eye(3)+coef[1:].T
        expected = np.diag(1+rec['time']*np.array([.02,-.01,-.01]))
        c = np.broadcast_to(deformation.T@deformation, metric.shape)
        captured_c = np.eye(3)+2*tensors(field('strain',6))
        inverse = np.linalg.inv(deformation)
        actual_s = np.linalg.det(deformation)*(inverse@tensors(field('stress',6))@inverse.T)
        stress, metric, peeq, _ = update(c,metric,peeq,curve,MU,BULK)
        state = field('state',rec['states'])
        actual_m = np.eye(3)-2*tensors(state[...,1:7])
        work += float((.25*np.einsum('...ij,...ij->...',stress+old_s,c-old_c)).mean())
        native_work = float(field('energy',2)[...,0].mean())
        errors = dict(stress_mpa=float(abs(stress-actual_s).max()),
            metric=float(abs(metric-actual_m).max()), peeq=float(abs(peeq-state[...,0]).max()),
            native_work_n_mm=abs(work-native_work), prescribed_f=float(abs(deformation-expected).max()),
            reconstructed_c=float(abs(c-captured_c).max()))
        for key, value in errors.items(): maxima[key] = max(maxima[key],value)
        history.append(dict(time=rec['time'],reference_change_n_mm=float(
            (1.5*MU*(np.cbrt(np.linalg.det(actual_m))-1)).mean()),
            native_work_n_mm=native_work, maximum_peeq=float(peeq.max()),
            metric_determinant=float(np.linalg.det(actual_m).mean())))
        old_c, old_s = c, stress
    # Retain prior replay limits; trajectory checks are independent algebra gates.
    limits = dict(stress_mpa=1e-4, metric=1e-8, peeq=1e-8,
                  native_work_n_mm=1e-8, prescribed_f=1e-12, reconstructed_c=1e-12)
    report = dict(run=str(run.relative_to(F)), divisions=divisions, state_gate=state_gate,
        maximum_errors=maxima, fixed_limits=limits,
        replay_checks_passed=all(maxima[k]<v for k,v in limits.items()),
        within_recovered_material_extent=history[-1]['maximum_peeq']<.197798202029,
        history=history, hashes={n:sha256(run/n) for n in
            ('job.inp','job.dat','job.sta','solver.log','fields.bin','fields.json')})
    assert report['replay_checks_passed'] and report['within_recovered_material_extent'], report
    return report


def run():
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    assert len(subprocess.run(['pgrep','-x','ccx_2.23_pardis'],capture_output=True,text=True).stdout.splitlines())<2
    assert shutil.disk_usage(F).free>25*1024**3
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines()
                         if s.startswith('MemAvailable:')))
    assert available>13048*1024
    exe = F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    gate = F/'results/output/midpoint_reference_20261011T173006Z.json'
    assert json.loads(gate.read_text())['identity_checks_passed']
    parent = F/'sym/runs/investigation/material_coupon/job.inp'
    # Preserve cube geometry and the shared A37 material; replace fixture and loading.
    base = parent.read_text().split('*BOUNDARY')[0]
    lines = base.splitlines(); curve = []
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'): break
        curve.append(list(map(float,line.split(','))))
    curve = np.array(curve)
    ROOT.mkdir(exist_ok=False)
    manifest = dict(parent=str(parent.relative_to(F)),parent_sha256=sha256(parent),
        solver_sha256=sha256(exe),capture_sha256=sha256(CAP),gate_sha256=sha256(gate),
        script_sha256=sha256(__file__),dependencies={n:sha256(F/'results/scripts'/n) for n in
        ('audit_plastic_return_map.py','review_plastic_state_capture.py')},
        source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'),
        trajectory='F(t)=diag(1+0.02t,1-0.01t,1-0.01t),0<=t<=1; all24DOFs prescribed',
        acceptance='Same exact F/C path and initial identity metric/zeroPEEQ; nested20/40/80 grids; unchanged native stress/metric/PEEQ/work replay gates. Report reference sensitivity separately from physical power.',
        limitations='Artificial homogeneous constrained coupon, not tensile experimental reproduction. No unloading, physical dissipation, asymptotic convergence, pipe/cap/contact/MPC/Abaqus or burst acceptance. No production change.',
        threads=1,native_timeout_seconds=120)
    (ROOT/'diagnostic_revision.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reports = []
    for divisions in (20,40,80):
        assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
        directory = ROOT/f'grid_{divisions}'; directory.mkdir()
        dt = 1/divisions
        deck = base+f'*STEP, NLGEOM, INC=1000\n*STATIC\n{dt},1.,1.e-8,{dt}\n*BOUNDARY\n'
        target = XYZ*np.array([.02,-.01,-.01])
        for node in range(8):
            for dof in range(3):
                deck += f'{node+1},{dof+1},{dof+1},{target[node,dof]:.17g}\n'
        deck += '*NODE PRINT, NSET=ALL\nU, RF\n*EL PRINT, ELSET=CUBE\nS, E, PEEQ, ELSE\n*END STEP\n'
        (directory/'job.inp').write_text(deck)
        (directory/'capture.gdb').write_text(f'set pagination off\nset confirm off\nsource {CAP}\nquit\n')
        env = dict(os.environ,THESIS_INTERNAL_BINARY=str(directory/'fields.bin'),
            THESIS_INTERNAL_INDEX=str(directory/'fields.json'),MKL_THREADING_LAYER='GNU',
            **{n:'1' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CCX_NPROC_RESULTS',
                              'CCX_NPROC_STIFFNESS','CCX_NPROC_EQUATION_SOLVER')})
        with (directory/'solver.log').open('w') as log:
            result = subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],
                cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        log = (directory/'solver.log').read_text()
        assert result.returncode==0 and 'exited normally' in log and '*ERROR' not in log and 'Python Exception' not in log
        reports.append(audit(directory,divisions,curve))
    endpoints = [r['history'][-1] for r in reports]
    differences = {key:[endpoints[i+1][key]-endpoints[i][key] for i in range(2)]
        for key in ('reference_change_n_mm','native_work_n_mm','maximum_peeq')}
    report = dict(**manifest,records=reports,nested_endpoint_differences=differences,
        differences_decrease={k:abs(v[1])<abs(v[0]) for k,v in differences.items()},
        identity_checks_passed=True,physical_acceptance=False)
    out = F/'results/output'/f'{TAG}.json'; assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),differences=differences,
                          differences_decrease=report['differences_decrease'])))


if __name__=='__main__': run()
