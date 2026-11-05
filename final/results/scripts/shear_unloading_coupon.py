"""Exact-volume homogeneous shear discriminator; independent native runs."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from audit_plastic_return_map import F, sha256, tensors, update
from review_plastic_state_capture import CAP, state_audit
from audit_frozen_metric_work import elastic

TAG = 'shear_unloading_coupon_20261012T003001Z'
ROOT = F/'sym/runs/investigation'/TAG
XYZ = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
MU = 202500/(2*1.3)
BULK = 202500/(3*.4)


def audit(run, divisions, curve):
    state_gate = state_audit(run)
    assert state_gate['accepted_frames'] == divisions
    records = json.loads((run/'fields.json').read_text())['records']
    sta=[line.split() for line in (run/'job.sta').read_text().splitlines()[2:] if line.strip()]
    assert len(sta)==divisions
    assert all(int(row[0])==1 and int(row[1])==i and abs(float(row[4])-records[i-1]['time'])<1e-6 for i,row in enumerate(sta,1))
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    metric = np.broadcast_to(np.eye(3), (1,8,3,3)).copy()
    peeq = np.zeros((1,8))
    initial_potential = float(elastic(metric,metric,MU,BULK)[0].mean())
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
        fraction = 2*rec['time'] if rec['time']<=.5 else 1-.04*(rec['time']-.5)
        expected = np.eye(3); expected[0,1] = .03*fraction
        assert abs(np.linalg.det(deformation)-1)<1e-12
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
        history.append(dict(time=rec['time'],path_fraction=fraction,
            second_piola_stress_mpa=stress[0,0].tolist(),metric=metric[0,0].tolist(),
            candidate_potential_change_n_mm=float(elastic(c,metric,MU,BULK)[0].mean())-initial_potential,
            reference_change_n_mm=float(
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
        within_recovered_material_extent=max(h['maximum_peeq'] for h in history)<.197798202029,
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
    gate = F/'results/output/metric_flow_identity_20261012T003001Z.json'
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
        ('audit_plastic_return_map.py','review_plastic_state_capture.py','audit_frozen_metric_work.py','audit_metric_flow_identity.py')},
        source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'),
        trajectory='F=I+.03a e1 tensor e2; exact totalJ1; a=2t until .5 then1-.04(t-.5); all24DOFs prescribed',
        acceptance='Unique accepted times40/80/160, totalJ1 within1e-12; bounded unloading1 to.98; unchanged stress/metric/PEEQ/work replay limits; elastic unloading retains metric within1e-10 and PEEQ within1e-12. Compare residual stress and common-F nested states. Require recovered work versus source-integrable fixed-state potential error decreasing across40/80/160 and finest below1e-6Nmm. Candidate potential is not physical storage or dissipation.',
        limitations='Artificial homogeneous constrained coupon, not tensile experimental reproduction. No physical dissipation, asymptotic convergence, pipe/cap/contact/MPC/Abaqus or burst acceptance. No production change.',
        threads=1,native_timeout_seconds=120)
    (ROOT/'diagnostic_revision.json').write_text(json.dumps(manifest,indent=2)+'\n')
    reports = []
    for divisions in (40,80,160):
        assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
        directory = ROOT/f'grid_{divisions}'; directory.mkdir()
        dt = 1/divisions
        deck = base+'*AMPLITUDE, NAME=PATH\n0.,0.,.5,1.,1.,.98\n'+f'*STEP, NLGEOM, INC=1000\n*STATIC\n{dt},1.,1.e-8,{dt}\n*BOUNDARY, AMPLITUDE=PATH\n'
        target = np.zeros_like(XYZ,dtype=float); target[:,0] = .03*XYZ[:,1]
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
    comparisons=[]
    for child in reports:
        previous=child['history'][child['divisions']//2-1]
        intervals=[]
        start=previous.copy()
        recovered_errors=[]
        for current in child['history'][child['divisions']//2:]:
            dm=float(abs(np.array(current['metric'])-np.array(previous['metric'])).max())
            dp=current['maximum_peeq']-previous['maximum_peeq']
            assert abs(dp)<1e-12 and dm<1e-10, (current['time'],dp,dm)
            assert abs(current['maximum_peeq']-start['maximum_peeq'])<1e-12
            assert abs(np.array(current['metric'])-np.array(start['metric'])).max()<1e-10
            intervals.append(dict(time=current['time'],metric_change=dm,peeq_change=dp))
            recovered_errors.append(abs((current['native_work_n_mm']-start['native_work_n_mm'])-(current['candidate_potential_change_n_mm']-start['candidate_potential_change_n_mm'])))
            previous=current
        comparisons.append(dict(divisions=child['divisions'],elastic_unloading_intervals=intervals, maximum_work_potential_error_n_mm=max(recovered_errors), recovered_work_n_mm=previous['native_work_n_mm']-start['native_work_n_mm'], potential_change_n_mm=previous['candidate_potential_change_n_mm']-start['candidate_potential_change_n_mm']))
    matched=[]
    for coarse,fine in zip(reports,reports[1:]):
        errors=dict(stress_mpa=0.,metric=0.,peeq=0.,native_work_n_mm=0.,candidate_potential_change_n_mm=0.)
        for i,a in enumerate(coarse['history']):
            b=fine['history'][2*i+1]
            assert abs(a['time']-b['time'])<1e-10
            if a['time']<=.5: continue
            for key,field in (('stress_mpa','second_piola_stress_mpa'),('metric','metric'),('peeq','maximum_peeq'),('native_work_n_mm','native_work_n_mm'),('candidate_potential_change_n_mm','candidate_potential_change_n_mm')):
                errors[key]=max(errors[key],float(abs(np.array(a[field])-np.array(b[field])).max()))
        matched.append(dict(coarse=coarse['divisions'],fine=fine['divisions'],maximum_common_unloading_differences=errors))
    endpoints = [r['history'][-1] for r in reports]
    differences = {key:[endpoints[i+1][key]-endpoints[i][key] for i in range(2)]
        for key in ('reference_change_n_mm','native_work_n_mm','maximum_peeq')}
    work_errors=[c['maximum_work_potential_error_n_mm'] for c in comparisons]
    assert work_errors[2]<work_errors[1]<work_errors[0] and work_errors[2]<1e-6, work_errors
    report = dict(**manifest,records=reports,parent_loading_elastic_checks=comparisons,matched_unloading=matched,nested_endpoint_differences=differences,
        differences_decrease={k:abs(v[1])<abs(v[0]) for k,v in differences.items()},
        identity_checks_passed=True,physical_acceptance=False, frozen_state_work_checks_passed=True, work_potential_errors_n_mm=work_errors)
    out = F/'results/output'/f'{TAG}.json'; assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),differences=differences,
                          differences_decrease=report['differences_decrease'])))


if __name__=='__main__': run()
