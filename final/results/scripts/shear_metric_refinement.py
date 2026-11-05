"""Bounded refinement of exact-J shear; no change to production or old gates."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from shear_unloading_coupon import F, CAP, audit, sha256

TAG = 'shear_metric_refinement_20261012T010000Z'
ROOT = F/'sym/runs/investigation'/TAG
GATE = F/'results/output/shear_unloading_recovery_20261012T003001Z.json'
OUT = F/'results/output'/f'{TAG}.json'


def admitted():
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    assert len(subprocess.run(['pgrep','-x','ccx_2.23_pardis'],capture_output=True,text=True).stdout.splitlines())<2
    assert shutil.disk_usage(F).free>25*1024**3
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
    assert available>13048*1024


def convention(record):
    old=np.eye(3); first_sum=0.; remainder=0.; maximum=0.; reconstruction=0.
    for h in record['history']:
        f=np.eye(3); f[0,1]=.03*h['path_fraction']; c=f.T@f
        new=np.array(h['metric']); delta=new-old
        direction=np.trace(c@old)/3*np.linalg.inv(c)-old
        norm=np.sum(direction**2)
        beta=float(np.sum(delta*direction)/norm) if norm>1e-30 else 0.
        error=float(abs(delta-beta*direction).max())
        reconstruction=max(reconstruction,error)
        first=float(np.trace(np.linalg.solve(old,delta)))
        factor=float(np.trace(c@old)/3*np.trace(np.linalg.solve(old,np.linalg.inv(c)))-3)
        assert beta>=-1e-12 and factor>=-1e-12
        assert error<1e-12 and abs(first-beta*factor)<1e-12
        # Documented M=Cp^-1 implies d(log Jp)=-tr(M^-1 dM)/2.
        # C:dM=0 in the implementation is a different trace constraint.
        assert abs(np.sum(c*delta))<1e-12
        exact=float(np.linalg.slogdet(new)[1]-np.linalg.slogdet(old)[1])
        first_sum+=first; remainder+=exact-first; maximum=max(maximum,first)
        old=new
    logdet=float(np.linalg.slogdet(old)[1])
    assert abs(first_sum+remainder-logdet)<1e-12
    return dict(sum_first_order_logdet=first_sum,finite_step_remainder=remainder,
        maximum_first_order_logdet_increment=maximum,maximum_update_reconstruction_error=reconstruction,
        implied_plastic_jacobian=float(np.linalg.det(old)**(-.5)))


def run():
    admitted()
    retained=json.loads(GATE.read_text())
    assert retained['immutable_replay_checks_passed']
    assert not retained['original_monotonic_work_gate_passed']
    source=F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    manual=F/'sym/vendor/CalculiX/ccx_2.23/doc/ccx/node161.html'
    assert 'plastic flow is isochoric' in manual.read_text()
    assert 'inverse plastic right Cauchy-Green tensor' in manual.read_text()
    assert sha256(source)==retained['manifest']['source_sha256']
    parent=F/retained['records'][-1]['run']
    for r in retained['records']:
        assert all(sha256(F/r['run']/n)==v for n,v in r['hashes'].items())
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==retained['manifest']['solver_sha256']
    text=(parent/'job.inp').read_text(); lines=text.splitlines(); curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'): break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve)
    ROOT.mkdir(exist_ok=False)
    manifest=dict(parent=str(parent.relative_to(F)),parent_sha256=sha256(parent/'job.inp'),
        gate_sha256=sha256(GATE),source_sha256=sha256(source),manual_sha256=sha256(manual),
        solver_sha256=sha256(exe),script_sha256=sha256(__file__),capture_sha256=sha256(CAP),
        dependencies={n:sha256(F/'results/scripts'/n) for n in
        ('shear_unloading_coupon.py','audit_plastic_return_map.py','review_plastic_state_capture.py','audit_frozen_metric_work.py')},
        change='Only STATIC initial/maximum increment1/160 to1/320 and1/640; same prescribed F, material, cube, solver, outputs and INC1000.',
        acceptance='Unchanged native replay/work/trajectory gates; totalJ1 and reconstructedC1e-12; material extent; frozen unloading metric1e-10/PEEQ1e-12; source direction and logdet reconstruction1e-12; immutable hashes. Report determinant and common-state changes; decreasing nested determinant differences is a refinement indicator, not asymptotic or physical acceptance.',
        original_monotonic_work_gate_passed=False,threads=1,native_timeout_seconds=120,
        limitations='Original monotonic-work diagnostic stays failed. No state normalization, constitutive patch, physical storage/dissipation, cap/contact/MPC/Abaqus, experimental or burst acceptance.')
    (ROOT/'diagnostic_revision.json').write_text(json.dumps(manifest,indent=2)+'\n')
    records=list(retained['records'])
    for divisions in (320,640):
        admitted(); directory=ROOT/f'grid_{divisions}'; directory.mkdir()
        deck=text.replace('0.00625,1.,1.e-8,0.00625',f'{1/divisions},1.,1.e-8,{1/divisions}')
        assert deck!=text
        assert deck.replace(f'{1/divisions},1.,1.e-8,{1/divisions}','0.00625,1.,1.e-8,0.00625')==text
        (directory/'job.inp').write_text(deck)
        (directory/'capture.gdb').write_text(f'set pagination off\nset confirm off\nsource {CAP}\nquit\n')
        env=dict(os.environ,THESIS_INTERNAL_BINARY=str(directory/'fields.bin'),THESIS_INTERNAL_INDEX=str(directory/'fields.json'),MKL_THREADING_LAYER='GNU',
            **{n:'1' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CCX_NPROC_RESULTS','CCX_NPROC_STIFFNESS','CCX_NPROC_EQUATION_SOLVER')})
        with (directory/'solver.log').open('w') as log:
            result=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        log=(directory/'solver.log').read_text()
        assert result.returncode==0 and 'exited normally' in log and '*ERROR' not in log and 'Python Exception' not in log
        records.append(audit(directory,divisions,curve))
    summaries=[]; matched=[]
    for r in records:
        start=r['history'][r['divisions']//2-1]
        for h in r['history'][r['divisions']//2:]:
            assert abs(np.array(h['metric'])-np.array(start['metric'])).max()<1e-10
            assert abs(h['maximum_peeq']-start['maximum_peeq'])<1e-12
        summaries.append(dict(run=r['run'],divisions=r['divisions'],hashes=r['hashes'],
            loaded_metric_determinant=start['metric_determinant'],loaded_peeq=start['maximum_peeq'],**convention(r)))
    for coarse,fine in zip(records,records[1:]):
        differences=dict(metric=0.,stress_mpa=0.,peeq=0.)
        for i,a in enumerate(coarse['history']):
            b=fine['history'][2*i+1]; assert abs(a['time']-b['time'])<1e-10
            for key,field in (('metric','metric'),('stress_mpa','second_piola_stress_mpa'),('peeq','maximum_peeq')):
                differences[key]=max(differences[key],float(abs(np.array(a[field])-np.array(b[field])).max()))
        matched.append(dict(coarse=coarse['divisions'],fine=fine['divisions'],maximum_common_state_differences=differences))
    delta=[b['loaded_metric_determinant']-a['loaded_metric_determinant'] for a,b in zip(summaries,summaries[1:])]
    result=dict(manifest=manifest,records=summaries,matched=matched,identity_checks_passed=True,physical_acceptance=False,
        nested_loaded_determinant_differences=delta,
        successive_difference_ratios=[abs(b/a) for a,b in zip(delta,delta[1:])],
        nested_difference_magnitudes_decrease=all(abs(b)<abs(a) for a,b in zip(delta,delta[1:])),
        derivation='Manual M=Cp^-1 and isochoric Jp=1 require tr(M^-1 dM)=0. Source deltaM=beta[(tr(CM)/3)C^-1-M] instead enforces C:deltaM=0; its metric trace beta[(tr(CM)/3)tr((CM)^-1)-3] is nonnegative for SPD C,M. These constraints differ even when totalJ=1. This necessary kinematic identity alone does not derive a replacement constitutive law.',
        references='Local node161.html:64-115; incplas.f:68-80,156-189,775-790,1024-1032.',
        limitations=manifest['limitations'],next='Compare the cited Simo algorithms with this update, and independently verify any candidate integration and consistent tangent before considering an isolated solver revision. No additional same-path refinement solely for activity.')
    for r in records:
        assert all(sha256(F/r['run']/n)==v for n,v in r['hashes'].items())
    assert not OUT.exists(); OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(OUT),records=[{k:v for k,v in r.items() if k!='hashes'} for r in summaries],delta=delta)))


if __name__=='__main__': run()
