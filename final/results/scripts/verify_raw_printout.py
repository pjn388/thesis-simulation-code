"""Bounded raw-output ABI coupon check, run in independent systemd unit."""
import json
import os
from pathlib import Path
import subprocess
from decimal import Decimal
import numpy as np
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_pressure_failures import converged_times, matches_sta

FINAL=Path(__file__).resolve().parents[2]
EXE=FINAL/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
CAPTURE=FINAL/'results/scripts/capture_raw_printout_gdb.py'
ROOT=FINAL/'sym/runs/investigation/raw_printout_coupon_20261011T083005Z'
SOURCE=FINAL/'sym/runs/investigation/plastic_work_20261010T103001Z/dt_0.025'

def main():
    assert json.loads((FINAL/'results/automation/paused.json').read_text())['state']=='resumed'
    assert sha256(EXE)==json.loads((EXE.parent/'build.json').read_text())['solver_sha256']
    ROOT.mkdir(exist_ok=False)
    (ROOT/'job.inp').write_bytes((SOURCE/'job.inp').read_bytes())
    (ROOT/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(CAPTURE)+'\nquit\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_RAW_OUTPUT=str(ROOT/'raw.json'))
    with (ROOT/'solver.log').open('w') as log:
        result=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(EXE),'-i','job'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    log=(ROOT/'solver.log').read_text()
    assert result.returncode==0 and 'exited normally' in log and '*ERROR' not in log and 'Python Exception' not in log
    raw=json.loads((ROOT/'raw.json').read_text())['records']
    accepted=converged_times(ROOT/'job.sta')
    frames={}
    for kind,group,time,rows in dat_blocks(ROOT/'job.dat'):
        assert matches_sta(accepted,time)
        frames.setdefault(time,{})[kind]=rows
    assert len(raw)==len(frames) and abs(raw[-1]['time']-2)<1e-12
    maximum_u=maximum_energy=0.
    for record,(time,fields) in zip(raw,sorted(frames.items())):
        assert matches_sta(accepted,record['time']) and abs(record['time']-time)<1e-7
        assert record['nodes']==8 and record['elements']==1 and len(record['energy'])==1
        for row in fields['U']:
            for value,actual in zip(row[1:],record['u'][int(row[0])-1]):
                # e13.6 output has seven significant decimal digits.
                allowance=0 if value==0 else .5*10**(int(np.floor(np.log10(abs(value))))-6)
                assert abs(actual-value)<=allowance+1e-14
                maximum_u=max(maximum_u,abs(actual-value))
        printed=fields[None][0][1]
        energy=record['energy'][0][1]
        allowance=.5*10**(int(np.floor(np.log10(abs(printed))))-6)
        assert abs(energy-printed)<=allowance+1e-12
        maximum_energy=max(maximum_energy,abs(energy-printed))
    assert (ROOT/'job.dat').read_bytes()==(SOURCE/'job.dat').read_bytes()
    report=dict(run=str(ROOT.relative_to(FINAL)),parent=str(SOURCE.relative_to(FINAL)),accepted_frames=len(raw),max_u_rounding_mm=maximum_u,max_energy_rounding_n_mm=maximum_energy,endpoint_energy_n_mm=raw[-1]['energy'][0][1],identical_parent_dat=True,hashes={n:sha256(ROOT/n) for n in ['job.inp','job.dat','job.sta','solver.log','raw.json']},parent_deck_sha256=sha256(SOURCE/'job.inp'),solver_sha256=sha256(EXE),capture_script_sha256=sha256(CAPTURE),verification_script_sha256=sha256(__file__),sources=['sym/vendor/CalculiX/ccx_2.23/src/printout.f:19-27','sym/vendor/CalculiX/ccx_2.23/src/printoutelem.f:19-26,502-521'],limitations='One C3D8 plastic loading/unloading coupon; no C3D20R pipe verification, external work/path or internal integration acceptance, contact/MPC, cap tangent or Abaqus equivalence. Cumulative energy subtraction precision requires assessment for multi-element pipe.')
    (FINAL/'results/output/raw_printout_coupon_20261011T083005Z.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    main()
