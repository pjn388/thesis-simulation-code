"""Native plastic unload/reload restart and dynamic energy regression coupons."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from pipe_sim.continuation import DENSITY, ENERGY, RESTART, ramp_points, energy_history
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256, solver_environment, write_json
from prepare_cases import FINAL

ROOT=FINAL/'sym/runs/investigation/restart_dynamic_coupon_20261022'
OUT=FINAL/'results/output/restart_dynamic_20261022'
EXE=FINAL/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
EXPECTED='0f974d82f4355a93006fcc4f4bcabcd93da3631013b6ba33934648c41f1fa72b'


def coupon():
    return '''*HEADING
Plastic unit cube: tensile load, unload, reload; mm,N,MPa,s,tonne
*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT,TYPE=C3D8,ELSET=PIPE
1,1,2,3,4,5,6,7,8
*NSET,NSET=ALL,GENERATE
1,8,1
*NSET,NSET=X0
1,4,5,8
*NSET,NSET=Y0
1,2,5,6
*NSET,NSET=Z0
1,2,3,4
*MATERIAL,NAME=STEEL
*ELASTIC
208000,0.3
*PLASTIC
365,0
500,0.1
650,0.5
*DENSITY
'''+f'{DENSITY:.12g}\n'+'''*SOLID SECTION,ELSET=PIPE,MATERIAL=STEEL
*BOUNDARY
X0,1,1
Y0,2,2
Z0,3,3
'''


def step(load, dynamic=False):
    ramp=''
    if dynamic:
        ramp='*AMPLITUDE,NAME=RAMP\n'+''.join(f'{t:.12g},{v:.12g}\n' for t,v in ramp_points(.02,100))
    text=ramp+'*STEP,NLGEOM,INC=2000\n'
    text+=('*DYNAMIC,ALPHA=-0.05,SOLVER=PARDISO\n0.00002,0.02,1e-9,0.0002\n' if dynamic else
           '*STATIC,SOLVER=PARDISO\n0.05,1,1e-8,0.05\n')
    text+=('*DLOAD,AMPLITUDE=RAMP\n' if dynamic else '*DLOAD\n')+f'1,P4,{-load}\n'
    return text+'''*NODE PRINT,NSET=ALL,FREQUENCY=1
U,RF
*EL PRINT,ELSET=PIPE,FREQUENCY=1
S,E,PEEQ
'''+ENERGY+RESTART+'*END STEP\n'


def execute(name, text, restart=None):
    directory=ROOT/name;directory.mkdir(parents=True,exist_ok=False)
    (directory/'job.inp').write_text(text)
    if restart:
        shutil.copy2(restart,directory/'job.rin')
        assert sha256(restart)==sha256(directory/'job.rin')
    with (directory/'solver.log').open('w') as log:
        result=subprocess.run([str(EXE),'-i','job'],cwd=directory,env=solver_environment(2),
            stdout=log,stderr=subprocess.STDOUT,timeout=90)
    assert result.returncode==0,(name,result.returncode)
    assert '*ERROR' not in (directory/'solver.log').read_text()
    return directory


def compare(continuous, restarted):
    def frames(directory):
        return {(kind,group,time):np.asarray(data) for kind,group,time,data in dat_blocks(directory/'job.dat')
                if kind in ('U','RF','S','E','PEEQ') and time>2.+1e-9}
    a,b=frames(continuous),frames(restarted)
    assert a.keys()==b.keys(), 'Restart changed accepted frame schedule'
    error=max(float(np.max(np.abs(a[k]-b[k]))) for k in a)
    assert error<=1e-7,error
    ea,eb=[energy_history(d/'job.dat') for d in (continuous,restarted)]
    energy_error=max(abs(ea[t][field]-eb[t][field]) for t in eb if t>2. for field in ('internal_energy_nmm','kinetic_energy_nmm'))
    assert energy_error<=1e-7,energy_error
    assert all(np.isfinite(v) for row in eb.values() for v in row.values())
    return {'field_blocks':len(a),'max_abs_printed_field_difference':error,'max_abs_energy_difference_nmm':energy_error}


def main():
    assert sha256(EXE)==EXPECTED
    seed_text=coupon()+step(400)+step(0)
    seed=execute('seed',seed_text)
    assert (seed/'job.rout').stat().st_size>0
    peeq=[r[2] for kind,_,time,data in dat_blocks(seed/'job.dat') if kind=='PEEQ' and time==2 for r in data]
    assert min(peeq)>.001, 'Coupon did not establish plastic history'
    results={}
    for dynamic in (False,True):
        label='dynamic' if dynamic else 'static'
        follow=step(420,dynamic)
        continuous=execute(label+'_continuous',seed_text+follow)
        restarted=execute(label+'_restart','*RESTART,READ\n'+follow,seed/'job.rout')
        results[label]=compare(continuous,restarted)
    OUT.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(FINAL)):sha256(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.inp','.dat','.rout','.rin')}
    write_json(OUT/'coupon_verification.json',{'passed':True,'solver_sha256':EXPECTED,
        'unloaded_peeq_min':min(peeq),'comparisons':results,'sources':sources,
        'limitation':'Plastic unit cube only; full-pipe checkpoint and restarted low-pressure overlap audit remains mandatory.'})
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
