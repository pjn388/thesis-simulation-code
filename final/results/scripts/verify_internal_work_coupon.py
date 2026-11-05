"""Independent raw material-work convention check on unchanged plastic cube."""
import json, os, subprocess
from pathlib import Path
import numpy as np
from pipe_sim.runner import sha256
from pipe_sim.results import dat_blocks
from audit_pressure_failures import converged_times,matches_sta
F=Path(__file__).resolve().parents[2]
TAG='internal_work_coupon_20261011T090414Z'
R=F/'sym/runs/investigation'/TAG
P=F/'sym/runs/investigation/plastic_work_20261010T103001Z/dt_0.025'
EXE=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
CAP=F/'results/scripts/capture_internal_printout_gdb.py'
def tensor(v):
    a,b,c,d,e,f=v;return np.array([[a,d,e],[d,b,f],[e,f,c]])
def run():
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    assert sha256(EXE)==json.loads((EXE.parent/'build.json').read_text())['solver_sha256']
    R.mkdir(exist_ok=False);(R/'job.inp').write_bytes((P/'job.inp').read_bytes())
    (R/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(CAP)+'\nquit\n')
    env=dict(os.environ,THESIS_INTERNAL_BINARY=str(R/'fields.bin'),THESIS_INTERNAL_INDEX=str(R/'fields.json'),MKL_THREADING_LAYER='GNU',**{n:'1' for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','CCX_NPROC_RESULTS','CCX_NPROC_STIFFNESS','CCX_NPROC_EQUATION_SOLVER')})
    with (R/'solver.log').open('w') as log:ret=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(EXE),'-i','job'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert ret.returncode==0 and 'exited normally' in (R/'solver.log').read_text() and 'Python Exception' not in (R/'solver.log').read_text()
    assert (R/'job.dat').read_bytes()==(P/'job.dat').read_bytes()
    records=json.loads((R/'fields.json').read_text())['records'];binary=np.memmap(R/'fields.bin',dtype='<f8',mode='r');accepted=converged_times(R/'job.sta')
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    design=np.c_[np.ones(8),xyz];prevS=np.zeros((8,3,3));prevE=prevS.copy();previous=np.zeros(8)
    maximum_affine=maximum_green=maximum_update=maximum_energy_output=0.;history=[]
    printed={}
    for k,g,t,rows in dat_blocks(R/'job.dat'):
        if k is None:printed[t]=rows[0][1]
    for rec in records:
        assert matches_sta(accepted,rec['time']) and rec['elements']==1 and rec['ips']==8
        def field(name):
            m=rec['fields'][name];return binary[m['offset']//8:m['offset']//8+m['count']]
        u=field('u').reshape(8,4)[:,1:];coef=np.linalg.lstsq(design,u,rcond=None)[0];maximum_affine=max(maximum_affine,float(abs(design@coef-u).max()))
        deformation=np.eye(3)+coef[1:].T;J=np.linalg.det(deformation);inverse=np.linalg.inv(deformation)
        sigma=np.array([tensor(v) for v in field('stress').reshape(8,6)]);S=J*np.einsum('ij,pjk,lk->pil',inverse,sigma,inverse)
        E=np.array([tensor(v) for v in field('strain').reshape(8,6)]);green=(deformation.T@deformation-np.eye(3))/2
        maximum_green=max(maximum_green,float(abs(E-green).max()))
        energy=field('energy').reshape(8,2)[:,0];expected=previous+.5*np.einsum('pij,pij->p',S+prevS,E-prevE)
        maximum_update=max(maximum_update,float(abs(energy-expected).max()))
        total=float(energy.mean());t=min(printed,key=lambda v:abs(v-rec['time']));value=printed[t]
        allowance=.5*10**(int(np.floor(np.log10(abs(value))))-6)
        assert abs(total-value)<=allowance+1e-12
        maximum_energy_output=max(maximum_energy_output,abs(total-value));history.append(dict(time=rec['time'],energy_n_mm=total))
        prevS,prevE,previous=S,E,energy
    assert len(records)==80 and maximum_affine<1e-12 and maximum_green<1e-12 and maximum_update<1e-10
    report=dict(run=str(R.relative_to(F)),parent=str(P.relative_to(F)),accepted_frames=len(records),identical_parent_dat=True,max_affine_residual_mm=maximum_affine,max_green_difference=maximum_green,max_native_energy_update_difference_n_mm_per_mm3=maximum_update,max_printed_energy_rounding_n_mm=maximum_energy_output,history=history,hashes={n:sha256(R/n) for n in ('job.inp','job.dat','job.sta','solver.log','fields.bin','fields.json')},solver_sha256=sha256(EXE),capture_sha256=sha256(CAP),script_sha256=sha256(__file__),sources=['sym/vendor/CalculiX/ccx_2.23/src/calctotstrain.f','sym/vendor/CalculiX/ccx_2.23/src/resultsmech.f:1064-1074,1203-1247','sym/vendor/CalculiX/ccx_2.23/src/printoutelem.f:247-252,428,502'],verdict='Homogeneous C3D8 supported-plastic load/unload confirms Cauchy pullback and Green-strain reference-volume energy update to floating precision.',limitations='Not C3D20R spatial-gradient verification, pipe work convergence, cap coupling, contact/MPC, Abaqus equivalence or physical validation. No production changes.')
    out=F/'results/output'/f'{TAG}.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('history','hashes')},indent=2))
if __name__=='__main__':run()
