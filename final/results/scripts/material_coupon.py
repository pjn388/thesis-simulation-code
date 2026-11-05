"""Independent uniaxial check of the material table passed to CalculiX."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from pipe_sim.config import Case
from pipe_sim.runner import solver_path,sha256,write_json
from pipe_sim.results import dat_blocks
from prepare_cases import FINAL,RESULTS


def main():
    case=Case.load(FINAL/'sym/cases/investigation/a37_curve_only.json')
    directory=FINAL/'sym/runs/investigation/material_coupon'
    directory.mkdir(parents=True,exist_ok=False)
    lines=['*HEADING','Unit cube tensile material verification, mm N MPa','*NODE',
        '1,0,0,0','2,1,0,0','3,1,1,0','4,0,1,0','5,0,0,1','6,1,0,1','7,1,1,1','8,0,1,1',
        '*ELEMENT, TYPE=C3D8, ELSET=CUBE','1,1,2,3,4,5,6,7,8',
        '*NSET, NSET=ALL','1,2,3,4,5,6,7,8','*NSET, NSET=RIGHT','2,3,6,7',
        '*NSET, NSET=LEFT','1,4,5,8','*NSET, NSET=Y0','1,2,5,6','*NSET, NSET=Z0','1,2,3,4',
        '*MATERIAL, NAME=STEEL','*ELASTIC',f'{case.material.young},{case.material.poisson}',
        '*PLASTIC']+[f'{s:.12g},{ep:.12g}' for s,ep in case.material.plastic]
    lines+=['*SOLID SECTION, ELSET=CUBE, MATERIAL=STEEL','*BOUNDARY','LEFT,1,1,0.','Y0,2,2,0.','Z0,3,3,0.',
        '*STEP, NLGEOM, INC=1000','*STATIC','.025,1.,1.e-8,.025','*BOUNDARY',f'RIGHT,1,1,{np.expm1(.15):.12g}',
        '*NODE PRINT, NSET=ALL','U, RF','*EL PRINT, ELSET=CUBE','S, PEEQ','*END STEP']
    (directory/'job.inp').write_text('\n'.join(lines)+'\n')
    exe=solver_path()
    with (directory/'solver.log').open('w') as log:
        result=subprocess.run([exe,'-i','job'],cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    log=(directory/'solver.log').read_text()
    if result.returncode or 'Job finished' not in log or '*ERROR' in log:raise RuntimeError('Coupon did not complete')
    frames={}
    for kind,group,t,data in dat_blocks(directory/'job.dat'):
        frames.setdefault(t,{})[kind]=data
    rows=[];plastic=np.array(case.material.plastic)
    for t,fields in frames.items():
        u={int(r[0]):r[1:] for r in fields['U']};rf={int(r[0]):r[1:] for r in fields['RF']}
        ep=float(np.mean([r[2] for r in fields['PEEQ']]))
        stress=float(np.mean([r[2] for r in fields['S']]))
        force=sum(rf[n][0] for n in (2,3,6,7));area=(1+u[7][1])*(1+u[7][2])
        tabulated=float(np.interp(ep,plastic[:,1],plastic[:,0]))
        rows.append(dict(time=t,true_strain=float(np.log1p(u[7][0])),peeq=ep,cauchy_stress_mpa=stress,
            force_over_current_area_mpa=force/area,tabulated_flow_stress_mpa=tabulated,
            plastic_flow_relative_error=(stress-tabulated)/tabulated if ep>1e-8 else None))
    out=RESULTS/'output/investigation';out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out/'material_coupon.csv',index=False)
    summary={'completed':True,'input_sha256':sha256(directory/'job.inp'),'solver_sha256':sha256(exe),
        'max_plastic_flow_relative_error':float(df.plastic_flow_relative_error.abs().max()),
        'max_force_stress_relative_difference':float(((df.force_over_current_area_mpa-df.cauchy_stress_mpa)/df.cauchy_stress_mpa).abs().max()),
        'final_true_strain':float(df.true_strain.iloc[-1])}
    write_json(out/'material_coupon.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
