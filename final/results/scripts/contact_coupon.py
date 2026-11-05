"""Analytical rigid-plane compression/release check of CCX contact tables."""
import json
import subprocess
import hashlib
from pathlib import Path
import pandas as pd
from pipe_sim.results import dat_blocks
from prepare_cases import FINAL,RESULTS


def deck(law):
    lines=['*HEADING','Unit-area rigid contact coupon: K=200000 MPa/mm; overlap=.01 mm','*NODE']
    for offset,z in [(0,-1.),(8,.1)]:
        for i,(x,y,dz) in enumerate([(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)],1):
            lines.append(f'{offset+i}, {x}, {y}, {z+dz}')
    lines+=['100,.5,.5,.6','101,0,0,0','*ELEMENT,TYPE=C3D8,ELSET=BOTTOM','1,1,2,3,4,5,6,7,8',
        '*ELEMENT,TYPE=C3D8,ELSET=TOP','2,9,10,11,12,13,14,15,16','*NSET,NSET=FIXED,GENERATE','1,8,1',
        '*NSET,NSET=TOOL,GENERATE','9,16,1','*NSET,NSET=REF','100','*MATERIAL,NAME=ELASTIC','*ELASTIC','200000,.3',
        '*SOLID SECTION,ELSET=BOTTOM,MATERIAL=ELASTIC','*SOLID SECTION,ELSET=TOP,MATERIAL=ELASTIC',
        '*RIGID BODY,NSET=TOOL,REF NODE=100,ROT NODE=101',
        '*SURFACE,NAME=SLAVE,TYPE=ELEMENT','1,S2','*SURFACE,NAME=MASTER,TYPE=ELEMENT','2,S1',
        '*SURFACE INTERACTION,NAME=CONTACT',f'*SURFACE BEHAVIOR,PRESSURE-OVERCLOSURE={law}']
    lines+=['200000'] if law=='LINEAR' else ['0,0','200000,1']
    lines+=['*CONTACT PAIR,INTERACTION=CONTACT,TYPE=SURFACE TO SURFACE','SLAVE,MASTER','*BOUNDARY',
        'FIXED,1,3,0','100,1,3,0','101,1,3,0']
    for travel in [-.11,0]:
        lines+=['*STEP,NLGEOM,INC=100','*STATIC','.1,1,1.e-8,.1','*BOUNDARY',f'100,3,3,{travel}',
            '*NODE PRINT,NSET=REF,FREQUENCY=1','U,RF','*END STEP']
    return '\n'.join(lines)+'\n'


def main():
    stock=FINAL/'sym/vendor/CalculiX/ccx_2.23/src/ccx_2.23'
    fixed=FINAL/'sym/vendor/ccx_tabular_fix/ccx_2.23_tabularfix'
    records=[]
    for name,law,exe in [('linear_stock','LINEAR',stock),('tabular_stock','TABULAR',stock),('tabular_fixed','TABULAR',fixed)]:
        directory=FINAL/'sym/runs/investigation/contact_coupon'/name
        directory.mkdir(parents=True,exist_ok=False);(directory/'job.inp').write_text(deck(law))
        with (directory/'solver.log').open('w') as log:
            result=subprocess.run([str(exe),'-i','job'],cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        rows={}
        for kind,group,time,data in dat_blocks(directory/'job.dat'):
            if kind in ('U','RF') and data:rows.setdefault(time,{'time':time})[kind]=data[0][3]
        table=pd.DataFrame(rows.values());table['force_n']=-table.RF
        table.to_csv(directory/'history.csv',index=False)
        peak=table.loc[(table.time-1).abs()<1e-7,'force_n'].iloc[0]
        release=table.loc[table.time>=1.1,'force_n'].abs().max()
        records.append(dict(case=name,returncode=result.returncode,peak_force_n=float(peak),expected_force_n=2000.,
            max_released_force_n=float(release),relative_peak_error=float(abs(peak-2000)/2000),
            solver_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()))
    out=RESULTS/'output/investigation/contact_coupon.json';out.write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))
    assert records[-1]['relative_peak_error']<1e-6
    assert records[-1]['max_released_force_n']<1e-6


if __name__=='__main__':main()
