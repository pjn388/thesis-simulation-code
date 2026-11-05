"""Compare analytical/reference coupons before enabling the PARDISO build."""
import os,json,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from pipe_sim.results import dat_blocks
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs'/('backend_checks_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'));EXE=ROOT/'vendor/ccx_pardiso/ccx_2.23_pardiso'

def main():
    records=[]
    for name,source in [('contact',ROOT/'runs/investigation/contact_coupon/tabular_fixed'),('material',ROOT/'runs/investigation/material_coupon'),('elastic',ROOT/'runs/elastic_001')]:
        target=OUT/name;target.mkdir(parents=True,exist_ok=False)
        for p in source.glob('*.inp'):shutil.copy2(p,target/p.name)
        deck=target/'job.inp';deck.write_text(deck.read_text().replace('*STATIC','*STATIC,SOLVER=PARDISO'))
        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MKL_THREADING_LAYER='GNU')
        with (target/'solver.log').open('w') as log:r=subprocess.run([str(EXE),'-i','job'],cwd=target,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        log=(target/'solver.log').read_text();assert r.returncode==0 and 'Job finished' in log and '*ERROR' not in log
        if name!='contact':assert 'pardiso solver' in log
        baseline={(q,g,t):np.array(a) for q,g,t,a in dat_blocks(source/'job.dat') if q and a}
        errors=[]
        for q,g,t,a in dat_blocks(target/'job.dat'):
            if (q,g,t) not in baseline:continue
            actual=np.array(a);expected=baseline[q,g,t];assert actual.shape==expected.shape
            scale=np.maximum(np.max(np.abs(expected),axis=0),1.)
            errors.append(float(np.max(abs(actual-expected)/scale)))
        assert errors and max(errors)<1.e-5,(name,max(errors))
        records.append(dict(case=name,max_normalized_output_difference=max(errors),blocks=len(errors)))
    build=ROOT/'vendor/ccx_pardiso/build.json';meta=json.loads(build.read_text());meta.update(verified=True,verification=records,verification_directory=str(OUT))
    build.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(records,indent=2))
if __name__=='__main__':main()
