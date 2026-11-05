"""Separate retained late FD internal/load errors without new solver probes."""
import json, sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256
from audit_pipe_assembled_action import assemble_actions

def main():
    report=F/'results/output/late_native_action_20261019.json'
    evidence=json.loads(report.read_text())
    basepath=F/'sym/runs/investigation/late_local_inputs_20261019/assembly.json'
    base=json.loads(basepath.read_text())
    direction=np.array(json.loads((F/evidence['cases'][0]['run']/'direction.json').read_text()))
    deck=(basepath.parent/'job.inp').read_text()
    table=np.array(base['ips'][0]['plconloc'][:400]).reshape(-1,2)[:,::-1]
    exact,_,external,coverage=assemble_actions(base,direction,table,208000,.3,deck)
    duration=float(deck.split('*STATIC, SOLVER=PARDISO\n')[1].splitlines()[0].split(',')[1]);external/=duration # Helper divides its fixed 2.3 MPa quadrature load, not step duration.
    del base
    states={}
    for case in evidence['cases']:
        p=F/case['run']/'assembly.json';assert sha256(p)==case['assembly_sha256']
        d=json.loads(p.read_text());states[case['h']]=(np.array(d['internal']),np.array(d['residual']['fext']));del d
    rows=[]
    for h in (1e-6,5e-7):
        internal=(states[h][0]-states[-h][0])/(2*h);load=(states[h][1]-states[-h][1])/(2*h)
        net=internal-load;scale=np.linalg.norm(net);err=exact-internal
        worst=np.argsort(abs(err))[-10:][::-1]
        rows.append(dict(h=h,internal_error_over_net=float(np.linalg.norm(err)/scale),load_error_over_net=float(np.linalg.norm(external-load)/scale),internal_relative=float(np.linalg.norm(err)/np.linalg.norm(internal)),load_relative=float(np.linalg.norm(external-load)/np.linalg.norm(load)),largest_internal_error_equations_one_based=(worst+1).tolist(),largest_internal_errors=err[worst].tolist(),corrected_net_relative=float(np.linalg.norm(exact-external-net)/scale),passed=bool(np.linalg.norm(exact-external-net)/scale<2e-7)))
    output=dict(checks_passed=all(r['passed'] for r in rows),relative_limit=2e-7,correction='Removed extra factor2.3: pressure helper integrates2.3MPa; its caller division normalizes that reference pressure. Actual static duration1s applied once.',comparisons=rows,coverage=coverage,sources={str(p.relative_to(F)):sha256(p) for p in (report,basepath,Path(__file__))},limitations='Localization only; failed net gate preserved. No native rerun, production correction, or physical stability inference.')
    p=F/'results/output/late_native_action_normalization_verified_20261019.json'
    with p.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
