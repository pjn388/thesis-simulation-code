"""Read-only component audit of the retained failed early-plastic pipe FD.

Local sampled PK2 stress derivatives are distinct from global matrix acceptance.
"""
import json, sys, gc
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F, sha256, tensors, update
from audit_material_linearization import derivative, vector
from audit_internal_pipe_fields import shape
from verify_heterogeneous_coupon import points
from verify_pipe_direction import action, mapping


def main(tag, parent_report="pipe_direction_early_plastic_v2_20261017T055357Z.json"):
    gate=F/'results/output'/parent_report
    evidence=json.loads(gate.read_text())
    # A failed derivative report is intentionally admissible as investigation data.
    assert len(evidence['cases'])==4
    pg=F/evidence['parent_gate'];assert sha256(pg)==evidence['parent_sha256']
    parent=json.loads(pg.read_text());assert parent['checks_passed']
    run=F/parent['run']
    for name,h in parent['hashes'].items():assert sha256(run/name)==h
    for name,h in evidence['provenance'].items():assert sha256(F/name)==h
    base=json.loads((run/'assembly.json').read_text());mapping(base)
    assert base['nmpc']==0 and base['ne']==2880
    material=json.loads((F/parent['parent_run']/'case.json').read_text())['material']
    young,nu=material['young'],material['poisson'];mu=young/(2*(1+nu));bulk=young/(3*(1-2*nu))
    lines=(run/'job.inp').read_text().splitlines();curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve)
    xyz=np.array(base['coordinates']).reshape(-1,3);connect=np.array(base['connectivity'])
    gradients={}
    for p in base['ips']:
        key=(p['element'],p['ip']);q=points()[p['ip']-1]
        d=np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T
        nodes=connect[base['ipkon'][p['element']-1]:base['ipkon'][p['element']-1]+20]-1
        gradients[key]=(nodes,d@np.linalg.inv(xyz[nodes].T@d))
    def samples(data):
        u=np.array(data['displacement_after']).reshape(-1,4)[:,1:]
        stress=np.array(data['stress']).reshape(data['ne'],8,6);out={}
        for p in data['ips']:
            key=p['element'],p['ip'];nodes,grad=gradients[key]
            deform=np.eye(3)+u[nodes].T@grad;inv=np.linalg.inv(deform)
            # Native results evaluation captured before Cauchy output transformation.
            pk2=tensors(stress[p['element']-1,p['ip']-1][None])[0]
            out[key]=(np.array(p['strain']),vector(pk2),p['previous'])
        return out
    sampled={};states={}
    for case in evidence['cases']:
        probe=F/case['run']
        for name,h in case['hashes'].items():assert sha256(probe/name)==h
        data=json.loads((probe/'assembly.json').read_text());mapping(data)
        for key in ('previous_state','previous_stress','displacement_before','mapping','time','ttime','coordinates','ipkon','connectivity'):
            assert data[key]==base[key],key
        sampled[case['h']]=samples(data)
        states[case['h']]=(np.array(data['internal']),np.array(data['residual']['fext']))
        del data;gc.collect()
    direction=np.array(json.loads((F/evidence['cases'][0]['run']/'direction.json').read_text()))
    predicted=action(base['global_matrix'],direction);local=[];global_rows=[]
    refs=samples(base)
    for h in (1e-7,5e-8):
        internal=(states[h][0]-states[-h][0])/(2*h);external=(states[h][1]-states[-h][1])/(2*h)
        defect=internal-predicted
        global_rows.append(dict(h=h,internal_defect_norm=float(np.linalg.norm(defect)),external_norm=float(np.linalg.norm(external)),external_over_internal_defect=float(np.linalg.norm(external)/np.linalg.norm(defect)),net_relative_error=float(np.linalg.norm(defect-external)/np.linalg.norm(internal-external))))
        for key,(e,s,previous) in refs.items():
            ep,sp,pp=sampled[h][key];em,sm,pm=sampled[-h][key];assert pp==pm==previous
            c=np.eye(3)+2*tensors(e[None])[0];prev=np.array(previous);m=np.eye(3)-2*tensors(prev[None,1:7])[0]
            dc=2*tensors(((ep-em)/(2*h))[None])[0]
            analytic=vector(derivative(c,m,prev[0],curve,mu,bulk,dc));fd=(sp-sm)/(2*h)
            stress,_,qp,active=update(c,m,np.array(prev[0]),curve,mu,bulk)
            norm=float(np.linalg.norm(fd));error=float(np.linalg.norm(fd-analytic))
            local.append(dict(element=key[0],ip=key[1],h=h,prior_peeq=float(prev[0]),current_peeq=float(qp),active=bool(active),stress_error_mpa=float(abs(vector(stress)-s).max()),fd_norm=norm,analytic_fd_absolute=error,analytic_fd_relative=error/norm if norm>1e-6 else None))
    passed=all(r['stress_error_mpa']<1e-4 and (r['analytic_fd_relative'] is None and r['analytic_fd_absolute']<1e-4 or r['analytic_fd_relative'] is not None and r['analytic_fd_relative']<2e-7) for r in local)
    coverage=all(r['fd_norm']>1e-6 for r in local)
    result=dict(checks_passed=passed and coverage,stress_and_action_checks_passed=passed,local_direction_coverage_passed=coverage,global_comparisons=global_rows,local_comparisons=local,parent_gate=str(gate.relative_to(F)),parent_sha256=sha256(gate),verified_probe_files=24,limits=dict(relative=2e-7,stress_mpa=1e-4,near_zero_action=1e-6),provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),Path(__file__).with_name('audit_material_linearization.py'),Path(__file__).with_name('audit_plastic_return_map.py'),F/parent['parent_run']/'case.json')},limitations='Retained captures only. Sixteen IPs in two elements, one sparse direction and early nonzero plastic history. Raw captured stress interpreted as PK2 at this results evaluation, verified against independent fixed-history stress update; resultsmech.f:1203-1247 output Cauchy branch is distinct. Zero local action never establishes derivative coverage. No global corrected assembly, convergence causation, finite cap/contact/MPC/Abaqus or experimental acceptance; production remains unchanged. Pressure norm ratio is a bound on measured action size, not a pressure-tangent accuracy test.')
    out=F/'results/output'/('early_plastic_component_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(checks_passed=result['checks_passed'],global_comparisons=global_rows,local_max_stress_error=max(r['stress_error_mpa'] for r in local),local_max_relative=max((r['analytic_fd_relative'] or 0) for r in local),active_samples=sum(r['active'] for r in local)),indent=2))
    assert passed,'Preserve failed component report; do not relax gates.'
    print('Local direction coverage:',coverage)

if __name__=='__main__':main(sys.argv[1],*sys.argv[2:])
