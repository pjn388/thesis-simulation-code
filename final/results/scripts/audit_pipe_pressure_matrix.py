"""Independent sparse surface derivative; never a native stiffness extraction."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from audit_plastic_return_map import F, sha256
from audit_quadratic_pressure_derivative import assemble
from audit_end_surface_traction import face_nodes
from pipe_sim.model import Model


def main(tag):
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    gate=F/'results/output/pipe_pressure_direction_resolution_v3_20261017T050602Z.json'
    accepted=json.loads(gate.read_text()); assert accepted['checks_passed']
    pg=F/accepted['parent_gate']; assert sha256(pg)==accepted['parent_sha256']
    parent=json.loads(pg.read_text()); assert parent['checks_passed']
    bg=F/parent['parent_gate']; assert sha256(bg)==parent['parent_sha256']
    baseline=json.loads(bg.read_text()); run=F/baseline['run']
    for n,h in baseline['hashes'].items(): assert sha256(run/n)==h
    base=json.loads((run/'assembly.json').read_text()); assert base['nmpc']==0
    source=F/'results/output/plastic_pipe_replay_20261011T101242Z.json'
    original=json.loads(source.read_text()); modelpath=F/original['run']/'model.json'
    for n,h in original['input_output_hashes'].items(): assert sha256(F/original['run']/n)==h
    model=Model.load(modelpath)
    mapping=np.array(base['mapping']).reshape(-1,4)[:,1:]
    current=np.array(base['coordinates']).reshape(-1,3)+np.array(base['displacement_after']).reshape(-1,4)[:,1:]
    rows=[];cols=[];values=[]; local_asym=[];count=0
    lines=(run/'job.inp').read_text().split('*DLOAD\n')[1].split('*')[0].strip().splitlines()
    # assemble() returns outward pressure2.3, while native positive DLOAD is inward.
    for line in lines:
        e,face,p=line.split(','); nodes=np.array(face_nodes(model,int(e),int(face.strip()[1:])))-1
        _,k=assemble(current[nodes],2); k*= -float(p)*base['time']/2.3
        local_asym.append(float(np.linalg.norm(k-k.T)/np.linalg.norm(k)))
        ids=mapping[nodes].ravel()-1; keep=ids>=0;ids=ids[keep]; k=k[np.ix_(keep,keep)]
        rows.extend(np.repeat(ids,len(ids)));cols.extend(np.tile(ids,len(ids)));values.extend(k.ravel());count+=1
    matrix=coo_matrix((values,(rows,cols)),shape=(base['neq'],base['neq'])).tocsr();matrix.eliminate_zeros()
    norm=float(np.sqrt(matrix.multiply(matrix).sum()));anti=matrix-matrix.T
    asym=float(np.sqrt(anti.multiply(anti).sum())/norm)
    case0=F/parent['cases'][0]['run'];direction=np.array(json.loads((case0/'direction.json').read_text()))
    predicted=matrix@direction; symmetric=((matrix+matrix.T)*.5)@direction
    states={};hashes={}
    for case in parent['cases']:
        probe=F/case['run']
        for n,h in case['hashes'].items(): assert sha256(probe/n)==h
        states[case['h']]=np.array(json.loads((probe/'assembly.json').read_text())['residual']['fext'])
        hashes[case['run']]=case['hashes']
    comparisons=[]
    for h in sorted((h for h in states if h>0),reverse=True):
        native=(states[h]-states[-h])/(2*h)
        error=float(np.linalg.norm(native-predicted)/np.linalg.norm(predicted))
        comparisons.append(dict(h=h,relative_error=error,symmetrized_relative_error=float(np.linalg.norm(native-symmetric)/np.linalg.norm(predicted)),passed=error<2e-7))
    report=dict(checks_passed=all(x['passed'] for x in comparisons),faces=count,neq=base['neq'],nnz=matrix.nnz,matrix_frobenius=norm,assembled_constrained_asymmetry=asym,local_face_asymmetry_range=[min(local_asym),max(local_asym)],action_norm=float(np.linalg.norm(predicted)),symmetrized_action_change=float(np.linalg.norm(symmetric-predicted)/np.linalg.norm(predicted)),comparisons=comparisons,parent_gate=str(gate.relative_to(F)),parent_sha256=sha256(gate),run_hashes=hashes,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),Path(__file__).with_name('audit_quadratic_pressure_derivative.py'),Path(__file__).with_name('audit_end_surface_traction.py'),modelpath,source,run/'assembly.json',run/'job.inp')},limitations='Independent analytic pressure-only sparse matrix at one early elastic geometry, two-size native external FD in one direction. Native e_c3d pressure-stiffness branch is not extracted or invoked. Retained cap convention; no coupled cap work, plastic/contact/MPC, convergence causation, production patch, Abaqus or experimental acceptance. Symmetrization is measured, not adopted.')
    out=F/'results/output'/('pipe_pressure_matrix_'+tag+'.json'); assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('run_hashes','provenance')}));assert report['checks_passed']

if __name__=='__main__': main(sys.argv[1])
