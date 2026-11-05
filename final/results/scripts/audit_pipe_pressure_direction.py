"""Independent follower-pressure action, normalized by pressure action alone."""
import json,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256
from audit_nodal_pressure_loads import shape8
from audit_end_surface_traction import face_nodes
from pipe_sim.model import Model
from verify_pipe_direction import action

def main(tag,parent_name='pipe_direction_v3_20261017T045902Z.json'):
    gate=F/'results/output'/parent_name
    parent=json.loads(gate.read_text());assert parent['checks_passed']
    baseline=json.loads((F/parent['parent_gate']).read_text())
    run=F/baseline['run'];base=json.loads((run/'assembly.json').read_text())
    for name,h in baseline['hashes'].items():assert sha256(run/name)==h
    source=json.loads((F/'results/output/plastic_pipe_replay_20261011T101242Z.json').read_text())
    model=Model.load(F/source['run']/'model.json')
    for name,h in source['input_output_hashes'].items():assert sha256(F/source['run']/name)==h
    nodes={}
    for line in (run/'mesh.inp').read_text().split('*NODE\n')[1].split('*')[0].strip().splitlines():
        fields=line.split(',');nodes[int(fields[0])]=list(map(float,fields[1:]))
    np.testing.assert_array_equal(np.array([nodes[i+1] for i in range(base['nk'])]),np.array(base['coordinates']).reshape(-1,3))
    for i,offset in enumerate(base['ipkon']):assert model.elements[i+1][1]==base['connectivity'][offset:offset+20]
    assert base['nmpc']==0
    mapping=np.array(base['mapping']).reshape(-1,4)[:,1:]
    xyz=np.array(base['coordinates']).reshape(-1,3)
    current=xyz+np.array(base['displacement_after']).reshape(-1,4)[:,1:]
    direction=np.array(json.loads((F/parent['cases'][0]['run']/'direction.json').read_text()))
    du=np.zeros_like(current);free=mapping>0;du[free]=direction[mapping[free]-1]
    assembled=np.zeros_like(current)
    lines=(run/'job.inp').read_text().split('*DLOAD\n')[1].split('*')[0].strip().splitlines()
    q,w=np.polynomial.legendre.leggauss(2)
    for line in lines:
        e,f,p=line.split(',');nodes=np.array(face_nodes(model,int(e),int(f.strip()[1:])))-1
        x=current[nodes];v=du[nodes];force=np.zeros((8,3))
        for r,wr in zip(q,w):
            for s,ws in zip(q,w):
                dr=shape8(r+1e-30j,s).imag/1e-30;ds=shape8(r,s+1e-30j).imag/1e-30
                area=np.cross(dr@v,ds@x)+np.cross(dr@x,ds@v)
                force+=-float(p)*base['time']*wr*ws*shape8(r,s)[:,None]*area
        np.add.at(assembled,nodes,force)
    predicted=np.zeros(base['neq']);predicted[mapping[free]-1]=assembled[free]
    states={};internals={};hashes={}
    for case in parent['cases']:
        probe=F/case['run']
        for name,h in case['hashes'].items():assert sha256(probe/name)==h
        data=json.loads((probe/'assembly.json').read_text())
        states[case['h']]=np.array(data['residual']['fext']);internals[case['h']]=np.array(data['internal']);hashes[case['run']]=case['hashes']
    matrix_action=action(base['global_matrix'],direction)
    rows=[]
    for h in sorted((h for h in states if h>0),reverse=True):
        native=(states[h]-states[-h])/(2*h)
        error=float(np.linalg.norm(native-predicted)/np.linalg.norm(predicted))
        internal=(internals[h]-internals[-h])/(2*h)
        defect=matrix_action-(internal-native)
        rows.append(dict(h=h,native_norm=float(np.linalg.norm(native)),independent_norm=float(np.linalg.norm(predicted)),relative_error=error,passed=error<2e-7,matrix_net_defect_over_pressure=float(np.linalg.norm(defect)/np.linalg.norm(predicted)),matrix_net_defect_minus_pressure_over_pressure=float(np.linalg.norm(defect-predicted)/np.linalg.norm(predicted))))
    report=dict(checks_passed=all(r['passed'] for r in rows),comparisons=rows,relative_limit=2e-7,parent_gate=str(gate.relative_to(F)),parent_sha256=sha256(gate),run_hashes=hashes,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),F/source['run']/'model.json',run/'assembly.json',run/'job.inp',Path(__file__).with_name('audit_end_surface_traction.py'),Path(__file__).with_name('audit_nodal_pressure_loads.py'))},limitations='Early elastic fixed-history iteration only, retained cap convention. Analytic surface-load action vs native external-force finite difference, no native matrix pressure tangent, cap work, plastic, contact/MPC, production or Abaqus acceptance. Failed audit retained without tolerance changes.')
    output=F/'results/output'/('pipe_pressure_direction_'+tag+'.json');assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows));assert report['checks_passed']
if __name__=='__main__':main(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'pipe_direction_v3_20261017T045902Z.json')
