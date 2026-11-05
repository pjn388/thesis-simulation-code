"""Source algebra replay of shape8q global mapping; diagnostic only."""
import json,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256
from audit_quadratic_pressure_derivative import shape,assemble
from audit_end_surface_traction import face_nodes
from pipe_sim.model import Model

def global_gradients(x,r,s):
    n,dr,ds=shape(r,s);xs=np.column_stack((dr@x,ds@x));area=np.cross(xs[:,0],xs[:,1]);d1,d2,d3=abs(area);v=np.zeros((2,3))
    # Literal case structure of shape8q.f:130-215 (one-based source axes).
    if d3>d2 and d3>d1:
        v[0,0]=xs[1,1]/area[2];v[1,1]=xs[0,0]/area[2];v[0,1]=-xs[0,1]/area[2];v[1,0]=-xs[1,0]/area[2]
        if d2>d1:
            if d2>=1e-10:v[1,2]=xs[0,0]/(-area[1]);v[0,2]=-xs[0,1]/(-area[1])
        else:
            if d1>=1e-10:v[1,2]=xs[1,0]/area[0];v[0,2]=-xs[1,1]/area[0]
    elif d2>d1 and d2>d3:
        v[0,0]=xs[2,1]/(-area[1]);v[1,2]=xs[0,0]/(-area[1]);v[0,2]=-xs[0,1]/(-area[1]);v[1,0]=-xs[2,0]/(-area[1])
        if d1>d3:
            if d1>=1e-10:v[0,1]=xs[2,1]/area[0];v[1,1]=-xs[2,0]/area[0]
        else:
            if d3>=1e-10:v[0,1]=-xs[0,1]/area[2];v[1,1]=xs[0,0]/area[2]
    else:
        v[0,1]=xs[2,1]/area[0];v[1,2]=xs[1,0]/area[0];v[0,2]=-xs[1,1]/area[0];v[1,1]=-xs[2,0]/area[0]
        if d3>d2:
            if d3>=1e-10:v[0,0]=xs[1,1]/area[2];v[1,0]=-xs[1,0]/area[2]
        else:
            if d2>=1e-10:v[0,0]=xs[2,1]/(-area[1]);v[1,0]=-xs[2,0]/(-area[1])
    grad=np.column_stack((dr,ds))@v
    return n,grad,area,xs,v

def branch(x,y,p):
    matrix=np.zeros((24,24));mapping=[]
    q,w=np.polynomial.legendre.leggauss(2)
    for r,wr in zip(q,w):
        for s,ws in zip(q,w):
            n,g,area,xs,v=global_gradients(x,r,s);xkl=y.T@g
            mapping.append(dict(reference_chain=(v@xs).tolist(),surface_inverse_error=float(np.linalg.norm(v@xs-np.eye(2))),area=area.tolist()))
            for i in range(8):
                for j in range(8):
                    for k in range(3):
                        for l in range(3):
                            if k==l:continue
                            if (k+1)*(l+1)==2:other=2;sign=-1 if k<l else 1
                            elif (k+1)*(l+1)==3:other=1;sign=-1 if k>l else 1
                            else:other=0;sign=-1 if k<l else 1
                            term=wr*ws*p*n[i]*sign*np.dot(area,np.cross(xkl[other],g[j]))
                            matrix[3*i+k,3*j+l]+=term/2
                            matrix[3*j+l,3*i+k]+=term/2
    return matrix,mapping

def main(tag):
    gate=F/'results/output/pressure_branch_v2_20261017T052035Z.json';parent=json.loads(gate.read_text());assert not parent['checks_passed']
    records=[]
    for run,hashes in parent['run_hashes'].items():
        for name,h in hashes.items():assert sha256(F/run/name)==h
        records.append(json.loads((F/run/'capture.json').read_text()))
    a,b=records;native=(np.array(a['matrix'])-b['matrix']).reshape(60,60,order='F')
    original=json.loads((F/'results/output/plastic_pipe_replay_20261011T101242Z.json').read_text());model=Model.load(F/original['run']/'model.json')
    face=face_nodes(model,a['element'],1);ids=np.array([a['nodes'].index(n) for n in face]);dofs=(3*ids[:,None]+np.arange(3)).ravel()
    x=np.array(a['coordinates'])[ids];y=x+np.array(a['displacement'])[ids,1:]
    selected=[i for i in range(len(a['loads'])//2) if a['load_elements'][2*i]==a['element']];assert len(selected)==1
    pressure=a['loads'][2*selected[0]]
    mapped,chain=branch(x,y,pressure);_,raw=assemble(y,2);raw*=pressure/2.3;sym=(raw+raw.T)/2
    full=np.zeros((60,60));full[np.ix_(dofs,dofs)]=mapped
    error=float(np.linalg.norm(native-full)/np.linalg.norm(native))
    # A planar reference with constant z is a separate mapping control.
    flat=x.copy();flat[:,2]=0
    flatbranch,flatchain=branch(flat,flat,pressure);_,flatraw=assemble(flat,2);flatraw*=pressure/2.3
    flaterr=float(np.linalg.norm(flatbranch-(flatraw+flatraw.T)/2)/np.linalg.norm(flatraw))
    report=dict(checks_passed=error<2e-7 and flaterr<2e-7,native_source_algebra_relative=error,relative_limit=2e-7,curved_vs_symmetric_relative=float(np.linalg.norm(mapped-sym)/np.linalg.norm(raw)),curved_vs_twice_symmetric_relative=float(np.linalg.norm(mapped-2*sym)/np.linalg.norm(raw)),curved_mapping=chain,flat_mapping=flatchain,flat_vs_symmetric_relative=flaterr,parent_gate=str(gate.relative_to(F)),parent_sha256=sha256(gate),provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),F/'sym/vendor/CalculiX/ccx_2.23/src/shape8q.f',F/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f',Path(__file__).with_name('audit_quadratic_pressure_derivative.py'))},limitations='Literal source algebra reproduced for one captured curved element and planar reference control. Failed original native derivative gate remains failed. No native unsymmetrized extraction, full global matrix, actual static branch activation, solver patch, convergence causation, plastic/cap work/Abaqus/physical validation.')
    out=F/'results/output'/('pressure_branch_mapping_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));assert report['checks_passed']
if __name__=='__main__':main(sys.argv[1])
