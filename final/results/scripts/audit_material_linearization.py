"""Analytic fixed-history derivative; isolate yield-stress determinant variation."""
import json,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,tensors,sha256,update


def derivative(c,m,q,curve,mu,bulk,dc,freeze_yield_j=False):
    stress,_,qp,active=update(c,m,np.array(q),curve,mu,bulk)
    j=np.sqrt(np.linalg.det(c)); ci=np.linalg.inv(c); z=j**(2/3)
    dj=j*np.trace(ci@dc)/2; dci=-ci@dc@ci; dz=2*z*dj/(3*j)
    t=np.sum(c*m)/(3*z); dt=np.sum(dc*m)/(3*z)-t*dz/z
    trial=mu*(m/z-t*ci); dtrial=mu*(-m*dz/z**2-dt*ci-t*dci)
    sc=trial@c; dsc=dtrial@c+trial@dc
    n=np.sqrt(np.trace(sc@sc)); dn=np.trace(sc@dsc)/n
    a=np.sqrt(2/3); cop=(qp-q)/a
    if active:
        seg=np.searchsorted(curve[:,1],qp,side='right')
        h=0. if seg in (0,len(curve)) else np.diff(curve[seg-1:seg+1,0])[0]/np.diff(curve[seg-1:seg+1,1])[0]
        y=np.interp(qp,curve[:,1],curve[:,0])
        dcop=(dn-a*y*(0 if freeze_yield_j else dj)-2*mu*cop*dt)/(a*a*j*h+2*mu*t)
        f=2*mu*t*cop/n
        df=2*mu*((dt*cop+t*dcop)/n-t*cop*dn/n**2)
    else:f=0.;df=0.
    return bulk*j*dj*ci+bulk/2*(j*j-1)*dci+(1-f)*dtrial-df*trial


def vector(s):return np.array([s[0,0],s[1,1],s[2,2],s[0,1],s[0,2],s[1,2]])


def main(tag):
    parent=F/'results/output/material_tangent_20261017T040006Z.json'
    evidence=json.loads(parent.read_text());run=F/evidence['run']
    for name,h in evidence['hashes'].items():assert sha256(run/name)==h
    records=json.loads((run/'tangent.json').read_text())['records']
    lines=(run/'job.inp').read_text().splitlines();curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve);rows=[]
    for index in (0,70,118,171):
        r=records[index];e=np.array(r['emec']);previous=np.array(r['previous_state'])
        c=np.eye(3)+2*tensors(e[None])[0];m=np.eye(3)-2*tensors(previous[None,1:7])[0]
        young,nu=r['elastic'];mu=young/(2*(1+nu));bulk=young/(3*(1-2*nu))
        k=np.zeros((6,6))
        for (i,j),v in zip([(i,j) for j in range(6) for i in range(j+1)],r['stiffness']):k[i,j]=k[j,i]=v
        exact=[];frozen=[];fd=[]
        for col in range(6):
            v=np.eye(6)[col]/(1 if col<3 else 2);dc=2*tensors(v[None])[0]
            exact.append(vector(derivative(c,m,previous[0],curve,mu,bulk,dc)))
            frozen.append(vector(derivative(c,m,previous[0],curve,mu,bulk,dc,True)))
            h=1e-7
            fd.append(vector((update(c+h*dc,m,np.array(previous[0]),curve,mu,bulk)[0]-update(c-h*dc,m,np.array(previous[0]),curve,mu,bulk)[0])/(2*h)))
        exact=np.array(exact).T;frozen=np.array(frozen).T;fd=np.array(fd).T
        rel=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(a))
        rows.append(dict(index=index,plastic=bool(update(c,m,np.array(previous[0]),curve,mu,bulk)[3]),analytic_vs_finite_difference=rel(exact,fd),analytic_vs_native=rel(exact,k),frozen_yield_j_vs_native=rel(frozen,k),analytic_antisymmetric=rel(exact,exact.T)/2,term_rank=int(np.linalg.matrix_rank(exact-frozen,tol=1e-6)),analytic_matrix=exact.tolist(),frozen_yield_j_matrix=frozen.tolist(),native_matrix=k.tolist()))
    result=dict(comparisons=rows,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),parent,Path(__file__).with_name('audit_plastic_return_map.py'),F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f')},run_hashes=evidence['hashes'],run=evidence['run'],limits=dict(analytic_fd_relative=2e-7),limitations='Fixed-history isotropic no-creep local derivative only. Frozen-yield-J is an explanatory counterfactual, not a constitutive or production revision. No assembled residual, pipe causation, energy, cap/contact/MPC or Abaqus acceptance.')
    out=F/'results/output'/('material_linearization_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if 'matrix' not in k} for r in rows],indent=2))
    assert all(r['analytic_vs_finite_difference']<2e-7 for r in rows)
if __name__=='__main__':main(sys.argv[1])
