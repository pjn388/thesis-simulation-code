"""Independent retained native coupon tangent checks for compiled runtime kernel."""
import ctypes,json
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256,tensors,update
from audit_material_linearization import derivative,vector

def main():
    gate=F/'results/output/native_material_kernel_v2_20261019.json'
    report=json.loads(gate.read_text());assert report['checks_passed']
    for name,value in report['sources'].items():assert sha256(F/name)==value
    lib=F/report['run']/'kernel.so';fn=ctypes.CDLL(str(lib)).thesis_material_delta
    ptr=ctypes.POINTER(ctypes.c_double)
    fn.argtypes=[ctypes.c_int,ptr,ptr,*([ctypes.c_double]*4),ctypes.c_int,ptr,ptr,ptr];fn.restype=ctypes.c_int
    parent=F/'results/output/material_tangent_20261017T040006Z.json'
    ev=json.loads(parent.read_text());root=F/ev['run']
    for name,value in ev['hashes'].items():assert sha256(root/name)==value
    records=json.loads((root/'tangent.json').read_text())['records']
    deck=(root/'job.inp').read_text()
    curve=np.array([list(map(float,l.split(','))) for l in deck.split('*PLASTIC\n')[1].split('*')[0].strip().splitlines()])
    ep=np.ascontiguousarray(curve[:,1]);ys=np.ascontiguousarray(curve[:,0]);rows=[]
    for index,r in enumerate(records):
        c=np.ascontiguousarray(np.eye(3)+2*tensors(np.array([r['emec']]))[0])
        m=np.ascontiguousarray(np.eye(3)-2*tensors(np.array([r['previous_state'][1:7]]))[0])
        q=r['previous_state'][0];qp=r['current_state'][0]
        E,nu=r['elastic'];mu=E/(2*(1+nu));bulk=E/(3*(1-2*nu));out=np.empty((6,6))
        snapshots=[a.tobytes() for a in (c,m,ep,ys)]
        assert fn(1,c.ctypes.data_as(ptr),m.ctypes.data_as(ptr),q,qp,mu,bulk,len(ep),ep.ctypes.data_as(ptr),ys.ctypes.data_as(ptr),out.ctypes.data_as(ptr))==0
        assert snapshots==[a.tobytes() for a in (c,m,ep,ys)]
        k=np.zeros((6,6))
        for (i,j),v in zip([(i,j) for j in range(6) for i in range(j+1)],r['stiffness']):k[i,j]=k[j,i]=v
        exact=[]
        for col in range(6):
            dc=2*tensors((np.eye(6)[col]/(1 if col<3 else 2))[None])[0]
            exact.append(vector(derivative(c,m,q,curve,mu,bulk,dc)))
        exact=np.array(exact).T
        rows.append(dict(index=index,time=r['time'],plastic=qp>q,
          native_corrected_relative=float(np.linalg.norm(k+out-exact)/np.linalg.norm(exact)),
          parent_relative=float(np.linalg.norm(k-exact)/np.linalg.norm(exact))))
    passed=all(r['native_corrected_relative']<2e-7 for r in rows)
    result=dict(checks_passed=passed,relative_limit=2e-7,records=len(rows),plastic_records=sum(r['plastic'] for r in rows),
      maximum_native_corrected_relative=max(r['native_corrected_relative'] for r in rows),maximum_parent_relative=max(r['parent_relative'] for r in rows),rows=rows,
      sources={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),gate,parent,lib,root/'tangent.json',root/'job.inp')},
      limitations='Retained coupon actual packed native tangents plus compiled correction, fixed-history local derivative only. No live solver linkage, disabled-parent whole-solver identity, state restoration, full-ray native sparse FD, convergence benefit or physical acceptance.',
      next='Integrate kernel into isolated native volume path; enforce supported scope and restoration gates before pressure successor.')
    out=F/'results/output/native_material_coupon_20261019.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','sources')},indent=2),flush=True)
    assert passed,'Native coupon kernel correction failed; report preserved'

if __name__=='__main__':main()
