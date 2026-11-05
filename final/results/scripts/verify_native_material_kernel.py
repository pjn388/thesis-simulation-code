"""Compile and check native runtime derivative kernel, without live solver edits."""
import ctypes, json, os, subprocess
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F, sha256, tensors, update
from audit_material_linearization import derivative, vector

def main():
    root=F/'sym/runs/investigation/native_material_kernel_v2_20261019'
    root.mkdir(exist_ok=False)
    source=Path(__file__).with_name('native_material_kernel.c');lib=root/'kernel.so'
    cmd=['gcc','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',str(source),'-lm','-o',str(lib)]
    with (root/'build.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    fn=ctypes.CDLL(str(lib)).thesis_material_delta
    ptr=ctypes.POINTER(ctypes.c_double)
    fn.argtypes=[ctypes.c_int,ptr,ptr,*([ctypes.c_double]*4),ctypes.c_int,ptr,ptr,ptr]
    fn.restype=ctypes.c_int
    def invoke(c,m,q,qp,mu,bulk,curve,enabled=1):
        args=[np.ascontiguousarray(x,dtype='d') for x in (c,m,curve[:,1],curve[:,0])]
        before=[x.tobytes() for x in args];out=np.empty((6,6))
        status=fn(enabled,args[0].ctypes.data_as(ptr),args[1].ctypes.data_as(ptr),q,qp,mu,bulk,len(curve),args[2].ctypes.data_as(ptr),args[3].ctypes.data_as(ptr),out.ctypes.data_as(ptr))
        assert before==[x.tobytes() for x in args], 'Kernel modified constitutive inputs'
        return status,out
    gate=F/'results/output/late_native_input_audit_20261019.json'
    assert json.loads(gate.read_text())['checks_passed']
    native=F/'sym/runs/investigation/late_local_inputs_20261019/assembly.json'
    base=json.loads(native.read_text()); rows=[]
    for r in base['ips']:
        c=np.eye(3)+2*tensors(np.array([r['strain']]))[0]
        m=np.eye(3)-2*tensors(np.array([r['previous'][1:7]]))[0]
        E,nu=r['elconloc'];mu=E/(2*(1+nu));bulk=E/(3*(1-2*nu))
        curve=np.array(r['plconloc'][:400]).reshape(-1,2)[:,::-1]
        q=r['previous'][0];qp=r['current'][0]
        status,k=invoke(c,m,q,qp,mu,bulk,curve);assert status==0
        exact=[];frozen=[];fd=[]
        for col in range(6):
            dc=2*tensors((np.eye(6)[col]/(1 if col<3 else 2))[None])[0]
            exact.append(vector(derivative(c,m,q,curve,mu,bulk,dc)))
            frozen.append(vector(derivative(c,m,q,curve,mu,bulk,dc,True)))
            h=1e-7
            fd.append(vector((update(c+h*dc,m,np.array(q),curve,mu,bulk)[0]-update(c-h*dc,m,np.array(q),curve,mu,bulk)[0])/(2*h)))
        exact=np.array(exact).T;frozen=np.array(frozen).T;fd=np.array(fd).T
        parent=(frozen+frozen.T)/2;revised=parent+k
        off,zero=invoke(c,m,q,qp,mu,bulk,curve,0)
        rows.append(dict(element=r['element'],ip=r['ip'],native_qp=qp,
          analytic_relative=float(np.linalg.norm(revised-exact)/np.linalg.norm(exact)),
          fd_relative=float(np.linalg.norm(revised-fd)/np.linalg.norm(fd)),
          disabled_zero=off==0 and np.array_equal(zero,np.zeros((6,6)))))
    # Elastic identity, including the zero-deviatoric-norm branch.
    elastic=[]
    for c in (np.eye(3),np.diag([1.001,.999,1.002])):
        status,k=invoke(c,np.eye(3),0,0,80000.,170000.,curve)
        elastic.append(bool(status==0 and np.max(abs(k))<1e-8))
    # Disabled branch is a true no-op even with no usable material pointers.
    out=np.ones((6,6));disabled_null=fn(0,None,None,0,0,0,0,0,None,None,out.ctypes.data_as(ptr))==0 and not out.any()
    # Fail closed on unsupported invalid data; no finite correction accepted.
    status,_=invoke(-np.eye(3),np.eye(3),0,0,80000.,170000.,curve)
    invalid_rejected=status!=0
    passed=all(r['analytic_relative']<2e-7 and r['fd_relative']<2e-7 and r['disabled_zero'] for r in rows) and all(elastic) and disabled_null and invalid_rejected
    report=dict(checks_passed=passed,relative_limit=2e-7,rows=rows,
      maximum_analytic_relative=max(r['analytic_relative'] for r in rows),maximum_fd_relative=max(r['fd_relative'] for r in rows),
      read_only_inputs=True,elastic_zero_correction=elastic,disabled_null_inputs=disabled_null,invalid_geometry_rejected=invalid_rejected,
      run=str(root.relative_to(F)),build_command=cmd,
      sources={str(p.relative_to(F)):sha256(p) for p in (source,Path(__file__),lib,gate,native,Path(__file__).with_name('audit_material_linearization.py'),Path(__file__).with_name('audit_plastic_return_map.py'))},
      limitations='Compiled local kernel only; native returned PEEQ supplied, isotropic monotone hardening/no creep/thermal/backstress. Read-only local inputs do not verify solver residual/state restoration. Not linked into live solver; no native full-ray FD, live sparse assembly, convergence benefit or physical burst acceptance.',
      next='Wire kernel to isolated native incplas/results/volume assembly with explicit supported-scope guards and opt-in switch. Preserve production ABI. Require disabled-parent identity, coupon outputs, residual/state restoration and full-direction native sparse action/FD before any pressure successor.')
    with (F/'results/output/native_material_kernel_v2_20261019.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','sources')},indent=2),flush=True)
    assert passed,'Native runtime kernel gate failed; report preserved'

if __name__=='__main__':main()
