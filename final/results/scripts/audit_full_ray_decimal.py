"""Decimal50 selected-IP retained replay; no solver or acceptance changes."""
import json
from decimal import Decimal as D, getcontext
import numpy as np
from audit_full_ray_local_replay import F, shape, points, sha256
getcontext().prec = 50

def arr(x):
    return np.array([D(str(v)) for v in np.asarray(x).flat], dtype=object).reshape(np.shape(x))
I=arr(np.eye(3))
def det(a):
    return sum(a[0,j]*(a[1,(j+1)%3]*a[2,(j+2)%3]-a[1,(j+2)%3]*a[2,(j+1)%3]) for j in range(3))
def inv(a):
    return np.array([[(-1)**(i+j)*det2(np.delete(np.delete(a,j,0),i,1)) for j in range(3)] for i in range(3)],dtype=object)/det(a)
def det2(a): return a[0,0]*a[1,1]-a[0,1]*a[1,0]
def tensor(v):
    return np.array([[v[0],v[3],v[4]],[v[3],v[1],v[5]],[v[4],v[5],v[2]]],dtype=object)
def vec(a): return np.array([a[0,0],a[1,1],a[2,2],a[0,1],a[0,2],a[1,2]])
def norm(a): return float(sum(x*x for x in a.flat).sqrt())
def stress(c,metric,p,curve):
    mu=D(208000)/D('2.6'); bulk=D(208000)/D('1.2')
    j=det(c).sqrt(); ci=inv(c); j23=j**(D(2)/3)
    tr=sum((c*metric).flat)/3/j23
    trial=mu*(metric/j23-tr*ci); sc=trial@c
    n=sum((sc*sc.T).flat).sqrt(); c2=(D(2)/3).sqrt()
    def flow(p):
        if p<=curve[0,1]: return curve[0,0]
        for k in range(1,len(curve)):
            if p<=curve[k,1]:
                a,b=curve[k-1],curve[k]
                return a[0]+(b[0]-a[0])*(p-a[1])/(b[1]-a[1])
        return curve[-1,0]
    f=n-c2*j*flow(p); assert f>0
    lo=D(0); hi=f/(2*mu*tr)
    for _ in range(180):
        cop=(lo+hi)/2
        if n-c2*j*flow(p+c2*cop)-2*mu*tr*cop>0: lo=cop
        else: hi=cop
    return bulk/2*(j*j-1)*ci+(1-2*mu*tr*((lo+hi)/2)/n)*trial

def main():
    root=F/'sym/runs/investigation'; bp=root/'volume_material_capture_v5_20261019/assembly.json'
    tp=root/'late_local_inputs_20261019/assembly.json'
    base=json.loads(bp.read_text()); table=json.loads(tp.read_text())
    curve=arr(np.array(table['ips'][0]['plconloc'][:400]).reshape(-1,2)[:,::-1]); del table
    xyz=arr(np.array(base['coordinates']).reshape(-1,3)); u=arr(np.array(base['displacement_after']).reshape(-1,4)[:,1:])
    direction=json.loads((bp.parent/'full_direction.json').read_text()); mapping=np.array(base['mapping']).reshape(-1,4)[:,1:]
    du=np.zeros_like(xyz); du[:]=D(0); du[mapping>0]=arr(direction)[mapping[mapping>0]-1]
    old=arr(np.array(base['previous_state']).reshape(base['ne'],8,base['nstate']))
    local={}; rows=[]; sources=[bp,tp,bp.parent/'full_direction.json',__file__]
    for e,ip in [(238,2),(1441,4),(2365,3)]:
        k=base['ipkon'][e-1]; nodes=np.array(base['connectivity'][k:k+20])-1
        q=points()[ip-1].astype(complex)
        natural=arr(np.array([shape(q+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T)
        grad=natural@inv(xyz[nodes].T@natural)
        local[e,ip]=(nodes,grad,I-2*tensor(old[e-1,ip-1,1:7]),old[e-1,ip-1,0])
    frames={}
    for h in (1e-6,-1e-6,5e-7,-5e-7):
        path=root/f'volume_material_full_ray_fd_20261019/h{h}/assembly.json'; sources.append(path)
        frame=json.loads(path.read_text()); uf=arr(np.array(frame['displacement_after']).reshape(-1,4)[:,1:]); frames[h]={}
        for key,(nodes,grad,metric,p) in local.items():
            def calc(v):
                f=I+v.T@grad
                return vec(stress(f.T@f,metric,p,curve))
            frames[h][key]=(calc(u[nodes]+D(str(h))*du[nodes]),calc(uf[nodes]))
        del frame
    replay=json.loads((F/'results/output/volume_material_full_ray_local_replay_20261019.json').read_text())
    for key in local:
        for h in (1e-6,5e-7):
            ideal=(frames[h][key][0]-frames[-h][key][0])/(2*D(str(h)))
            retained=(frames[h][key][1]-frames[-h][key][1])/(2*D(str(h)))
            r=next(r for r in replay['rows'] if (r['element'],r['ip'],r['h'])==(*key,h))
            analytic=arr(r['analytic_action'])
            rows.append(dict(element=key[0],ip=key[1],h=h,ideal_vs_analytic=norm(ideal-analytic)/norm(analytic),retained_vs_ideal=norm(retained-ideal)/norm(ideal),ideal_action=[float(x) for x in ideal]))
    out=dict(kind='decimal50_selected_ip',precision=50,rows=rows,acceptance=False,sources={str(__import__('pathlib').Path(p).relative_to(F)):sha256(p) for p in sources},limitations='Three IPs only; natural shape derivatives originate in double precision. Decimal ideal ray differs from already rounded native displacement frames. No global FD acceptance, native branch identity, rollback or pressure benefit.',next='Assess ideal-ray versus rounded-frame differentiation before selecting any native experiment; unchanged global gate remains failed.')
    path=F/'results/output/volume_material_full_ray_decimal_20261019.json'
    with path.open('x') as f: json.dump(out,f,indent=2); f.write('\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
