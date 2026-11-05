"""Independent C3D20R kinematics/reference energy audit; no model changes."""
import json, sys, re
from pathlib import Path
import numpy as np
from pipe_sim.model import Model, HEX_LOCAL
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_pressure_failures import converged_times, matches_sta
from review_internal_pipe_fields import paths, PARENTS, F
from audit_end_surface_traction import face_nodes, shape8
SCRIPT_SHA256=sha256(__file__)

def shape(q):
    # Standard serendipity basis, independently expressed by reference-node signs.
    out=[]
    for a in HEX_LOCAL:
        if np.all(a):out.append(np.prod(1+a*q)*(a@q-2)/8)
        else:
            axis=int(np.flatnonzero(a==0)[0]);others=[j for j in range(3) if j!=axis]
            out.append((1-q[axis]**2)*np.prod(1+a[others]*q[others])/4)
    return np.array(out)

def source_basis(q):
    # Evaluate literal scalar/basis assignments from retained primary Fortran source;
    # no compilation or solver modification. Compare to independent sign basis.
    source=F/'sym/vendor/CalculiX/ccx_2.23/src/shape20h.f'
    env=dict(xi=q[0],et=q[1],ze=q[2]);values=np.zeros(20);derivatives=np.zeros((20,3))
    for line in source.read_text().splitlines():
        if line.strip()=='do i=1,3':break
        if line.startswith('!'):continue
        match=re.fullmatch(r'\s*(\w+|shp\(4,\s*\d+\)|shpe\([123],\s*\d+\))\s*=\s*(.+)',line)
        if not match:continue
        lhs,expr=match.groups();expr=re.sub(r'(\d+(?:\.\d*)?)d([+-]?\d+)',lambda m:m.group(1)+'e'+m.group(2),expr)
        value=eval(expr,{'__builtins__':{}},env)
        if lhs.startswith('shpe'):
            axis,node=map(int,re.findall(r'\d+',lhs));derivatives[node-1,axis-1]=value
        elif lhs.startswith('shp'):values[int(re.findall(r'\d+',lhs)[1])-1]=value
        else:env[lhs]=value
    return values,derivatives

def spatial(model):
    ids=sorted(model.elements);assert all(model.elements[e][0]=='C3D20R' for e in ids)
    nodes=np.array([model.elements[e][1] for e in ids]);xyz=np.array([[model.nodes[int(n)] for n in ns] for ns in nodes])
    points=np.array([[x,y,z] for z in (-1,1) for y in (-1,1) for x in (-1,1)])/np.sqrt(3)
    # Interpolation, partition, affine reproduction and finite difference checks.
    nodal_error=float(abs(np.array([shape(q) for q in HEX_LOCAL])-np.eye(20)).max());maximum=0.;fd=0.;source_error=0.;grads=[];weights=[]
    for q in points:
        d=np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T
        source_values,source_derivatives=source_basis(q)
        source_error=max(source_error,float(abs(source_values-shape(q)).max()),float(abs(source_derivatives-d).max()))
        numeric=np.array([(shape(q+1e-5*np.eye(3)[j])-shape(q-1e-5*np.eye(3)[j]))/2e-5 for j in range(3)]).T
        fd=max(fd,float(abs(d-numeric).max()))
        maximum=max(maximum,abs(shape(q).sum()-1),float(abs(d.sum(axis=0)).max()),float(abs(HEX_LOCAL.T@d-np.eye(3)).max()))
        jac=np.einsum('eni,nj->eij',xyz,d);det=np.linalg.det(jac);assert np.all(det>0)
        grads.append(np.einsum('ni,eij->enj',d,np.linalg.inv(jac)));weights.append(det)
    assert nodal_error<1e-14 and maximum<1e-14 and fd<1e-9 and source_error<1e-14
    return ids,nodes,np.stack(grads,axis=1),np.stack(weights,axis=1),dict(nodal_error=nodal_error,maximum_primary_source_basis_difference=source_error,primary_source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/shape20h.f'),partition_affine_error=maximum,finite_difference_derivative_error=fd,minimum_reference_jacobian=float(np.min(weights)))

def tensors(v):
    t=np.zeros(v.shape[:-1]+(3,3));t[...,0,0]=v[...,0];t[...,1,1]=v[...,1];t[...,2,2]=v[...,2]
    for k,(i,j) in enumerate(((0,1),(0,2),(1,2)),3):t[...,i,j]=t[...,j,i]=v[...,k]
    return t

def audit(lane):
    if lane=='elastic_half':
        from review_elastic_half_control import paths as half_paths
        tag,r=half_paths(lane)
    elif lane in ('elastic','elastic_quarter'):
        from review_elastic_work_control import paths as elastic_paths, configure
        configure('quarter' if lane=='elastic_quarter' else 'standard');tag,r=elastic_paths(lane)
    else:tag,r=paths(lane)
    tp=F/'results/output'/f'{tag}_terminal.json';term=json.loads(tp.read_text());assert term['completed_native']
    if lane=='elastic_half':
        pp=F/'results/output/elastic_work_control_20261011T093005Z_terminal.json'
    elif lane in ('elastic','elastic_quarter'):
        parent_lane='quarter' if lane=='elastic_quarter' else 'standard'
        pp=F/'results/output'/f'internal_pipe_fields_{parent_lane}_20261011T090414Z_terminal.json'
    else:
        pp=F/'results/output'/f'{PARENTS[lane]}_terminal.json'
    parent=json.loads(pp.read_text());assert sha256(pp)==term['parent_report_sha256']
    if lane=='elastic_half':
        assert all(term['input_hashes'][n]==parent['input_hashes'][n] for n in term['input_hashes'] if n!='job.inp')
        original=(F/parent['run']/'job.inp').read_text()
        assert (r/'job.inp').read_text()==original.replace('0.02291666666666667, 1., 1e-06, 0.02291666666666667','0.011458333333333334, 1., 1e-06, 0.011458333333333334')
    elif lane in ('elastic','elastic_quarter'):
        assert all(term['input_hashes'][n]==parent['input_hashes'][n] for n in ('mesh.inp','model.json'))
        original=(F/parent['run']/'job.inp').read_text()
        assert (r/'job.inp').read_text()==re.sub(r'\*PLASTIC\n[^*]*','',original)
    else:assert term['input_hashes']==parent['input_hashes']
    assert sha256(F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso')==term['solver_sha256']==parent['solver_sha256']
    assert sha256(F/'results/scripts/capture_internal_printout_gdb.py')==term['capture_script_sha256']
    pins=parent.get('pins')
    if pins is None:pins=json.loads((F/'results/output'/f'{PARENTS["standard"]}_terminal.json').read_text())['pins']
    hashes={**term['input_hashes'],**term['output_hashes']};assert all(sha256(r/n)==h for n,h in hashes.items())
    model=Model.load(r/'model.json');literal={};in_nodes=False
    for line in (r/'mesh.inp').read_text().splitlines():
        if line.startswith('**'):continue
        if line.startswith('*'):in_nodes=line.upper().strip()=='*NODE';continue
        if in_nodes and line.strip():
            tokens=line.split(',');literal[int(tokens[0])]=list(map(float,tokens[1:4]))
    assert set(literal)==set(model.nodes)
    coordinate_difference=max(abs(np.array(literal[n])-model.nodes[n]).max() for n in model.nodes)
    model.nodes=literal;ids,nodes,grad,weights,checks=spatial(model)
    checks['maximum_serialized_coordinate_difference_mm']=float(coordinate_difference)
    checks['coordinate_source']='Literal immutable mesh.inp NODE coordinates actually read by native solver; model.json high precision retained unchanged.'
    records=json.loads((r/'fields.json').read_text())['records'];binary=np.memmap(r/'fields.bin',dtype='<f8',mode='r');accepted=converged_times(r/'job.sta')
    frames={}
    for k,g,t,rows in dat_blocks(r/'job.dat'):
        assert matches_sta(accepted,t)
        frames.setdefault(t,{})[(k,g)]=rows
    assert len(frames)==len(records)
    deck=(r/'job.inp').read_text().split('*DLOAD\n')[1].split('*',1)[0]
    magnitudes={(int(e),int(f.strip()[1:])):float(p) for e,f,p in (line.split(',') for line in deck.strip().splitlines())}
    groups={}
    for surface in ('INNER','END_MINUS','END_PLUS'):
        faces=[(face_nodes(model,e,f),magnitudes[e,f]) for e,f in model.surfaces[surface]]
        groups[surface]=(np.array([ns for ns,p in faces]),np.array([-p for ns,p in faces]))
    q,w=np.polynomial.legendre.leggauss(2)
    kernels=[(shape8(a,b),shape8(a+1e-30j,b).imag/1e-30,shape8(a,b+1e-30j).imag/1e-30,wa*wb) for a,wa in zip(q,w) for b,wb in zip(q,w)]
    previousU=np.zeros((records[0]['nodes'],3));previousTime=0.;external={s:0. for s in groups}
    previousS=previousE=np.zeros((len(ids),8,3,3));previousEnergy=np.zeros((len(ids),8));maximum_green=maximum_update=maximum_interval_excess=maximum_time=0.;history=[];interval_by_kind={};worst_interval=None;previousF=np.broadcast_to(np.eye(3),(len(ids),8,3,3)).copy();linear_material_work=0.;maximum_elastic_stress_error=0.;maximum_elastic_energy_error=0.
    def interval(v):return 0. if v==0 else .5*10**(int(np.floor(np.log10(abs(v))))-6)
    for rec,(t,fields) in zip(records,sorted(frames.items())):
        assert matches_sta(accepted,rec['time']) and abs(rec['time']-t)<=interval(t)+1e-14
        maximum_time=max(maximum_time,abs(rec['time']-t));assert rec['elements']>=max(ids) and rec['ips']>=8
        def field(name,width):
            m=rec['fields'][name];v=binary[m['offset']//8:m['offset']//8+m['count']]
            return v.reshape((-1,width)) if name=='u' else v.reshape((rec['elements'],rec['ips'],width))[np.array(ids)-1,:8]
        u=field('u',4)[:,1:];assert all(u[n-1,d]==0 for n,d in pins)
        for surface,(ns,pressure) in groups.items():
            start=np.array([[model.nodes[int(n)] for n in face] for face in ns])+previousU[ns-1]
            delta=u[ns-1]-previousU[ns-1]
            for a,wtime in zip((.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)),(.5,.5)):
                points=start+a*delta
                for N,dr,ds,wspace in kernels:
                    normal=np.cross(np.einsum('n,fni->fi',dr,points),np.einsum('n,fni->fi',ds,points))
                    motion=np.einsum('n,fni->fi',N,delta)
                    external[surface]+=wtime*wspace*float(np.sum(pressure*(previousTime+a*(rec['time']-previousTime))*np.einsum('fi,fi->f',normal,motion)))
        deformation=np.eye(3)+np.einsum('eni,epnj->epij',u[nodes-1],grad);J=np.linalg.det(deformation);assert np.all(J>0);inv=np.linalg.inv(deformation)
        sigma=tensors(field('stress',6));S=J[...,None,None]*(inv@sigma@np.swapaxes(inv,-1,-2));E=tensors(field('strain',6))
        green=(np.swapaxes(deformation,-1,-2)@deformation-np.eye(3))/2
        maximum_green=max(maximum_green,float(abs(E-green).max()));energy=field('energy',2)[...,0]
        if lane in ('elastic','elastic_quarter','elastic_half'):
            case=json.loads((r/'case.json').read_text());young=case['material']['young'];nu=case['material']['poisson']
            mu=young/(2*(1+nu));lam=young*nu/((1+nu)*(1-2*nu));trace=np.trace(E,axis1=-2,axis2=-1)
            expectedS=lam*trace[...,None,None]*np.eye(3)+2*mu*E
            expectedEnergy=.5*lam*trace**2+mu*np.einsum('epij,epij->ep',E,E)
            maximum_elastic_stress_error=max(maximum_elastic_stress_error,float(abs(S-expectedS).max()))
            maximum_elastic_energy_error=max(maximum_elastic_energy_error,float(abs(energy-expectedEnergy).max()))
        predicted=previousEnergy+.5*np.einsum('epij,epij->ep',S+previousS,E-previousE)
        difference=energy-predicted;maximum_update=max(maximum_update,float(abs(difference).max()))
        integrated=energy*weights;total=float(integrated.sum());update_gap=float((difference*weights).sum())
        deltaF=deformation-previousF;linear_increment=0.
        for a,w in zip((.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)),(.5,.5)):
            Fa=previousF+a*deltaF;dE=(np.swapaxes(Fa,-1,-2)@deltaF+np.swapaxes(deltaF,-1,-2)@Fa)/2
            Sa=previousS+a*(S-previousS)
            linear_increment+=w*float((np.einsum('epij,epij->ep',Sa,dE)*weights).sum())
        linear_material_work+=linear_increment
        def note(kind,actual,value,identity):
            nonlocal maximum_interval_excess,worst_interval
            excess=abs(actual-value)-interval(value)
            interval_by_kind[kind]=max(interval_by_kind.get(kind,0.),excess)
            if excess>maximum_interval_excess:
                maximum_interval_excess=excess;worst_interval=dict(kind=kind,time=rec['time'],identity=identity,raw=float(actual),printed=float(value),half_decimal_unit=interval(value),excess=float(excess))
        for (kind,group),rows in fields.items():
            if kind=='U':
                for row in rows:
                    for actual,value in zip(u[int(row[0])-1],row[1:]):note('U',actual,value,int(row[0]))
            elif kind in ('S','E'):
                raw=field('stress' if kind=='S' else 'strain',6);lookup={e:i for i,e in enumerate(ids)}
                for row in rows:
                    for actual,value in zip(raw[lookup[int(row[0])],int(row[1])-1],row[2:]):note(kind,actual,value,[int(row[0]),int(row[1])])
            elif kind is None:
                lookup={e:i for i,e in enumerate(ids)}
                for row in rows:note('ELSE',float(integrated[lookup[int(row[0])]].sum()),row[1],int(row[0]))
        history.append(dict(time=rec['time'],internal_energy_n_mm=total,linear_output_path_material_work_n_mm=linear_material_work,linear_material_minus_native_energy_n_mm=linear_material_work-total,external_work_by_surface_n_mm=dict(external),external_minus_native_n_mm=sum(external.values())-total,external_minus_linear_material_n_mm=sum(external.values())-linear_material_work,endpoint_average_update_gap_n_mm=update_gap,max_green_difference=float(abs(E-green).max()),maximum_peeq=max((row[-1] for (k,g),rows in fields.items() if k=='PEEQ' for row in rows),default=0. if lane in ('elastic','elastic_quarter','elastic_half') else float('nan'))))
        previousS,previousE,previousEnergy=S,E,energy;previousF=deformation;previousU=u.copy();previousTime=rec['time']
    if lane in ('elastic','elastic_quarter','elastic_half'):assert all(f['maximum_peeq']==0 for f in history) and maximum_elastic_stress_error<1e-8 and maximum_elastic_energy_error<1e-10
    passed=maximum_green<1e-10 and maximum_update<1e-10 and maximum_interval_excess<1e-7
    parentr=F/parent['run'];assert all(sha256(parentr/n)==h for n,h in {**parent['input_hashes'],**parent['output_hashes']}.items())
    parent_frames={}
    crown_node=model.nsets['CROWN'][0]
    for kind,group,t,rows in dat_blocks(parentr/'job.dat'):
        if kind=='U':
            for row in rows:
                if int(row[0])==crown_node:parent_frames.setdefault(t,{})['crown']=-row[2]
        if kind=='PEEQ':parent_frames.setdefault(t,{})['peeq']=max(row[-1] for row in rows)
    crown_difference=peeq_difference=0.
    for t,fields in frames.items():
        if t not in parent_frames:continue
        crowns=[-row[2] for (k,g),rows in fields.items() if k=='U' for row in rows if int(row[0])==crown_node]
        if crowns:crown_difference=max(crown_difference,abs(crowns[0]-parent_frames[t]['crown']))
        peeq=max((row[-1] for (k,g),rows in fields.items() if k=='PEEQ' for row in rows),default=0. if lane in ('elastic','elastic_quarter','elastic_half') else float('nan'))
        peeq_difference=max(peeq_difference,abs(peeq-parent_frames[t].get('peeq',0. if lane=='elastic_half' else float('nan'))))
    assert all(sha256(r/n)==h for n,h in hashes.items())
    return dict(maximum_elastic_material_stress_difference_mpa=maximum_elastic_stress_error,maximum_elastic_conservative_energy_density_difference=maximum_elastic_energy_error,elastic_peeq_note='Native omits PEEQ when no plastic state is allocated; zero here denotes nonplastic constitutive control, verified by elastic S(E)/energy identities.' if lane in ('elastic','elastic_quarter','elastic_half') else None,maximum_parent_crown_difference_mm=crown_difference,maximum_parent_peeq_difference=peeq_difference,parent_response_comparison='Matched printed times; elastic response difference is intentional constitutive sensitivity.',interval_excess_by_kind=interval_by_kind,worst_interval=worst_interval,run=term['run'],accepted_frames=len(records),field_checks_passed=passed,green_check_limit=1e-10,density_update_limit=1e-10,literal_interval_numerical_slack=1e-7,identical_parent_dat=(r/'job.dat').read_bytes()==(parentr/'job.dat').read_bytes(),spatial_checks=checks,maximum_green_difference=maximum_green,maximum_endpoint_average_density_update_difference=maximum_update,maximum_literal_interval_excess=maximum_interval_excess,maximum_time_rounding=maximum_time,history=history,terminal_sha256=sha256(tp),input_output_hashes=hashes,script_sha256=SCRIPT_SHA256,sources=['shape20h.f:72-160 standard serendipity basis','gauss.f:95-110,324 native IP order and unit weights','printoutelem.f:328-331,428 reference-volume energy','resultsmech.f:1064-1074,1203-1247 material work and Cauchy pushforward'],verdict='Spatial identities and positive reference Jacobians passed; captured Green/literal checks '+('PASS' if passed else 'FAIL')+'. Endpoint-average temporal update discrepancy quantified without acceptance fitting.',limitations='Output frames may not expose native material initial stress/strain state or every energy update; discrepancy is not physical work convergence. No cap/contact/MPC/Abaqus or experimental validation.')
if __name__=='__main__':
    report=audit(sys.argv[1]);out=F/'results/output'/f'internal_pipe_work_verified_{sys.argv[1]}_20261011T093005Z.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('history','input_output_hashes')},indent=2))
