"""Read CalculiX integration-point histories; never treat nonconvergence as burst."""
import csv
import json
import math
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .config import Case
from .model import Model
from .decks import steps
from .runner import write_json, sha256

POSTPROCESSOR_SHA256 = sha256(__file__)

NUMBER = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?'
HEADER = re.compile(r'for set\s+(\S+)\s+and time\s+('+NUMBER+')', re.I)


def dat_blocks(path):
    """Yield (quantity, set, total time, rows); tolerate blank lines and Fortran exponents."""
    current=None; rows=[]
    with Path(path).open(errors='replace') as f:
        for line in f:
            match=HEADER.search(line)
            if match:
                if current is not None: yield (*current, rows)
                title=line.lower().strip()
                kind=next((k for prefix,k in [('equivalent plastic strain','PEEQ'),('stresses','S'),('strains','E'),
                           ('global coordinates','COORD'),('coordinates','COORD'),('logarithmic strains','LE'),
                           ('displacements','U'),('forces','RF')] if title.startswith(prefix)),None)
                current=(kind,match[1].upper(),float(match[2].replace('D','E')));rows=[]
            elif current is not None and line.strip() and re.match(r'^\s*\d+\s',line):
                try: rows.append([float(v.replace('D','E')) for v in re.findall(NUMBER,line)])
                except ValueError: raise ValueError(f'Cannot parse numerical output: {line[:100]}')
    if current is not None:yield (*current,rows)


def stage_at(case, total_time):
    names=[n for n,_ in steps(case)]
    index=max(0,min(len(names)-1,int(math.ceil(total_time-1e-8))-1))
    stage=names[index]
    pressure=case.analysis.max_pressure*max(0,min(1,total_time-index)) if stage=='pressurise' else 0.
    return stage,pressure


def stress_invariants(values):
    xx,yy,zz,xy,xz,yz=values
    matrix=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
    mean=np.trace(matrix)/3
    vm=float(np.sqrt(1.5*np.sum((matrix-np.eye(3)*mean)**2)))
    return vm,float(mean/vm) if vm>1e-12 else None,float(np.linalg.eigvalsh(matrix)[-1])


def failure_assessment(history, case, completed):
    threshold=case.analysis.failure_peeq
    dent=history[history.stage!='pressurise']
    if len(dent) and dent.max_peeq.max()>=threshold:
        return {'status':'criterion_reached_during_denting','burst_pressure_mpa':None,
                'reason':'The failure strain was reached before pressurisation.'}
    press=history[history.stage=='pressurise']
    if press.empty:return {'status':'no_pressure_results','burst_pressure_mpa':None}
    cross=press[press.max_peeq>=threshold]
    if cross.empty:
        return {'status':'criterion_not_reached' if completed else 'incomplete_before_criterion',
                'burst_pressure_mpa':None,'last_converged_pressure_mpa':float(press.pressure_mpa.iloc[-1]),
                'reason':'No criterion crossing in the available converged results; maximum applied pressure is not burst pressure.'}
    upper=cross.iloc[0]
    previous=press[press.time<upper.time]
    lower_p=float(previous.pressure_mpa.iloc[-1]) if len(previous) else 0.
    return {'status':'criterion_crossed','burst_pressure_mpa':float(upper.pressure_mpa),
            'pressure_bracket_mpa':[lower_p,float(upper.pressure_mpa)],
            'failure_element':int(upper.max_peeq_element),'failure_integration_point':int(upper.max_peeq_ip),
            'criterion':'maximum accumulated integration-point PEEQ >= '+str(threshold),
            'reason':'Reported value is the first sampled crossing; refine pressure increments within the bracket.',
            'simulation_completed':bool(completed),'validation_status':'unvalidated'}


def residual_displacement(history):
    """Only a fully retracted tool gives the unloaded crown displacement."""
    withdrawal=history[(history.stage=='withdraw') & (abs(history.time-2.)<1e-7)]
    return float(withdrawal.crown_inward_displacement_mm.iloc[-1]) if len(withdrawal) else None


def final_frd_displacements(path, accepted_times=None):
    """Last terminated DISP block, optionally restricted to accepted history times."""
    result={};candidate={};active=False
    time=None
    if not Path(path).exists():return result
    with Path(path).open(errors='replace') as f:
        for line in f:
            if line.lstrip().startswith('100CL'):
                time=float(line.split()[2])
            elif line.startswith(' -4'):
                active='DISP' in line
                if active:candidate={}
            elif active and line.startswith(' -1'):
                values=re.findall(NUMBER,line[3:])
                if len(values)>=4:candidate[int(values[0])]=[float(v) for v in values[1:4]]
            elif line.startswith(' -3'):
                if active and (accepted_times is None or
                               (time is not None and any(abs(time-t)<=1e-6+1e-12 for t in accepted_times))):
                    result=candidate
                active=False
    return result


def contact_history(path,case):
    """Read complete nodal CONTACT blocks; FRD COPEN is negative penetration."""
    rows=[];active=False;t=0.;penetration=0.;pressure=0.;minimum_pressure=0.
    if not Path(path).exists():return rows
    with Path(path).open(errors='replace') as f:
        for line in f:
            if line.lstrip().startswith('100CL'):
                t=float(line.split()[2])
            elif line.startswith(' -4'):
                active=line.split()[1]=='CONTACT';penetration=0.;pressure=0.;minimum_pressure=0.
            elif active and line.startswith(' -1'):
                values=re.findall(NUMBER,line[3:])
                if len(values)>=7:
                    penetration=max(penetration,-float(values[1]));pressure=max(pressure,float(values[4]))
                    minimum_pressure=min(minimum_pressure,float(values[4]))
            elif line.startswith(' -3'):
                if active:
                    stage,load=stage_at(case,t)
                    rows.append(dict(time=t,stage=stage,pressure_mpa=load,max_nodal_penetration_mm=penetration,
                                     max_nodal_contact_pressure_mpa=pressure,min_nodal_contact_pressure_mpa=minimum_pressure))
                active=False
    return rows


def extract(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['solver'] != 'calculix': raise ValueError('Only CalculiX is supported')
    case=Case.load(directory/'case.json');model=Model.load(directory/'model.json')
    source=directory/'job.dat'
    status=json.loads((directory/'status.json').read_text())
    output=directory/'results';output.mkdir(exist_ok=True)
    defect=set(model.elsets['DEFECT'])
    centers={e:np.mean([model.nodes[n] for n in model.elements[e][1][:8]],axis=0) for e in model.elsets['PIPE']}
    history=[];node_rows=[];frame=None;last_time=None;discarded=[]
    fields=['time','stage','pressure_mpa','element','ip','peeq','x','y','z','sxx','syy','szz','sxy','sxz','syz',
            'exx','eyy','ezz','exy','exz','eyz','strain_measure','hoop_strain','axial_strain',
            'hoop_stress_mpa','axial_stress_mpa','von_mises_mpa','triaxiality','max_principal_mpa']
    with (output/'integration_points.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        def flush():
            if frame is None:return
            if case.material.plastic and frame['peeq_count']!=8*len(model.elsets['PIPE']):
                discarded.append(last_time)
                return
            stage,pressure=stage_at(case,last_time)
            row={'time':last_time,'stage':stage,'pressure_mpa':pressure,
                 'max_peeq':frame['max'][0],'max_peeq_element':frame['max'][1],'max_peeq_ip':frame['max'][2]}
            ref=model.nsets.get('INDENTER_REF',[None])[0]; crown=model.nsets['CROWN'][0]
            # Compression-positive actuator force; raw global RF stays in nodes.csv.
            row['indenter_travel_mm']=-frame['U'].get(ref,[0,0,0])[1]
            row['indenter_force_n']=-frame['RF'].get(ref,[0,0,0])[1]
            row['crown_inward_displacement_mm']=-frame['U'].get(crown,[0,0,0])[1]
            history.append(row)
            for n,vals in frame['U'].items():
                force=frame['RF'].get(n,[None]*3)
                node_rows.append(dict(time=last_time,stage=stage,pressure_mpa=pressure,node=n,
                                      ux=vals[0],uy=vals[1],uz=vals[2],rfx=force[0],rfy=force[1],rfz=force[2]))
            for (e,ip),data in sorted(frame['ip'].items()):
                record=dict(time=last_time,stage=stage,pressure_mpa=pressure,element=e,ip=ip,peeq=data.get('PEEQ'))
                xyz=data.get('COORD',centers[e]);record.update(zip(['x','y','z'],xyz))
                x,y,_=xyz;norm=math.hypot(x,y);co,si=y/norm,x/norm
                if 'S' in data:
                    record.update(zip(['sxx','syy','szz','sxy','sxz','syz'],data['S']))
                    vm,eta,principal=stress_invariants(data['S'])
                    record.update(von_mises_mpa=vm,triaxiality=eta,max_principal_mpa=principal)
                    xx,yy,zz,xy,_,_=data['S'];record.update(hoop_stress_mpa=co*co*xx+si*si*yy-2*si*co*xy,axial_stress_mpa=zz)
                strain=data.get('E',data.get('LE'))
                if strain is not None:
                    xx,yy,zz,xy,xz,yz=strain
                    record.update(zip(['exx','eyy','ezz','exy','exz','eyz'],strain))
                    # E is material Green-Lagrange; use the undeformed cylindrical basis.
                    if 'E' in data:
                        x,y,_=centers[e];norm=math.hypot(x,y);co,si=y/norm,x/norm
                    record.update(strain_measure='green_lagrange' if 'E' in data else 'spatial_logarithmic',
                                  hoop_strain=co*co*xx+si*si*yy-2*si*co*xy,axial_strain=zz)
                writer.writerow(record)
        for kind,group,t,rows in dat_blocks(source):
            if kind is None:continue
            if last_time is None or abs(t-last_time)>1e-9:
                flush();last_time=t;frame={'max':(0.,0,0),'U':{},'RF':{},'ip':{},'peeq_count':0}
            for row in rows:
                if kind in ('U','RF') and len(row)>=4:
                    frame[kind][int(row[0])]=row[1:4]
                elif kind in ('PEEQ','S','E','LE','COORD') and len(row)>=3:
                    e,ip=int(row[0]),int(row[1])
                    if kind=='PEEQ':
                        frame['peeq_count']+=1
                        if row[2]>frame['max'][0]:frame['max']=(row[2],e,ip)
                        if e in defect:frame['ip'].setdefault((e,ip),{})[kind]=row[2]
                    elif e in defect and kind=='COORD' and len(row)>=5:frame['ip'].setdefault((e,ip),{})[kind]=row[2:5]
                    elif e in defect and len(row)>=8:frame['ip'].setdefault((e,ip),{})[kind]=row[2:8]
        flush()
    if not history:raise ValueError('No recognised converged result frames; inspect solver.log')
    df=pd.DataFrame(history).sort_values('time');df.to_csv(output/'history.csv',index=False)
    pd.DataFrame(node_rows).to_csv(output/'nodes.csv',index=False)
    assessment=failure_assessment(df,case,status['state']=='completed')
    assessment['solver_state']=status['state']
    if status['state']=='running' and assessment['status']=='incomplete_before_criterion':
        assessment['status']='analysis_in_progress'
        assessment['reason']='Only converged output available so far; the solver is still running.'
    # An interrupted withdrawal still has tool loading and is not a residual dent.
    residual=residual_displacement(df)
    assessment['withdrawal_completed']=(residual is not None) if case.dent.enabled else None
    assessment['residual_crown_displacement_mm']=residual
    assessment['residual_depth_definition']='Crown inward displacement relative to the initially gouged surface; not gouge depth plus dent depth.'
    assessment['case']=case.name;assessment['notes']=case.notes
    assessment['discarded_incomplete_frame_times']=discarded
    assessment['maximum_observed_peeq']=float(df.max_peeq.max())
    assessment['minimum_compression_positive_tool_force_n']=float(df.indenter_force_n.min())
    assessment['tensile_tool_force_detected']=bool(df.indenter_force_n.min() < -max(1.,float(df.indenter_force_n.max())*.001))
    assessment['response_quality_issues']=[]
    if assessment['tensile_tool_force_detected']:
        assessment['response_quality_issues'].append('Tensile tool reaction: check contact separation or geometric interlock before using withdrawal and subsequent pressure results.')
    if assessment.get('failure_element') in centers:
        assessment['failure_element_initial_centroid_mm']=centers[assessment['failure_element']].tolist()
    assessment['postprocessor_source']='pipe_sim.results; strain components retain the documented solver-specific measure'
    assessment['postprocessor_sha256']=POSTPROCESSOR_SHA256
    contact=contact_history(directory/'job.frd',case)
    if contact:
        pd.DataFrame(contact).to_csv(output/'contact_history.csv',index=False)
        assessment['max_nodal_contact_penetration_mm']=max(row['max_nodal_penetration_mm'] for row in contact)
        assessment['min_nodal_contact_pressure_mpa']=min(row['min_nodal_contact_pressure_mpa'] for row in contact)
        assessment['contact_penetration_exceeds_wall']=assessment['max_nodal_contact_penetration_mm']>=case.pipe.thickness
        assessment['contact_penetration_fraction_of_wall']=assessment['max_nodal_contact_penetration_mm']/case.pipe.thickness
        assessment['excessive_contact_penetration']=assessment['contact_penetration_fraction_of_wall']>.1
        if assessment['excessive_contact_penetration']:
            assessment['response_quality_issues'].append('Contact penetration exceeds 10% of wall thickness; gross contact failure. Smaller penetration still needs a stiffness study.')
    assessment['contact_checks_passed']=not assessment['response_quality_issues']
    assessment['residual_dent_reliable']=bool(assessment['withdrawal_completed'] and assessment['contact_checks_passed'])
    write_json(output/'summary.json',assessment)
    u=final_frd_displacements(directory/'job.frd', accepted_times=df.time.to_numpy())
    if u:
        model.export_vtu(output/'last_frame.vtu',u)
        from .preview import render_deformation
        render_deformation(case,model,u,output/'deformation.png')
    plot_history(df,output,case)
    plot_probes(output,model,case)
    return assessment


def plot_history(df, output, case):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for stage in ['indent','withdraw']:
        sub=df[df.stage==stage]
        if len(sub):axs[0].plot(sub.indenter_travel_mm,sub.indenter_force_n/1000,label=stage)
    axs[0].set(xlabel='Tool travel including initial clearance (mm)',ylabel='Compressive tool force (kN)')
    if case.dent.enabled:axs[0].legend()
    press=df[df.stage=='pressurise']
    axs[1].plot(press.pressure_mpa,press.max_peeq)
    axs[1].axhline(case.analysis.failure_peeq,color='r',ls='--',label='Provisional failure criterion')
    axs[1].set(xlabel='Internal pressure (MPa)',ylabel='Maximum integration-point PEEQ');axs[1].legend(fontsize=7)
    axs[2].plot(press.pressure_mpa,press.crown_inward_displacement_mm)
    axs[2].set(xlabel='Internal pressure (MPa)',ylabel='Crown inward displacement (mm)')
    for ax in axs:ax.grid(alpha=.25)
    fig.suptitle(case.name+' — unvalidated development results')
    fig.savefig(output/'history.png',dpi=170);fig.savefig(output/'history.pdf');plt.close(fig)


def plot_probes(output,model,case):
    """Track nearby integration points; explicitly expose offsets from requested nodes."""
    import matplotlib.pyplot as plt
    path=output/'integration_points.csv'
    # Stream CSV in chunks to avoid retaining the full IP history for large meshes.
    initial=None
    for chunk in pd.read_csv(path,chunksize=100000):
        first_time=chunk.time.min()
        if initial is None:initial=chunk[chunk.time==first_time].copy();t0=first_time
        else:initial=pd.concat([initial,chunk[chunk.time==t0]],ignore_index=True)
        if chunk.time.max()>t0:break
    if initial is None or initial.empty:return
    locations={}
    for node in model.nsets['GAUGES']:
        distances=np.linalg.norm(initial[['x','y','z']].to_numpy()-np.array(model.nodes[node]),axis=1)
        i=int(np.argmin(distances));row=initial.iloc[i]
        locations[node]=(int(row.element),int(row.ip),float(distances[i]))
    probe_rows=[]
    for chunk in pd.read_csv(path,chunksize=100000):
        for node,(element,ip,distance) in locations.items():
            sub=chunk[(chunk.element==element)&(chunk.ip==ip)].copy()
            sub['requested_node']=node;sub['initial_offset_mm']=distance
            probe_rows.append(sub)
    if not probe_rows:return
    probes=pd.concat(probe_rows,ignore_index=True);probes.to_csv(output/'probes.csv',index=False)
    # Save the final defect-region IP cloud for direct stress/strain inspection.
    last=None
    for chunk in pd.read_csv(path,chunksize=100000):
        maximum=chunk.time.max()
        if last is None or maximum>last.time.iloc[0]:last=chunk[chunk.time==maximum].copy()
        else:last=pd.concat([last,chunk[chunk.time==maximum]],ignore_index=True)
    if last is not None and len(last):
        import meshio
        cloud=last[['x','y','z']].to_numpy()
        attributes={name:last[name].to_numpy() for name in last.select_dtypes(include='number').columns if name not in ('x','y','z')}
        meshio.write_points_cells(output/'integration_points_last.vtu',cloud,[('vertex',np.arange(len(cloud)).reshape(-1,1))],point_data=attributes)
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for node in model.nsets['GAUGES']:
        sub=probes[(probes.requested_node==node)&(probes.stage=='pressurise')].sort_values('time')
        for ax,key in zip(axs,['hoop_strain','axial_strain']):
            ax.plot(sub.pressure_mpa,sub[key],label=f'node {node}')
    for ax,title in zip(axs,['Hoop strain','Axial strain']):
        ax.set(xlabel='Internal pressure (MPa)',ylabel=title);ax.grid(alpha=.25)
    axs[1].legend(fontsize=6)
    fig.suptitle('Nearby integration-point probes — offsets in probes.csv; '+str(initial.strain_measure.iloc[0]))
    fig.savefig(output/'strain_probes.png',dpi=160);fig.savefig(output/'strain_probes.pdf');plt.close(fig)
