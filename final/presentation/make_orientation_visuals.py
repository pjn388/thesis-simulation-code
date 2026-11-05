"""Presentation figures from frozen numerical evidence and actual FRD displacement.

No simulation, fitting, invented geometry or threshold changes. Run with
final/sym/.venv/bin/python final/presentation/make_orientation_visuals.py
"""
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyBboxPatch, Ellipse, ConnectionPatch

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
SNAP = FINAL / 'thesis/evidence/results_20261022T080154Z'
RUN = FINAL / 'sym/runs/campaign/full_zhao2021_p1_adaptive'
sys.path.insert(0, str(FINAL / 'sym/src'))
from pipe_sim.model import Model, HEX_FACES, HEX_LOCAL

OUT = HERE / 'assets'
REPORT = {'snapshot': str(SNAP.relative_to(FINAL)), 'sources': {}, 'figures': {}}
ANGLES = list(range(0, 91, 15))
BLUE, ORANGE = '#176493', '#be561b'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    REPORT['sources'][str(path.relative_to(FINAL))] = sha(path)


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=220, bbox_inches='tight')
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    REPORT['figures'][name] = {ext: sha(OUT / (name + '.' + ext)) for ext in ['png','pdf']}


def style_axes(ax):
    ax.set(xlim=(-5, 95), ylim=(16, 27), xticks=ANGLES,
           xlabel='Gouge angle from pipe axis (degrees)',
           ylabel='Pressure at first sampled PEEQ = 0.8 crossing (MPa)')
    ax.grid(axis='y', alpha=.18)


def markers(ax, records, color, marker='o'):
    for row in records:
        if row['upper_mpa'] is not None:
            lo, hi = row['lower_mpa'], row['upper_mpa']
            ax.errorbar(row['angle_deg'], hi, yerr=[[hi-lo],[0]],
                        fmt=marker, color=color, capsize=6, markersize=8,
                        elinewidth=2, markeredgewidth=1.5)


def plot_criteria():
    p = SNAP / 'pressure005_comparison.csv'; record(p)
    rows = pd.read_csv(p)
    p1 = rows[(rows.specimen == 'P1') & rows.refined].iloc[0]
    records = [dict(angle_deg=0, case_id=p1.case_id,
                    lower_mpa=float(p1.lower_pressure_mpa), upper_mpa=float(p1.criterion_pressure_mpa),
                    endpoint_mpa=float(p1.last_pressure_mpa), endpoint_peeq=float(p1.max_peeq),
                    status='criterion_crossed')]
    for angle in ANGLES[1:]:
        path = SNAP / f'combined{angle}_pressure005_audit.json'; record(path)
        data = json.loads(path.read_text())
        selected = [r for r in data['cases'] if 'pressure005' in r['case_id']]
        assert len(selected) == 1
        r = selected[0]; bracket = r['bracket_mpa']
        assert r['contact_pass'] and r['withdrawal_completed']
        if bracket:
            assert r['bracket_peeq'][0] < .8 <= r['bracket_peeq'][1]
        records.append(dict(angle_deg=angle, case_id=r['case_id'],
                            lower_mpa=bracket[0] if bracket else None,
                            upper_mpa=bracket[1] if bracket else None,
                            endpoint_mpa=r['endpoint_mpa'], endpoint_peeq=r['max_peeq'],
                            status='criterion_crossed' if bracket else 'no_crossing_observed'))
    pd.DataFrame(records).to_csv(OUT / 'orientation_peeq08_data.csv', index=False)
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    fig.subplots_adjust(left=.10, right=.98, top=.81, bottom=.23)
    style_axes(ax); markers(ax, records, ORANGE)
    for r in records:
        if r['upper_mpa'] is not None:
            ax.annotate(f"{r['lower_mpa']:.3f}–{r['upper_mpa']:.3f}",
                        (r['angle_deg'], r['upper_mpa']), xytext=(0,15),
                        textcoords='offset points', ha='center', fontsize=11, color=ORANGE)
        else:
            ax.text(r['angle_deg'], .15, 'No crossing\nobserved', transform=ax.get_xaxis_transform(),
                    ha='center', color='#666666', fontsize=11,
                    bbox=dict(boxstyle='round,pad=.35',fc='#f1f3f5',ec='none'))
    fig.text(.10,.95,'Gouge orientation versus provisional criterion pressure',fontsize=20,weight='bold')
    fig.text(.10,.89,'Dent + gouge  |  X52 baseline  |  Pressure-refined C3D20R cases',fontsize=13,color=ORANGE)
    fig.text(.10,.11,'Dots: first accepted crossing frame. Whiskers: preceding-to-crossing pressure interval, not confidence.',fontsize=10)
    fig.text(.10,.065,'75° and 90° stopped below the criterion; their endpoints are not plotted as failure pressures.',fontsize=10)
    fig.text(.10,.02,'Provisional numerical response, not validated physical burst. Source: 22 Oct 2026 frozen thesis evidence.',fontsize=10,color='#555555')
    save(fig,'orientation_peeq08')

    # A separate uniform parent-family selection prevents mixing refined and parent comparisons.
    path = SNAP / 'orientation_histories.csv'; record(path)
    histories = pd.read_csv(path)
    parent_rows = []
    fig, axes = plt.subplots(1,2,figsize=(14,6.7),sharey=True)
    fig.subplots_adjust(left=.08,right=.98,top=.79,bottom=.24,wspace=.15)
    for ax, combined, color in zip(axes,[False,True],[BLUE,ORANGE]):
        group=[]
        for angle in ANGLES:
            cid = ('full_zhao2021_p1_adaptive' if combined and angle==0 else
                   f'orientation_{"combined" if combined else "gouge"}_{angle:02d}_adaptive')
            d=histories[histories.case_id==cid].sort_values('time').reset_index(drop=True)
            assert len(d)>1 and np.all(np.diff(d.pressure_mpa)>0)
            crossings=np.flatnonzero(d.max_peeq.to_numpy()>=.8)
            lo=hi=None
            if len(crossings):
                k=int(crossings[0]);assert k>0 and d.iloc[k-1].max_peeq<.8
                lo=float(d.iloc[k-1].pressure_mpa);hi=float(d.iloc[k].pressure_mpa)
            row=dict(angle_deg=angle,case_id=cid,combined=combined,lower_mpa=lo,upper_mpa=hi,
                     endpoint_mpa=float(d.iloc[-1].pressure_mpa),endpoint_peeq=float(d.iloc[-1].max_peeq))
            group.append(row);parent_rows.append(row)
        style_axes(ax);markers(ax,group,color)
        ax.set_title('Dent + gouge' if combined else 'Gouge only',color=color,weight='bold')
        ax.text(.02,.03,'No crossing at: '+', '.join(f"{r['angle_deg']}°" for r in group if r['upper_mpa'] is None),
                transform=ax.transAxes,fontsize=10,color='#555555')
    axes[1].set_ylabel('')
    fig.text(.08,.94,'Component comparison: retained parent calculations',fontsize=20,weight='bold')
    fig.text(.08,.88,'Same parent selection as the frozen orientation histories; no lines fitted across missing outcomes.',fontsize=12)
    fig.text(.08,.115,'Markers are first sampled PEEQ = 0.8 crossings; whiskers show load-step brackets, not confidence.',fontsize=10)
    fig.text(.08,.065,'Gouge-only has one crossing (60°). Missing pressures prevent a complete dent-contribution or strength ranking.',fontsize=10)
    fig.text(.08,.02,'Not validated physical burst. These parent results are separate from the pressure-refined combined plot.',fontsize=10,color='#555555')
    save(fig,'orientation_peeq08_parent_families')
    pd.DataFrame(parent_rows).to_csv(OUT/'orientation_peeq08_parent_data.csv',index=False)
    REPORT['criterion_records']=records
    REPORT['parent_criterion_records']=parent_rows


def displacement_at_two():
    """Read exactly the complete native DISP block at withdrawal time 2.0."""
    path=RUN/'job.frd'; before=path.stat();time=None;active=False;frame={};hasher=hashlib.sha256()
    numeric=re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?')
    with path.open('rb') as stream:
        for raw in stream:
            line=raw.decode('ascii')
            if line.lstrip().startswith('100CL'):
                time=float(line.split()[2])
            elif line.startswith(' -4'):
                active=(line.split()[1]=='DISP' and time is not None and abs(time-2)<1e-8)
                if active:hasher.update(raw)
            elif active:
                hasher.update(raw)
                if line.startswith(' -1'):
                    v=numeric.findall(line[3:]);frame[int(v[0])]=np.array(list(map(float,v[1:4])))
                elif line.startswith(' -3'):
                    break
        else:raise ValueError('No complete time=2 displacement block')
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    REPORT['native_displacement']={'source':str(path.relative_to(FINAL)),'time':2.0,
        'stage':'withdrawal complete; zero pressure','displacement_scale':1,
        'record_count':len(frame),'disp_block_sha256':hasher.hexdigest(),
        'source_size':before.st_size,'source_mtime_ns':before.st_mtime_ns,
        'hash_scope':'Complete selected DISP block only; no full FRD hash claimed.'}
    return frame


def face_triangles(model, element, face, displacement):
    kind,nodes=model.elements[element];assert kind=='C3D20R'
    corners=HEX_FACES[face];boundary=[]
    for i,j in zip(corners,corners[1:]+corners[:1]):
        boundary.append(nodes[i]);local=(HEX_LOCAL[i]+HEX_LOCAL[j])/2
        k=np.flatnonzero(np.all(HEX_LOCAL==local,axis=1))[0];boundary.append(nodes[k])
    xyz=np.array([model.nodes[n] for n in boundary]);u=np.array([displacement[n] for n in boundary])
    centre=-.25*xyz[::2].sum(axis=0)+.5*xyz[1::2].sum(axis=0)
    uc=-.25*u[::2].sum(axis=0)+.5*u[1::2].sum(axis=0)
    for i in range(len(boundary)):
        j=(i+1)%len(boundary)
        yield np.array([xyz[i]+u[i],xyz[j]+u[j],centre+uc])[:,[2,0,1]],float(-(u[i,1]+u[j,1]+uc[1])/3)


def clip_polygon(poly, axis, bound, keep_less):
    """Intersect the displayed surface with a plane; never alter native geometry."""
    output = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ina = a[axis] <= bound if keep_less else a[axis] >= bound
        inb = b[axis] <= bound if keep_less else b[axis] >= bound
        if ina:
            output.append(a)
        if ina != inb:
            output.append(a + (b-a) * ((bound-a[axis])/(b[axis]-a[axis])))
    return np.array(output)


def render_pipe():
    for f in ['model.json','case.json','manifest.json','results/history.csv']:record(RUN/f)
    manifest=json.loads((RUN/'manifest.json').read_text())
    for f in ['model.json','case.json']:assert sha(RUN/f)==manifest['files'][f]
    provenance=json.loads((SNAP/'provenance.json').read_text())['sources']
    for f in ['case.json','manifest.json','results/history.csv']:
        assert sha(RUN/f)==provenance[str((RUN/f).relative_to(FINAL))]
    model=Model.load(RUN/'model.json');u=displacement_at_two()
    pipe=set(model.nsets['PIPE_NODES']);assert pipe.issubset(u)
    history=pd.read_csv(RUN/'results/history.csv');row=history[np.isclose(history.time,2)].iloc[-1]
    assert row.stage=='withdraw' and row.pressure_mpa==0 and abs(row.indenter_force_n)<1e-5
    crown=model.nsets['CROWN'];assert len(crown)==1
    measured=-float(u[crown[0]][1]);assert abs(measured-row.crown_inward_displacement_mm)<5e-5
    REPORT['native_displacement'].update({'pipe_node_count':len(pipe),'crown_inward_mm':measured,
                                          'history_crown_mm':float(row.crown_inward_displacement_mm)})
    polygons=[];values=[];outer=[]
    for surface in ['OUTER', 'INNER']:
        for element,face in model.surfaces[surface]:
            for triangle,value in face_triangles(model,element,face,u):
                polygons.append(triangle);values.append(value);outer.append(surface=='OUTER')
    norm=Normalize(-2,24);cmap=plt.get_cmap('coolwarm')

    def draw_surface(ax, detail, zoom=1.18):
        span=90 if detail else 480
        lateral=65 if detail else 215
        bottom=150 if detail else -215
        clipped=[];colors=[]
        for poly,value,is_outer in zip(polygons,values,outer):
            if detail and not is_outer:
                continue
            bounds=[(0, -span, False), (0, span, True)]
            if detail:
                bounds.extend([(1,-lateral,False),(1,lateral,True),(2,bottom,False)])
            for axis,bound,keep_less in bounds:
                if len(poly)<3:break
                poly=clip_polygon(poly,axis,bound,keep_less)
            if len(poly)>=3:
                if detail:
                    assert np.min(poly[:,2]) >= bottom - 1e-8
                clipped.append(poly)
                colors.append(cmap(norm(value)) if is_outer else (.32,.37,.43,1))
        ax.add_collection3d(Poly3DCollection(clipped,facecolors=colors,
                           edgecolors=(.12,.18,.22,.09),linewidths=.10,rasterized=True),autolim=False)
        top=205 if detail else 220
        ax.set(xlim=(-span,span),ylim=(-lateral,lateral),zlim=(bottom,top))
        ax.set_box_aspect((2*span,2*lateral,top-bottom),zoom=zoom)
        ax.view_init(elev=35 if detail else 32,azim=-75 if detail else -52)
        ax.set_axis_off()
        return clipped, colors

    for name,detail in [('dented_pipe_residual',False),('dented_pipe_detail',True),
                        ('dented_pipe_with_inset',False)]:
        composite=name=='dented_pipe_with_inset'
        fig=plt.figure(figsize=(14.4,8.1) if composite else (12.8,7.2))
        ax=fig.add_axes([.005,.13,.81,.70] if composite else [.015,.13,.85,.70],projection='3d')
        draw_surface(ax,detail,zoom=1.13 if composite else 1.18)
        if composite:
            # A second camera views the same native surface, with no displacement amplification.
            frame=(.545,.475,.365,.35)
            fig.add_artist(FancyBboxPatch((frame[0]+.006,frame[1]-.008),frame[2],frame[3],
                           boxstyle='round,pad=.007',transform=fig.transFigure,
                           facecolor='#dbe1e7',edgecolor='none',zorder=3))
            fig.add_artist(FancyBboxPatch(frame[:2],frame[2],frame[3],
                           boxstyle='round,pad=.007',transform=fig.transFigure,
                           facecolor='white',edgecolor='#30485a',linewidth=1.4,zorder=4))
            # Project the native triangles into a 2D viewport so the close-up fills
            # the wide inset without the square viewport imposed by Axes3D.
            camera=fig.add_axes([0,0,.1,.1],projection='3d')
            detail_polys,detail_colors=draw_surface(camera,True)
            matrix=camera.get_proj()
            projected=[]
            for poly,color in zip(detail_polys,detail_colors):
                px,py,pz=proj3d.proj_transform(poly[:,0],poly[:,1],poly[:,2],matrix)
                projected.append((float(np.mean(pz)),np.column_stack([px,py]),color))
            camera.remove()
            projected.sort(key=lambda item:item[0],reverse=True)
            inset=fig.add_axes([.558,.515,.34,.265],zorder=5)
            inset.add_collection(PolyCollection([r[1] for r in projected],
                facecolors=[r[2] for r in projected],edgecolors=(.12,.18,.22,.09),
                linewidths=.10,rasterized=True))
            inset.autoscale_view()
            inset.margins(.015)
            inset.set_aspect('equal')
            inset.set_axis_off()
            fig.text(.562,.798,'DENT CLOSE-UP',fontsize=12,weight='bold',color='#30485a',zorder=6)
            fig.text(.562,.482,'180 × 130 mm window · deformation scale ×1',fontsize=10,color='#30485a',zorder=6)
            fig.canvas.draw()
            crown_xyz=(np.array(model.nodes[crown[0]])+u[crown[0]])[[2,0,1]]
            xp,yp,_=proj3d.proj_transform(*crown_xyz,ax.get_proj())
            cx,cy=fig.transFigure.inverted().transform(ax.transData.transform((xp,yp)))
            fig.add_artist(Ellipse((cx,cy),.075,.105,transform=fig.transFigure,
                           fill=False,edgecolor='#30485a',linewidth=1.4,zorder=6))
            fig.add_artist(ConnectionPatch(xyA=(cx+.037,cy+.025),coordsA=fig.transFigure,
                           xyB=(frame[0],frame[1]+.12),coordsB=fig.transFigure,
                           color='#30485a',linewidth=1.4,zorder=6))
        cax=fig.add_axes([.935 if composite else .89,.25,.015 if composite else .018,.46])
        cb=fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax)
        cb.set_label('Inward displacement −Uy (mm)',fontsize=11)
        fig.text(.045 if composite else .06,.95,'Residual dent after indenter withdrawal',fontsize=21,weight='bold')
        fig.text(.045 if composite else .06,.895,('Dent close-up' if detail else 'Local pipe segment')+
                 '  |  Actual X52 P1 model geometry + native displacements  |  Scale ×1',fontsize=12)
        fig.text(.045 if composite else .06,.10,f'Crown inward displacement: {measured:.3f} mm relative to the initially gouged surface. Pressure: 0 MPa.',fontsize=11)
        fig.text(.045 if composite else .06,.055,'Axial gouge, 5 mm nominal depth; assumed smooth profile. Tool withdrawn. No rupture simulated.',fontsize=10)
        fig.text(.045 if composite else .06,.015,'Local views clipped for display. Shared colour scale: displacement, not stress or damage; dark surface: inner wall.',fontsize=10,color='#555555')
        save(fig,name)
    REPORT['native_displacement']['detail_window_mm']={'axial_z':[-90,90],'lateral_x':[-65,65]}
    REPORT['native_displacement']['composite']='Same native displacement frame and colour scale in main view and inset; camera magnification only, deformation scale one.'
    REPORT['native_displacement']['surface_interpolation']='Quadratic face edge nodes plus serendipity centre, subdivided into triangles and clipped to display planes; outer face colours averaged from nodal displacement; inner wall neutral grey.'


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    plot_criteria()
    render_pipe()
    record(Path(__file__))
    (OUT/'orientation_visuals_provenance.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print('Saved orientation pressure plots and actual residual-dent views to',OUT)
