"""Publication figures from retained evidence; no solves or changes to source data.

Run with final/sym/.venv/bin/python. Output identities and display-only operations
are recorded separately from the immutable numerical evidence snapshot.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.tri import Triangulation, LinearTriInterpolator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ROOT = FINAL.parent
OUT = HERE / 'figures'
AUDIT = HERE / 'evidence/figure_review_20261031'
SNAP = HERE / 'evidence/results_20261022T080154Z'
SHAPES = ROOT / 'artifacts/x52-failure-locations-20261031'
sys.path[:0] = [str(FINAL/'sym/src'), str(FINAL/'results/scripts')]
from pipe_sim.model import Model, HEX_FACES, HEX_LOCAL
from comparison_utils import canonical_curve

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                     'axes.titlesize':10, 'axes.labelsize':9,
                     'legend.fontsize':8, 'xtick.labelsize':8, 'ytick.labelsize':8,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42, 'savefig.facecolor':'white'})
BLUE, ORANGE = '#176493', '#be561b'
REPORT = {'sources':{}, 'figures':{}, 'checks':[]}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def record(p, expected=None):
    actual = sha(p)
    if expected is not None:
        assert actual == expected, f'Changed retained source: {p}'
    REPORT['sources'][str(p.relative_to(ROOT))] = actual
    return p


def read(p, expected=None):
    return json.loads(record(p, expected).read_text())


def save(fig, name, policy):
    for suffix in ['pdf', 'png']:
        p = OUT / f'{name}.{suffix}'
        kw = {'metadata':{'CreationDate':None, 'ModDate':None}} if suffix=='pdf' else {'dpi':220}
        fig.savefig(p, bbox_inches="tight", pad_inches=.04, **kw)
        REPORT['figures'][str(p.relative_to(ROOT))] = {'sha256':sha(p), 'policy':policy}
    plt.close(fig)
    print(name, flush=True)


def references(entry, ax, identities):
    p = FINAL / entry['reference_file']
    df = pd.read_csv(record(p, identities[entry['reference_file']])).fillna({'panel':''})
    for key, style, colour, label in [('experimental_series','o-','black','Experiment'),
                                    ('numerical_series','--','0.5','Published FE')]:
        chosen = df[(df.panel==entry['panel']) & (df.series==entry[key])]
        curve = canonical_curve(chosen, 'mm', 'kN')
        ax.plot(curve.x, curve.y, style, color=colour, ms=2.5, lw=1, label=label)


def contact_curve(run, identities):
    p = FINAL / run
    h = pd.read_csv(record(p/'results/history.csv', identities[run+'/results/history.csv']))
    case = read(p/'case.json', identities.get(run+'/case.json'))
    c = h[h.stage.isin(['indent','withdraw'])].reset_index(drop=True)
    ends = np.flatnonzero((c.stage=='withdraw') & (np.abs(c.indenter_force_n)<=1e-6))
    assert len(ends)
    end = int(ends[0])
    assert c.indenter_force_n.iloc[end-1] > 1e-6
    assert np.all(np.abs(c.indenter_force_n.iloc[end:])<=1e-6)
    clipped = c.iloc[:end+1].copy()
    assert clipped.indenter_force_n.max() == c.indenter_force_n.max()
    pd.testing.assert_frame_equal(clipped[clipped.stage=='indent'], c[c.stage=='indent'])
    REPORT['checks'].append({'run':run, 'display_points':end+1,
        'omitted_free_withdrawal':len(c)-end-1, 'peak_and_loading_unchanged':True})
    return clipped.indenter_travel_mm-case['dent']['clearance'], clipped.indenter_force_n/1000


def loops():
    frozen = read(SNAP/'provenance.json')['sources']
    jobs = {j['id']:j for j in read(FINAL/'results/config/investigation.json')['jobs']}
    fig, axs = plt.subplots(1,2,figsize=(6.1,2.8),layout='constrained')
    for ax, parent, title in zip(axs, ['a37_contact_fixed','a37_8pct_contact_fixed'], ['(a) Shallow A37 dent','(b) Deeper A37 dent']):
        references(jobs[parent]['entry'], ax, frozen)
        for suffix, colour, style, label in [('',BLUE,'-','CalculiX: 2 layers'),('_wall3_20261018',ORANGE,':','CalculiX: 3 layers')]:
            x,y = contact_curve(jobs[parent+suffix]['run'], frozen)
            ax.plot(x,y,style,color=colour,lw=1.5,label=label)
        ax.set(title=title,xlabel='Tool penetration (mm)',ylabel='Force (kN)',xlim=(0,None),ylim=(0,None))
        ax.grid(alpha=.2)
    axs[0].legend(loc='upper left',fontsize=7)
    save(fig,'a37_contact_comparison','Same four A37 histories as the thesis tables; truncate display after first sampled zero-force withdrawal frame. No alignment, rescaling or metric changes.')
    jobs = {j['id']:j for j in read(FINAL/'results/config/full_campaign.json')['jobs']}
    frozen = read(FINAL/'results/output/contact_limited_20261026/provenance.json')['sources']
    fig,axs = plt.subplots(2,2,figsize=(6.1,4.5),layout='constrained')
    for i,ax in enumerate(axs.flat,1):
        parent=f'full_zhao2021_p{i}_adaptive'
        references(jobs[parent]['entry'], ax, frozen)
        x,y=contact_curve(jobs[parent]['run'], frozen)
        ax.plot(x,y,color=BLUE,lw=1.5,label='CalculiX: C3D20R')
        ax.set(title=f'({chr(96+i)}) X52 P{i}',xlabel='Tool penetration (mm)',ylabel='Force (kN)',xlim=(0,None),ylim=(0,None))
        ax.grid(alpha=.2)
    axs.flat[0].legend(loc='upper left',fontsize=7)
    save(fig,'x52_contact_comparison','Original P1–P4 C3D20R denting histories and Zhao digitised references. First sampled zero-force withdrawal endpoint; no response scaling. Pressure refinement preserves pre-pressure inputs.')


def boundary(model,e,face):
    kind,ns=model.elements[e]
    assert kind=='C3D20R'
    corners=HEX_FACES[face]; ids=[]
    for i,j in zip(corners,corners[1:]+corners[:1]):
        ids.append(ns[i]); mid=(HEX_LOCAL[i]+HEX_LOCAL[j])/2
        ids.append(ns[np.flatnonzero(np.all(HEX_LOCAL==mid,axis=1))[0]])
    return ids


def meshes():
    fig,axs=plt.subplots(1,2,figsize=(6.1,2.8),layout='constrained')
    shape_records={r['case_id']:r for r in read(SHAPES/'postdent-provenance.json')}
    for ax,cid,angle in zip(axs,['full_zhao2021_p1_adaptive','orientation_combined_30_adaptive'],[0,30]):
        r=shape_records[cid];p=FINAL/'sym/runs/campaign'/cid
        for name in ['model.json','case.json']:
            record(p/name,r['sources'][str((p/name).relative_to(ROOT))])
        model=Model.load(p/'model.json');case=json.loads((p/'case.json').read_text())
        edges=[]
        for e,face in model.surfaces['OUTER']:
            q=np.array([model.nodes[n] for n in boundary(model,e,face)])
            if q[:,1].mean()<=0:continue
            poly=np.column_stack((q[:,2],case['pipe']['od']/2*np.arctan2(q[:,0],q[:,1])))
            if np.max(np.abs(poly))>430:continue
            edges.append(np.vstack([poly,poly[0]]))
        ax.add_collection(LineCollection(edges,colors='0.35',lw=.28,rasterized=True))
        ax.set(xlim=(-350,350),ylim=(-240,240),aspect='equal',xlabel='Axial z (mm)',ylabel='Hoop arc s (mm)',title=f'Actual outer mesh: {angle}°')
    save(fig,'actual_surface_mesh','Initial outer-surface C3D20R edges including midside nodes, unrolled into (z,s). Retained production meshes; no mesh regenerated.')


def stages():
    fig,axs=plt.subplots(1,2,figsize=(6.1,2.05),layout='constrained')
    for ax,y,title in zip(axs,[[0,1,0,0],[0,0,0,1]],['(a) Imposed tool travel','(b) Imposed pressure']):
        ax.plot([0,1,2,3],y,color=BLUE,lw=2)
        for t in [1,2]:ax.axvline(t,color='0.7',ls=':',lw=.8)
        ax.set(xticks=[.5,1.5,2.5],xticklabels=['Indent','Withdraw','Pressure'],yticks=[0,1],ylim=(-.06,1.13),title=title,ylabel='Normalised control')
    save(fig,'loading_sequence','Schematic normalised stage controls, not calculated response or physical time. Travel includes initial clearance; internal pressure is zero during both denting stages.')


def sensitivity_and_pressure():
    df=pd.read_csv(record(SNAP/'validation_metrics.csv'))
    fig,axs=plt.subplots(1,2,figsize=(6.1,2.6),layout='constrained')
    for ax,parent,title in zip(axs,['a37_contact_fixed','a37_8pct_contact_fixed'],['(a) Shallow A37 dent','(b) Deeper A37 dent']):
        pair=[df[df.case_id==n].iloc[0] for n in [parent,parent+'_wall3_20261018']]
        vals=[100*(pair[1][k]/pair[0][k]-1) for k in ['peak_force_kn','residual_mm','max_peeq']]
        bars=ax.bar(range(3),vals,color=[BLUE,BLUE,ORANGE],width=.6)
        ax.axhline(0,color='0.4',lw=.6);ax.bar_label(bars,fmt='%+.2f',fontsize=8,padding=2)
        ax.set(xticks=range(3),xticklabels=['Peak\nforce','Residual\ncrown','Maximum\nPEEQ'],ylim=(-.7,8),ylabel='Change (%)',title=title)
    save(fig,'wall_layer_sensitivity','Percentage change from two to three wall layers, computed from the frozen validation table. Separate observables, not discretisation error estimates.')
    df=pd.read_csv(record(SNAP/'pressure005_comparison.csv'))
    df=df[df.refined==True].sort_values('specimen')
    fig,ax=plt.subplots(figsize=(6.1,2.7),layout='constrained')
    for i,r in enumerate(df.itertuples()):
        ax.plot(r.experiment_mpa,i+.18,'o',color='black',ms=5,label='Experiment: burst' if i==0 else None)
        ax.plot(r.paper_fe_mpa,i,'D',color='0.5',ms=4,label='Published FE' if i==0 else None)
        if pd.notna(r.criterion_pressure_mpa):
            ax.plot([r.lower_pressure_mpa,r.criterion_pressure_mpa],[i-.18,i-.18],color=BLUE,lw=2.5,label='CalculiX: PEEQ 0.8 bracket' if i==0 else None)
            ax.plot(r.lower_pressure_mpa,i-.18,'o',mfc='white',mec=BLUE,ms=4)
            ax.plot(r.criterion_pressure_mpa,i-.18,'o',color=BLUE,ms=4)
        else:ax.text(14.8,i-.18,'Criterion not reached',color=BLUE,va='center',fontsize=8)
    ax.set(yticks=range(len(df)),yticklabels=df.specimen,xlabel='Pressure (MPa)',xlim=(12,26),ylim=(-.5,4.2))
    ax.legend(loc='upper left',ncol=2,fontsize=7.5);ax.grid(axis='x',alpha=.2)
    save(fig,'x52_pressure_comparison','Zhao experimental and published FE pressures, and pressure-refined numerical criterion intervals. Open lower and filled upper ends; no assigned value for P4. Intervals are load sampling, not confidence.')


def deformation(orientation=False):
    spec=importlib.util.spec_from_file_location('orientation_renderer',FINAL/'presentation/make_orientation_visuals.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    # The imported module changes plot defaults; restore publication font sizes.
    plt.rcParams.update({'font.size':9,'axes.labelsize':8,'axes.titlesize':10})
    records={r['case_id']:r for r in read(SHAPES/'postdent-provenance.json')}
    cases = [(f'full_zhao2021_p{i}_adaptive',f'P{i}') for i in range(1,5)]
    if orientation:
        cases = [('full_zhao2021_p1_adaptive','0°'),('orientation_combined_45_adaptive','45°'),('orientation_combined_90_adaptive','90°')]
    fig=plt.figure(figsize=(6.1,2.3 if orientation else 4.5)); norm=Normalize(0,50);cmap=plt.get_cmap('viridis')
    for i,(cid,label) in enumerate(cases,1):
        r=records[cid];p=FINAL/'sym/runs/campaign'/cid
        for name in ['model.json','results/history.csv']:
            record(p/name,r['sources'][str((p/name).relative_to(ROOT))])
        data=np.load(record(SHAPES/(cid+'-dent-frames.npz')))
        audit=read(SHAPES/(cid+'-dent-frame-audit.json'))
        assert audit==r['archive']
        u={int(n):v for n,v in zip(data['ids2'],data['u2'])};model=Model.load(p/'model.json')
        assert set(model.nsets['PIPE_NODES']).issubset(u)
        crown=-u[model.nsets['CROWN'][0]][1]
        assert abs(crown-r['crown_inward_mm']['2'])<1e-9
        polys=[];vals=[]
        for e,face in model.surfaces['OUTER']:
            q=np.array([model.nodes[n] for n in model.elements[e][1]])
            if q[:,1].mean()<100 or abs(q[:,0].mean())>145 or abs(q[:,2].mean())>190:continue
            for poly,val in helper.face_triangles(model,e,face,u):
                for axis,bound,less in [(0,-150,False),(0,150,True),(1,-105,False),(1,105,True)]:
                    if len(poly)<3:break
                    poly=helper.clip_polygon(poly,axis,bound,less)
                if len(poly)>=3:polys.append(poly);vals.append(val)
        ax=fig.add_subplot(1,3,i,projection='3d') if orientation else fig.add_subplot(2,2,i,projection='3d')
        ax.add_collection3d(Poly3DCollection(polys,facecolors=cmap(norm(vals)),edgecolors=(.1,.1,.1,.12),linewidths=.1,rasterized=True),autolim=False)
        ax.set(xlim=(-150,150),ylim=(-105,105),zlim=(125,205))
        ax.set_box_aspect((300,210,80));ax.view_init(elev=24,azim=-66);ax.set_axis_off()
        ax.set_title(f'{label}: {crown:.2f} mm' if orientation else f'{label}: residual crown {crown:.2f} mm',fontsize=9,y=.91)
        # Common coordinate triad, in the actual plotting axes (z,x,y).
        for vector,label in [([55,0,0],'z'),([0,45,0],'x'),([0,0,25],'y')]:
            origin=np.array([-135,-90,135]);end=origin+vector
            ax.quiver(*origin,*vector,color='0.25',arrow_length_ratio=.15,lw=.7)
            ax.text(*end,label,fontsize=7)
    fig.subplots_adjust(left=0,right=.88,top=.96,bottom=.03,hspace=0,wspace=0)
    cb=fig.add_axes([.91,.20,.018,.60]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Inward displacement −Uy (mm)')
    save(fig,'orientation_residual_shapes' if orientation else 'x52_residual_shapes','Native t=2 displacement after completed withdrawal, zero pressure, scale x1. Shared 300 x210 mm window/camera/colour limits. Quadratic surface tessellation, no smoothing. Source FRDs independently verified in the retained shape audit.')


def damage_families():
    """Explain the four configurations without pretending to show solved fields."""
    fig,axs=plt.subplots(2,2,figsize=(6.1,2.7),layout='constrained')
    x=np.linspace(-2,2,500)
    dent=.55*np.exp(-(x/1.05)**2)
    gouge=np.where(np.abs(x)<.45,.4*np.cos(np.pi*x/.9)**2,0)
    for ax,(d,g,title) in zip(axs.flat,[(0,0,'Intact'),(1,0,'Dent only'),(0,1,'Gouge only'),(1,1,'Dent + gouge')]):
        outside=-d*dent-g*gouge;inside=-1-d*dent
        ax.fill_between(x,inside,outside,color='#cbdde9')
        ax.plot(x,outside,color=BLUE,lw=1.7);ax.plot(x,inside,color=BLUE,lw=1)
        ax.plot(x,np.zeros_like(x),color='0.6',ls=':',lw=.8)
        ax.set(title=title,xlim=(-2,2),ylim=(-1.7,.25));ax.set_axis_off()
    save(fig,'damage_families','Dimensionless local section schematics: dent changes wall shape, gouge removes outer material, combined includes both. Not calculated displacements, actual specimen profiles or geometrically scaled damage.')


def dent_flank_locations():
    """Locate the actual controlling IP relative to the residual dent and root."""
    records=read(SHAPES/'provenance.json')[:3]
    shape_records={r['case_id']:r for r in read(SHAPES/'postdent-provenance.json')}
    source=record(FINAL/'sym/vendor/CalculiX/ccx_2.23/src/gauss.f').read_text()
    block=source.split('gauss3d2=reshape((/',1)[1].split('/)',1)[0]
    gauss=np.array([float(v.replace('d','e')) for v in re.findall(r'[-+]?\d+\.\d+d[+-]?\d+',block)]).reshape(8,3)
    def shape(q):
        return np.array([np.prod(1+a*q)*(a@q-2)/8 if np.all(a) else
                         (1-q[np.flatnonzero(a==0)[0]]**2)*np.prod(1+a[a!=0]*q[a!=0])/4 for a in HEX_LOCAL])
    assert np.max(np.abs(np.array([shape(q) for q in HEX_LOCAL])-np.eye(20)))<1e-12
    assert np.max(np.abs(np.array([shape(q).sum() for q in gauss])-1))<1e-12
    fig,axs=plt.subplots(2,3,figsize=(6.1,4.65));result=[]
    for col,r in enumerate(records):
        cid=r['case_id'];run=FINAL/'sym/runs/campaign'/cid
        model=Model.load(record(run/'model.json',r['sources'][str((run/'model.json').relative_to(ROOT))]))
        case=read(run/'case.json',r['sources'][str((run/'case.json').relative_to(ROOT))])
        parent=cid.replace('_pressure005_20261020','');pr=shape_records[parent]
        parent_model=FINAL/'sym/runs/campaign'/parent/'model.json'
        pm=Model.load(record(parent_model,pr['sources'][str(parent_model.relative_to(ROOT))]))
        assert pm.nodes==model.nodes and pm.elements==model.elements
        data=np.load(record(SHAPES/(parent+'-dent-frames.npz')))
        assert read(SHAPES/(parent+'-dent-frame-audit.json'))==pr['archive']
        u={int(n):v for n,v in zip(data['ids2'],data['u2'])}
        e=r['failure_element'];natural=gauss[r['failure_ip']-1]
        ns=model.elements[e][1];xyz=shape(natural)@np.array([model.nodes[n] for n in ns])
        outer_face=[f for element,f in model.surfaces['OUTER'] if element==e]
        assert outer_face==[2] and natural[2]>0
        radius=case['pipe']['od']/2;ri=radius-case['pipe']['thickness']
        z=float(xyz[2]);arc=float(radius*np.arctan2(xyz[0],xyz[1]));radial=float(np.hypot(xyz[0],xyz[1]))
        outer_ids=set(n for element,face in model.surfaces['OUTER'] for n in boundary(model,element,face))
        ids=[n for n in outer_ids if model.nodes[n][1]>0 and abs(model.nodes[n][2])<210 and abs(model.nodes[n][0])<60]
        q=np.array([model.nodes[n] for n in ids]);ss=radius*np.arctan2(q[:,0],q[:,1]);zz=q[:,2]
        interp=LinearTriInterpolator(Triangulation(zz,ss),np.array([-u[n][1] for n in ids]))
        x=np.linspace(-150,150,401);profile=interp(x,np.zeros_like(x));assert not np.ma.getmaskarray(profile).any()
        projected=float(interp(z,0));ax=axs[0,col]
        ax.plot(x,profile,color=BLUE,lw=1.7);ax.axvline(0,color='0.65',ls=':',lw=.7)
        ax.axvline(z,color=ORANGE,ls='--',lw=.9);ax.plot(z,projected,'*',color=ORANGE,ms=9)
        ax.set(title=f'P{col+1}',xlim=(-150,150),ylim=(42,-2),xlabel='Axial z (mm)',xticks=[-100,0,100])
        if col==0:ax.set_ylabel('Residual inward\ndisplacement (mm)')
        ax.text(.04,.07,f'z = {z:+.1f} mm',transform=ax.transAxes,fontsize=8)
        g=case['gouge'];assert g['angle_deg']==0
        ss=np.linspace(-16,16,500)
        removal=np.where(np.abs(ss)<g['width']/2,g['depth']*np.cos(np.pi*z/g['length'])**2*np.cos(np.pi*ss/g['width'])**2,0)
        thickness=case['pipe']['thickness']-removal
        ax=axs[1,col];ax.fill_between(ss,0,thickness,color='#cbdde9')
        for fraction in [0,.25,.5,.75,1]:ax.plot(ss,fraction*thickness,color=BLUE if fraction in [0,1] else '0.5',lw=1 if fraction in [0,1] else .55)
        ax.plot(arc,radial-ri,'*',color=ORANGE,ms=9,mec='white',mew=.4)
        ax.set(xlim=(-16,16),ylim=(-.5,10.8),xlabel='Hoop arc s (mm)',xticks=[-10,0,10])
        if col==0:ax.set_ylabel('Height above inner\nsurface (mm)')
        ax.annotate('Gouge root',xy=(0,float(np.interp(0,ss,thickness))),xytext=(-15,9.65),fontsize=8,arrowprops={'arrowstyle':'->','lw':.7,'color':'0.3'})
        result.append({'specimen':f'P{col+1}','case_id':cid,'element':e,'ip':r['failure_ip'],
                       'initial_ip_xyz_mm':xyz.tolist(),'initial_ip_z_s_mm':[z,arc],
                       'outer_wall_element':True,'natural_ip':natural.tolist(),
                       'initial_radial_height_above_inner_mm':radial-ri,
                       'residual_centreline_displacement_at_ip_axial_projection_mm':projected,
                       'pressure_mpa':r['pressure_bracket_mpa'][1]})
    fig.subplots_adjust(left=.12,right=.99,top=.87,bottom=.10,wspace=.32,hspace=.65)
    fig.text(.12,.98,'Residual dent profiles after withdrawal',fontsize=9,va='top')
    fig.text(.12,.49,'Initial wall sections at the controlling locations',fontsize=9,va='top')
    save(fig,'dent_flank_root_locations','P1–P3 actual controlling Gauss-point coordinates reconstructed with C3D20R shape functions and native quadrature ordering. Top: nodal residual centreline displacement interpolated on the actual outer surface, with axial location projected onto that profile. Bottom: analytical initial mapped wall section at that axial coordinate, four layers, and exact reconstructed IP position. Pressure crossing and residual dent are separate stages.')
    REPORT['dent_flank_locations']=result
    REPORT['failure_reference']={'source':'Zhao et al. (2021); original paper obtained separately',
        'reference':'Zhao et al. (2021), Section 2.3.2 and Figure 5 (experimental rupture locations); Section 4.1 and Figures 17–19 (published FE strain and location).',
        'finding':'Paper reports failure at dent flank in the reduced defect region; the present P1–P3 controlling IPs are in outer-layer elements near the gouge centreline on the dent flank. Regional qualitative comparison, not a measured experimental coordinate match.'}


def hotspots():
    records=read(SHAPES/'provenance.json');fig,axs=plt.subplots(2,4,figsize=(6.1,3.65))
    norm=Normalize(0,.9);cmap=plt.get_cmap('inferno')
    for i,(r,ax) in enumerate(zip(records,axs.flat)):
        cid=r['case_id'];run=FINAL/'sym/runs/campaign'/cid
        for name in ['model.json','case.json','results/history.csv','results/summary.json']:
            record(run/name,r['sources'][str((run/name).relative_to(ROOT))])
        model=Model.load(run/'model.json');case=json.loads((run/'case.json').read_text())
        cache=np.load(record(SHAPES/(cid+'.npz')));data=cache['data']
        assert json.loads(str(cache['audit']))==r['native_frame']
        maximum=data[np.argmax(data[:,2])]
        assert int(maximum[0])==r['failure_element'] and int(maximum[1])==r['failure_ip']
        assert abs(maximum[2]-r['max_peeq'])<1e-9
        assert len(data)==8*len(model.elsets['PIPE'])
        assert len(set(map(tuple,data[:,:2].astype(int))))==len(data)
        values={}
        for e,ip,v in data:values[int(e)]=max(values.get(int(e),0),v)
        polys=[];colours=[]
        for e in sorted(values,key=values.get):
            xyz=np.array([model.nodes[n] for n in model.elements[e][1][:8]])
            if xyz[:,1].mean()<=0:continue
            q=xyz[:4];poly=np.column_stack((q[:,2],case['pipe']['od']/2*np.arctan2(q[:,0],q[:,1])))
            if np.max(np.abs(poly))>130:continue
            polys.append(poly);colours.append(values[e])
        ax.add_collection(PolyCollection(polys,array=np.array(colours),cmap=cmap,norm=norm,edgecolors='none',rasterized=True))
        g=case['gouge'];a=np.deg2rad(g['angle_deg']);uv=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*[g['length']/2,g['width']/2]
        outline=uv@np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
        ax.plot(*outline.T,color='#40cfff',lw=.7,ls='--')
        ax.plot(*r['initial_element_centroid_z_s_mm'],'*',color='#00eea1',mec='black',mew=.4,ms=7)
        ax.plot(0,0,'+',color='#40cfff',ms=4)
        title=['P1 / combined 0°','P2','P3','Combined 15°','Combined 30°','Combined 45°','Combined 60°','Gouge only 60°'][i]
        ax.set(title=title+f"\n{r['pressure_bracket_mpa'][1]:.3f} MPa",xlim=(-90,90),ylim=(-90,90),aspect='equal',xticks=[-60,0,60],yticks=[-60,0,60],facecolor=cmap(0))
        ax.title.set_fontsize(8);ax.tick_params(labelsize=7)
        if i>=4:ax.set_xlabel('z (mm)',fontsize=8)
        if i%4==0:ax.set_ylabel('s (mm)',fontsize=8)
        else:ax.set_yticklabels([])
    fig.subplots_adjust(left=.09,right=.88,top=.89,bottom=.13,hspace=.6,wspace=.22)
    cb=fig.add_axes([.92,.20,.018,.6]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Through-wall maximum PEEQ')
    save(fig,'x52_criterion_locations','Eight first sampled PEEQ >=0.8 frames; integration-point maxima projected through the wall in initial unrolled coordinates. Star marks element centroid, not exact IP coordinate or crack. Different pressure in each panel; shared colour scale and crop.')


def main():
    OUT.mkdir(exist_ok=True);AUDIT.mkdir(parents=True,exist_ok=True)
    loops();meshes();stages();sensitivity_and_pressure();deformation();deformation(orientation=True);hotspots();damage_families();dent_flank_locations()
    for relative,expected in REPORT['sources'].items():assert sha(ROOT/relative)==expected
    REPORT['script_sha256']=sha(Path(__file__))
    (AUDIT/'provenance.json').write_text(json.dumps(REPORT,indent=2)+'\n')


if __name__=='__main__':main()
