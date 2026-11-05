"""Verify archived FRDs and render native X52 indent/withdrawal frames read-only."""
from pathlib import Path
import hashlib, importlib.util, json, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1];FINAL=ROOT/'final'
sys.path.insert(0,str(FINAL/'sym/src'))
from pipe_sim.model import Model
spec=importlib.util.spec_from_file_location('existing_renderer',FINAL/'presentation/make_orientation_visuals.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
NUM=re.compile(rb'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?')
CASES=[(f'X52 P{i}'+(' / combined 0°' if i==1 else ''),f'full_zhao2021_p{i}_adaptive') for i in (1,2,3,4)]
CASES += [(f'X52 dent + gouge {a}°',f'orientation_combined_{a:02d}_adaptive') for a in (15,30,45,60,75,90)]
CASES += [('X52 dent only','orientation_dent_00_adaptive')]
entries=json.loads((OUT/'archived-shape-files.json').read_text())
index={x['entry']['relative_path']:x for x in entries}

def read_frames(path,expected):
    frames={};blocks={};digest=hashlib.sha256();before=path.stat()
    def block(b):
        if not b.startswith(b'  100CL'):return
        lines=b.split(b'\n',2)
        t=float(lines[0].split()[2])
        if not any(abs(t-k)<1e-7 for k in (1.,2.)) or not lines[1].startswith(b' -4  DISP'):return
        end=b.find(b'\n -3')
        assert end>0
        frame={}
        for line in b[:end].splitlines():
            if line.startswith(b' -1'):
                v=NUM.findall(line[3:]);frame[int(v[0])]=np.array([float(x) for x in v[1:4]])
        count=int(lines[0].split()[3]);assert len(frame)==count
        frames[round(t)]=frame;blocks[round(t)]=hashlib.sha256(b[:end+4]).hexdigest()
    buffer=b'';total=0
    with path.open('rb') as f:
        while chunk:=f.read(8*1024*1024):
            digest.update(chunk);total+=len(chunk);buffer+=chunk
            while True:
                k=buffer.find(b'\n  100CL',1)
                if k<0:break
                block(buffer[:k]);buffer=buffer[k+1:]
        block(buffer)
    after=path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    assert total==expected['source_identity']['size']
    assert digest.hexdigest()==expected['sha256'],f'Archive SHA mismatch: {path}'
    assert set(frames)=={1,2}
    return frames,dict(archive_path=str(path),archive_sha256=digest.hexdigest(),archive_bytes=total,selected_block_sha256=blocks)

def polygons(model,u):
    polys=[];vals=[]
    for e,face in model.surfaces['OUTER']:
        xyz=np.array([model.nodes[n] for n in model.elements[e][1]])
        c=xyz.mean(axis=0)
        if c[1]<100 or abs(c[0])>145 or abs(c[2])>190:continue
        for triangle,value in helper.face_triangles(model,e,face,u):
            poly=triangle
            for axis,bound,less in [(0,-150,False),(0,150,True),(1,-105,False),(1,105,True)]:
                if len(poly)<3:break
                poly=helper.clip_polygon(poly,axis,bound,less)
            if len(poly)>=3:polys.append(poly);vals.append(value)
    return polys,vals

norm=Normalize(0,50);cmap=plt.get_cmap('viridis')
def draw(ax,pv):
    p,v=pv
    ax.add_collection3d(Poly3DCollection(p,facecolors=cmap(norm(v)),edgecolors=(.05,.1,.12,.10),linewidths=.12,rasterized=True),autolim=False)
    ax.set(xlim=(-150,150),ylim=(-105,105),zlim=(125,205),xlabel='Axial z (mm)',ylabel='Lateral x (mm)',zlabel='Height y (mm)')
    ax.set_box_aspect((300,210,80),zoom=.93)
    ax.view_init(elev=24,azim=-66)
    ax.tick_params(labelsize=8,pad=0)

plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','pdf.fonttype':42})
records=[];overview=[]
with PdfPages(OUT/'all-X52-post-dent-shapes.pdf') as pdf:
    for title,cid in CASES:
        print('Verifying archive and extracting '+cid,flush=True)
        run=FINAL/'sym/runs/campaign'/cid
        rel=str((run/'job.frd').relative_to(FINAL));entry=index[rel]
        path=Path(entry['archive_directory'])/'final'/rel
        cache=OUT/(cid+'-dent-frames.npz');auditfile=OUT/(cid+'-dent-frame-audit.json')
        if cache.exists() and auditfile.exists():
            d=np.load(cache);audit=json.loads(auditfile.read_text());frames={t:{int(n):u for n,u in zip(d['ids'+str(t)],d['u'+str(t)])} for t in (1,2)}
            assert audit['archive_sha256']==entry['entry']['sha256']
        else:
            frames,audit=read_frames(path,entry['entry'])
            arrays={}
            for t,frame in frames.items():arrays['ids'+str(t)]=np.array(list(frame));arrays['u'+str(t)]=np.array(list(frame.values()))
            np.savez_compressed(cache,**arrays);auditfile.write_text(json.dumps(audit,indent=2)+'\n')
        model=Model.load(run/'model.json');case=json.loads((run/'case.json').read_text())
        h=pd.read_csv(run/'results/history.csv');summary=json.loads((run/'results/summary.json').read_text())
        assert summary['withdrawal_completed'] and summary['residual_dent_reliable']
        pipe=set(model.nsets['PIPE_NODES']);crown=model.nsets['CROWN'];assert len(crown)==1
        measured={};pv={}
        for t in (1,2):
            assert pipe.issubset(frames[t])
            row=h[np.isclose(h.time,t,atol=1e-7,rtol=0)].iloc[-1]
            assert row.stage==('indent' if t==1 else 'withdraw') and row.pressure_mpa==0
            measured[t]=-float(frames[t][crown[0]][1])
            assert abs(measured[t]-row.crown_inward_displacement_mm)<1e-3
            if t==2:assert abs(row.indenter_force_n)<1e-4
            pv[t]=polygons(model,frames[t])
        fig=plt.figure(figsize=(13,6.4))
        axes=[fig.add_axes([.00,.20,.46,.59],projection='3d'),fig.add_axes([.46,.20,.46,.59],projection='3d')]
        for ax,t in zip(axes,(1,2)):
            draw(ax,pv[t]);ax.set_title(('At maximum indentation' if t==1 else 'After tool withdrawal')+f'\nCrown displacement: {measured[t]:.3f} mm',fontsize=12,pad=5)
        fig.suptitle(title,x=.05,ha='left',y=.97,fontsize=19,weight='bold')
        fig.text(.05,.89,'Native calculated shapes • zero internal pressure • deformation scale ×1',fontsize=12)
        cb=fig.add_axes([.935,.30,.013,.40]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Inward displacement −Uy (mm)',extend='both')
        g=case['gouge'];desc=(f"Gouge: {g['length']:g} × {g['width']:g} mm; depth {g['depth']:g} mm; angle {g['angle_deg']:g}° from axis." if g['enabled'] else 'No gouge; dent-only control.')
        fig.text(.05,.15,desc,fontsize=10)
        fig.text(.05,.10,f"Crown spring-back: {measured[1]-measured[2]:.3f} mm. Tool hidden for visibility; same camera and colour scale in all cases.",fontsize=10)
        fig.text(.05,.04,'Local 300 × 210 mm surface window. Displacement is relative to the initially gouged mesh; gouge depth is not added.\nActual quadratic mesh geometry and native displacements; no simulation rerun, smoothing or deformation amplification.',fontsize=9,color='#444')
        name='postdent-'+cid+'.png';fig.savefig(OUT/name,dpi=170);pdf.savefig(fig);plt.close(fig)
        overview.append((title,pv[2],measured[2]))
        records.append(dict(title=title,case_id=cid,image=name,stage_times=[1,2],pressure_mpa=0,displacement_scale=1,crown_inward_mm=measured,archive=audit,sources={str(p.relative_to(ROOT)):helper.sha(p) for p in [run/'model.json',run/'case.json',run/'results/history.csv',run/'results/summary.json']}))
        (OUT/'postdent-provenance.json').write_text(json.dumps(records,indent=2)+'\n')
        print('Rendered '+title,flush=True)
    for name,items in [('P1-P4-post-withdrawal',overview[:4]),('orientation-post-withdrawal',[overview[0]]+overview[4:10]),('dent-only-post-withdrawal',overview[10:])]:
        n=len(items);cols=2 if n>1 else 1;rows=(n+cols-1)//cols
        fig=plt.figure(figsize=(12,6.5 if n==1 else rows*3.6+1.5))
        for i,(title,pv,dent) in enumerate(items):
            ax=fig.add_subplot(rows,cols,i+1,projection='3d');draw(ax,pv);ax.set_axis_off();ax.set_title(title+f' | residual crown {dent:.3f} mm',fontsize=11,y=.93)
        fig.subplots_adjust(top=.81 if n==1 else .88,bottom=.10,left=.01,right=.91,hspace=.04,wspace=.03)
        fig.suptitle('After complete tool withdrawal — pressure 0 MPa',fontsize=17,y=.97)
        fig.text(.04,.89 if n==1 else .925,'Actual shapes at scale ×1; all views share the same dimensions and colour scale.',fontsize=11)
        cb=fig.add_axes([.93,.30,.014,.40]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cb,label='Inward displacement −Uy (mm)',extend='both')
        fig.text(.04,.035,'Local pipe surface. Crown displacement is measured from the initially gouged surface; no rupture is simulated.',fontsize=10)
        fig.savefig(OUT/(name+'.png'),dpi=170);pdf.savefig(fig);plt.close(fig)
parts=['<!doctype html><html><body><h2>X52 post-dent shapes</h2><p>Actual calculated pipe surfaces after full tool withdrawal, before pressurisation. All use deformation scale ×1 and the same colour scale. P1 is also the combined 0° baseline.</p>']
for name in ['P1-P4-post-withdrawal','orientation-post-withdrawal','dent-only-post-withdrawal']:parts.append(f'<img src="{OUT/(name+".png")}" width="100%" alt="{name}">')
parts.append('<h3>Maximum indentation and spring-back, case by case</h3>')
for r in records:parts.append(f'<details><summary>{r["title"]}</summary><img src="{OUT/r["image"]}" width="100%" alt="{r["title"]} indentation and withdrawal"></details>')
parts.append('<p>All 11 source FRDs matched their archived SHA-256 checksums. Complete displacement frames and crown values were checked against retained histories. These are the original C3D20R runs; the failure maps use the pressure-refined variants where available.</p></body></html>')
(OUT/'postdent.html').write_text('\n'.join(parts))
print('All post-dent shapes complete',flush=True)
