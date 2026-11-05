"""Read-only rendering of native X52 PEEQ frames, in initial surface coordinates."""
from pathlib import Path
import sys, json, mmap, hashlib, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'final/sym/src'))
from pipe_sim.model import Model, HEX_FACES

RUNS=ROOT/'final/sym/runs/campaign'
CASES=[(f'P{i}'+(' / combined 0°' if i==1 else ''),f'full_zhao2021_p{i}_adaptive_pressure005_20261020') for i in (1,2,3)]
CASES += [(f'Dent + gouge {a}°',f'orientation_combined_{a:02d}_adaptive_pressure005_20261020') for a in (15,30,45,60)]
CASES += [('Gouge only 60°','orientation_gouge_60_adaptive')]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def frame(path,target):
    before=path.stat()
    with path.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as m:
        pos=0
        while True:
            pos=m.find(b'equivalent plastic strain',pos)
            if pos<0:raise ValueError(f'Missing PEEQ frame {target} in {path}')
            end=m.find(b'\n',pos)
            header=m[pos:end]
            t=float(header.split(b'time')[-1])
            if abs(t-target)<1e-7:
                start=end+1
                while m[start:start+1] in (b'\n',b'\r'):start+=1
                end=m.find(b'\n\n',start)
                block=m[start:end]
                data=np.loadtxt(block.splitlines())
                digest=hashlib.sha256(block).hexdigest()
                break
            pos=end
    after=path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    return data,dict(time=t,sha256=digest,hash_scope='selected native PEEQ rows',source_bytes=before.st_size,source_mtime_ns=before.st_mtime_ns)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
norm=Normalize(0,.9);cmap=plt.get_cmap('inferno')
records=[]
pdf=PdfPages(OUT/'all-failure-locations.pdf')
for title,cid in CASES:
    print('Reading '+cid,flush=True)
    run=RUNS/cid
    h=pd.read_csv(run/'results/history.csv')
    row=h[(h.stage=='pressurise') & (h.max_peeq>=.8)].iloc[0]
    summary=json.loads((run/'results/summary.json').read_text())
    case=json.loads((run/'case.json').read_text());model=Model.load(run/'model.json')
    cache=OUT/(cid+'.npz')
    if cache.exists():
        d=np.load(cache);data=d['data'];audit=json.loads(str(d['audit']))
    else:
        data,audit=frame(run/'job.dat',float(row.time))
        np.savez_compressed(cache,data=data,audit=json.dumps(audit))
    maximum=data[np.argmax(data[:,2])]
    assert abs(maximum[2]-row.max_peeq)<1e-6
    assert int(maximum[0])==summary['failure_element']==int(row.max_peeq_element)
    assert int(maximum[1])==summary['failure_integration_point']==int(row.max_peeq_ip)
    assert len(data)==len(model.elsets['PIPE'])*8
    assert len(set(map(tuple,data[:,:2].astype(int))))==len(data)
    values={}
    for e,ip,p in data:values[int(e)]=max(values.get(int(e),0),float(p))
    radius=case['pipe']['od']/2
    polygons=[];colors=[]
    # Project every element through the wall onto initial unrolled coordinates.
    # Sorting by PEEQ puts the largest integration-point values on top.
    for e in sorted(values,key=values.get):
        xyz=np.array([model.nodes[n] for n in model.elements[e][1][:8]])
        if xyz[:,1].mean()<=0:continue
        # Each extrusion cell's first four corners bound its surface footprint.
        q=xyz[:4]
        poly=np.column_stack((q[:,2],radius*np.arctan2(q[:,0],q[:,1])))
        if np.max(np.abs(poly))>500:continue
        polygons.append(poly);colors.append(values[e])
    e=int(maximum[0]);center=np.mean([model.nodes[n] for n in model.elements[e][1][:8]],axis=0)
    hx,hy=float(center[2]),float(radius*np.arctan2(center[0],center[1]))
    g=case['gouge'];a=np.deg2rad(g['angle_deg'])
    uv=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]])*[g['length']/2,g['width']/2]
    outline=uv@np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
    fig,axs=plt.subplots(1,2,figsize=(12,6.5))
    fig.subplots_adjust(top=.78,bottom=.20,left=.07,right=.88,wspace=.30)
    for i,ax in enumerate(axs):
        ax.add_collection(PolyCollection(polygons,array=np.array(colors),cmap=cmap,norm=norm,edgecolors='none'))
        ax.plot(outline[:,0],outline[:,1],color='#40cfff',lw=1.0,ls='--',label='Gouge boundary')
        ax.plot(0,0,'+',color='#32dbff',ms=11,mew=1.5,label='Dent / gouge centre')
        ax.plot(hx,hy,'*',color='#00eea1',mec='black',mew=.6,ms=15,label='First crossing element')
        span=max(g['length']/2+25,90) if i==0 else 90
        ax.set(xlim=(-span,span),ylim=(-span,span),aspect='equal',xlabel='Along pipe axis z (mm)',ylabel='Around pipe, s (mm)')
        ax.set_title('Full gouge footprint' if i==0 else 'Dent-region close-up')
        ax.set_facecolor(cmap(0))
    fig.suptitle(title+' — first sampled failure-criterion crossing',x=.07,ha='left',fontsize=16,weight='bold',y=.97)
    lo,hi=summary['pressure_bracket_mpa']
    fig.text(.07,.88,f'Pressure {hi:.3f} MPa  |  bracket {lo:.3f}–{hi:.3f} MPa  |  maximum PEEQ {maximum[2]:.4f}',fontsize=11)
    cbax=fig.add_axes([.91,.28,.018,.42]);fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cbax,label='Maximum PEEQ projected through wall',extend='max')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower left',bbox_to_anchor=(.06,.085),ncol=3,frameon=False,fontsize=9)
    fig.text(.07,.07,f'Hotspot element {e}, integration point {int(maximum[1])}; element-centre location z = {hx:+.1f}, s = {hy:+.1f} mm.',fontsize=9)
    fig.text(.07,.025,'Initial coordinates, pipe surface unrolled. Colour uses raw integration-point maxima; no nodal averaging.\nNumerical PEEQ ≥ 0.8 location, not a simulated crack or a deformed-shape image.',fontsize=9,color='#444444')
    stem='failure-'+cid
    fig.savefig(OUT/(stem+'.png'),dpi=165)
    pdf.savefig(fig);plt.close(fig)
    records.append(dict(title=title,case_id=cid,image=stem+'.png',pressure_bracket_mpa=[lo,hi],max_peeq=float(maximum[2]),failure_element=e,failure_ip=int(maximum[1]),initial_element_centroid_xyz_mm=center.tolist(),initial_element_centroid_z_s_mm=[hx,hy],native_frame=audit,sources={str(p.relative_to(ROOT)):sha(p) for p in [run/'model.json',run/'case.json',run/'results/history.csv',run/'results/summary.json']}))
    print('Verified and rendered '+title,flush=True)
pdf.close()
(OUT/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
shutil.copy2(ROOT/'final/presentation/assets/dented_pipe_with_inset.png',OUT/'P1-post-withdrawal.png')
# A simple document wrapper for scientific plots; no application UI.
parts=['<!doctype html><html><body><h2>X52 failure locations</h2><p>Eight distinct configurations reached PEEQ 0.8. P1 also supplies combined 0°. These maps show the first crossing frame on the original, unrolled pipe surface. Green stars identify the crossing elements; bright areas have larger plastic strain.</p><p>P1–P3 and combined 15°–60° use the pressure-refined runs. Gouge-only 60° uses its retained parent run. P4 and combined 75°/90° did not reach the criterion.</p>']
for r in records:parts.append(f'<img src="{OUT/r["image"]}" width="100%" alt="{r["title"]} failure location">')
parts+=['<h2>Post-dent shape: X52 P1</h2><p>Actual calculated shape after complete tool withdrawal, before pressure, at deformation scale ×1. Crown inward displacement: 21.813 mm.</p>',f'<img src="{OUT/"P1-post-withdrawal.png"}" width="100%" alt="P1 residual dent">','<p>The complete X52 post-dent set is in <a href="postdent.html">postdent.html</a> and <a href="all-X52-post-dent-shapes.pdf">the post-dent PDF</a>. These show zero-pressure states after withdrawal, not last-pressure shapes.</p></body></html>']
(OUT/'index.html').write_text('\n'.join(parts))
print('Done',flush=True)
