#!/usr/bin/env python3
"""Rebuild analytical illustrations and published-reference plots, never solver results."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'evidence'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'axes.labelsize':11, 'pdf.fonttype':42,
                     'savefig.dpi':160, 'axes.titleweight':'bold'})
BLUE='#173d59'; ORANGE='#b95922'

def rows(name):
    with (DATA/name).open(newline='') as stream:
        return list(csv.DictReader(stream))

def save(fig, name):
    fig.tight_layout()
    for suffix in ['pdf','png']:
        fig.savefig(OUT/f'{name}.{suffix}', bbox_inches='tight', metadata={'Creator':'Thesis draft figure generator'})
    plt.close(fig)

angle=np.linspace(0,90,181)
fig,ax=plt.subplots(figsize=(7.4,3.7))
ax.plot(angle,.5*(1+np.cos(np.deg2rad(angle))**2),color=BLUE,lw=2)
ax.set(xlim=(0,90),ylim=(.45,1.05),xticks=np.arange(0,91,15),xlabel='Gouge angle from pipe axis, α (degrees)',ylabel='Resolved normal stress / hoop stress',title='Analytical intact-membrane reference')
ax.grid(alpha=.2)
save(fig,'nominal_orientation')

ref=rows('zhao_burst.csv')
fig,ax=plt.subplots(figsize=(7.4,3.8))
x=np.arange(len(ref)); y=[float(r['burst_pressure_MPa']) for r in ref]
ax.bar(x,y,color=BLUE,width=.58)
for a,b in zip(x,y): ax.text(a,b+.4,f'{b:.2f}',ha='center')
ax.set(xticks=x,xticklabels=[r['specimen'] for r in ref],ylim=(0,27),ylabel='Published experimental burst pressure (MPa)',xlabel='Zhao et al. (2021), Table 4 specimen',title='Literature experiments — no thesis predictions')
save(fig,'zhao_reference')

ref=rows('allouti_burst.csv'); p0=float(rows('allouti_intact.csv')[0]['intact_burst_pressure_MPa'])
fig,ax=plt.subplots(figsize=(7.4,3.7))
ax.scatter([float(r['dent_depth_percent_OD_reported']) for r in ref],[float(r['burst_pressure_MPa']) for r in ref],color=BLUE,s=55,zorder=3,label='Published dented specimens')
ax.axhline(p0,ls='--',color=ORANGE,label=f'Published intact reference: {p0:g} MPa')
ax.set(xlim=(0,31),ylim=(29.8,32.3),xlabel='Reported dent-depth label (% OD)',ylabel='Published burst pressure (MPa)',title='Allouti et al. (2012), Table 4 and intact reference')
ax.legend(loc='upper left',frameon=False); ax.grid(alpha=.2)
save(fig,'allouti_reference')

fig,axes=plt.subplots(1,2,figsize=(8,3.5))
u=np.linspace(-.55,.55,250)
q=np.where(abs(u)<.5,np.cos(np.pi*u)**2,0)
axes[0].plot(u,q,color=BLUE,lw=2)
axes[0].set(xlabel='Along-gouge coordinate u / Lg',ylabel='Removal depth q / a',title='Centreline, v = 0')
xx,yy=np.meshgrid(np.linspace(-.5,.5,161),np.linspace(-.5,.5,161))
zz=np.cos(np.pi*xx)**2*np.cos(np.pi*yy)**2
im=axes[1].contourf(xx,yy,zz,levels=np.linspace(0,1,11),cmap='Blues')
axes[1].set(xlabel='u / Lg',ylabel='v / Wg',title='Unrolled profile (normalised axes)')
fig.colorbar(im,ax=axes[1],label='q / a')
save(fig,'gouge_profile')

# Extra reference illustrations supplied for convenient later use.
ref=rows('zhao_material.csv')
fig,ax=plt.subplots(figsize=(7.4,3.6))
ax.plot([float(r['x']) for r in ref],[float(r['y']) for r in ref],color=BLUE,lw=1.8)
ax.set(xlabel='Published true total strain (dimensionless)',ylabel='Published true stress (MPa)',title='Digitised Zhao et al. (2021), Figure 11')
ax.grid(alpha=.2)
save(fig,'zhao_material_reference')

ref=rows('allouti_load.csv')
fig,ax=plt.subplots(figsize=(7.4,3.9))
for tag,color in [('3',BLUE),('8',ORANGE)]:
    selected=[r for r in ref if r['series']==f'dent_{tag}_percent_OD_experimental']
    selected.sort(key=lambda r:int(r['point_order']))
    if selected:
        ax.plot([float(r['x']) for r in selected],[float(r['y']) for r in selected],'-o',ms=2.5,color=color,label=f'Experiment: {tag}% OD label')
ax.set(xlabel='Published tool travel (mm)',ylabel='Published indenter force (kN)',title='Digitised Allouti et al. (2012), Figure 10')
ax.legend(frameon=False);ax.grid(alpha=.2)
save(fig,'allouti_load_reference')
print('Generated six reference/analytical figures; no simulation data read.')
