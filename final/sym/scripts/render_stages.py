"""Render actual converged CalculiX stage shapes; no simulation rerun required.

Usage: uv run python scripts/render_stages.py runs/demo_001
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pipe_sim.config import Case
from pipe_sim.model import Model, HEX_FACES, HEX_LOCAL, TET_FACES
from pipe_sim.results import NUMBER
import re


def frames_at(path, times):
    frames = {}; active = False; time = None; candidate = {}
    with path.open() as stream:
        for line in stream:
            if line.lstrip().startswith('100CL'):
                time = float(line.split()[2])
            elif line.startswith(' -4'):
                active = line.split()[1] == 'DISP' and any(abs(time-t)<1e-7 for t in times)
                if active: candidate = {}
            elif active and line.startswith(' -1'):
                values = re.findall(NUMBER, line[3:])
                candidate[int(values[0])] = np.array([float(x) for x in values[1:4]])
            elif line.startswith(' -3') and active:
                frames[round(time, 7)] = candidate; active = False
    for time in times:
        if time not in frames: raise ValueError(f'No complete displacement frame at stage time {time}')
    return frames


def face_patches(model, element, face, displacement):
    """Use quadratic face edge nodes, with a serendipity-interpolated centre."""
    kind, nodes = model.elements[element]
    corners = HEX_FACES[face]
    boundary = []
    for i, j in zip(corners, corners[1:]+corners[:1]):
        boundary.append(nodes[i])
        if kind == 'C3D20R':
            local = (HEX_LOCAL[i]+HEX_LOCAL[j])/2
            k = np.flatnonzero(np.all(HEX_LOCAL == local, axis=1))[0]
            boundary.append(nodes[k])
    xyz = np.array([model.nodes[n] for n in boundary])
    u = np.array([displacement.get(n, np.zeros(3)) for n in boundary])
    if kind == 'C3D20R':
        centre = -.25*xyz[::2].sum(axis=0)+.5*xyz[1::2].sum(axis=0)
        uc = -.25*u[::2].sum(axis=0)+.5*u[1::2].sum(axis=0)
    else: centre, uc = xyz.mean(axis=0), u.mean(axis=0)
    for i in range(len(boundary)):
        j = (i+1)%len(boundary)
        yield np.array([xyz[i]+u[i], xyz[j]+u[j], centre+uc]), float(-(u[i,1]+u[j,1]+uc[1])/3)


def main(directory):
    model = Model.load(directory/'model.json'); case = Case.load(directory/'case.json')
    if not case.dent.enabled: raise ValueError('This stage visualisation expects a dent/withdraw/pressure run')
    frames = {0: {}, **frames_at(directory/'job.frd', [1., 2., 3.])}
    history = pd.read_csv(directory/'results/history.csv')
    out = directory/'results/stages'; out.mkdir(exist_ok=True)
    radius = case.pipe.od/2
    span = max(case.gouge.length*.8, case.dent.indenter_diameter*1.5)
    width = min(radius*.65, span*.7)
    titles = ['1  Initial gouged pipe', '2  Maximum indentation', '3  After withdrawal / spring-back',
              f'4  Pressurised to {case.analysis.max_pressure:g} MPa']
    subtitles = ['Material removed before denting', 'Tool at its lowest position',
                'Tool retracted; permanent dent remains', 'Final converged pressure step; no rupture simulated']
    # Use pipe-only displacement limits: rigid-tool movement is not pipe deformation.
    pipe_nodes = {n for e in model.elsets['PIPE'] for n in model.elements[e][1]}
    norm = Normalize(0, max(-u[n][1] for u in frames.values() for n in pipe_nodes if n in u))
    cmap = plt.get_cmap('YlOrRd')

    def draw(ax, stage):
        # Matplotlib sorts whole collections by depth; the pipe otherwise hides
        # half the sphere incorrectly. Draw the transparent tool over the pipe.
        ax.computed_zorder = False
        u = frames[stage]; polygons = []; values = []
        for e, face in model.surfaces['OUTER']:
            ns = [model.elements[e][1][i] for i in HEX_FACES[face]]
            centre = np.mean([model.nodes[n] for n in ns], axis=0)
            if centre[1] <= 0 or abs(centre[0]) > width or abs(centre[2]) > span: continue
            for xyz, value in face_patches(model, e, face, u):
                polygons.append(xyz[:, [2, 0, 1]]); values.append(value)
        ax.add_collection3d(Poly3DCollection(polygons, facecolors=cmap(norm(values)),
                                             edgecolors='#655b4c', linewidths=.13, zorder=2))
        tool = []
        for e, face in model.surfaces['INDENTER_SURFACE']:
            ns = [model.elements[e][1][i] for i in TET_FACES[face]]
            xyz = np.array([np.array(model.nodes[n])+u.get(n, np.zeros(3)) for n in ns])
            tool.append(xyz[:, [2, 0, 1]])
        # Transparent tool shows the actual sphere position without hiding the dent.
        ax.add_collection3d(Poly3DCollection(tool, facecolors='#7098b3', edgecolors='#47617a',
                                             linewidths=.12, alpha=.10, zorder=3))
        ax.set(xlim=(-span,span), ylim=(-width,width),
               zlim=(radius-15, radius+case.dent.indenter_diameter+2),
               xlabel='Axial z (mm)', ylabel='Lateral x (mm)', zlabel='Height y (mm)')
        ax.set_box_aspect((2*span,2*width,case.dent.indenter_diameter+17))
        ax.view_init(elev=28,azim=-63)
        ax.set_title(titles[stage]+'\n'+subtitles[stage], loc='left', fontsize=11, pad=10)
        if stage:
            row = history[np.isclose(history.time, stage)].iloc[-1]
            label = f'Crown inward displacement: {row.crown_inward_displacement_mm:.3f} mm'
            if stage == 1: label += f'  |  Force: {row.indenter_force_n/1000:.2f} kN'
        else: label = f'Initial gouge depth: {case.gouge.depth:g} mm  |  Orientation: {case.gouge.angle_deg:g}°'
        ax.text2D(.02,.01,label,transform=ax.transAxes,fontsize=9)
        ax.tick_params(labelsize=8)

    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig = plt.figure(figsize=(15,11),layout='constrained')
    axes = [fig.add_subplot(2,2,i+1,projection='3d') for i in range(4)]
    for i, ax in enumerate(axes): draw(ax,i)
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes,shrink=.65,pad=.02,
                 label='Pipe inward displacement −Uy (mm); outward values clipped to zero')
    fig.suptitle(f'{case.gouge.angle_deg:g}° gouge: denting → spring-back → pressurisation\nActual calculated shapes, displacement scale ×1; local pipe surface shown',fontsize=16)
    fig.savefig(out/'stage_overview.png',dpi=170)
    fig.savefig(out/'stage_overview.pdf'); plt.close(fig)
    for stage in range(4):
        fig = plt.figure(figsize=(9,6),layout='constrained'); ax=fig.add_subplot(111,projection='3d')
        draw(ax,stage)
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.65,label='Pipe inward displacement (mm)')
        fig.savefig(out/f'stage_{stage+1:02d}.png',dpi=180); plt.close(fig)

    # Exact mesh-node centreline profiles make millimetre-scale spring-back visible.
    outer = set()
    for e, face in model.surfaces['OUTER']:
        corners = HEX_FACES[face]; ns = model.elements[e][1]
        for i in range(len(ns)):
            if i in corners or (i>=8 and any(np.all(HEX_LOCAL[i] == (HEX_LOCAL[a]+HEX_LOCAL[b])/2)
                                  for a,b in zip(corners,corners[1:]+corners[:1]))): outer.add(ns[i])
    profile = sorted([n for n in outer if abs(model.nodes[n][0])<1e-6 and model.nodes[n][1]>0
                      and abs(model.nodes[n][2])<=span],key=lambda n:model.nodes[n][2])
    original = np.array([model.nodes[n] for n in profile])
    fig, axs = plt.subplots(2,2,figsize=(13,7),sharex=True,sharey=True,layout='constrained')
    for stage, ax in enumerate(axs.flat):
        xyz = np.array([np.array(model.nodes[n])+frames[stage].get(n,np.zeros(3)) for n in profile])
        ax.axhline(radius,color='#a0a0a0',ls=':',label='Original cylinder before gouging')
        ax.plot(original[:,2],original[:,1],color='#777',ls='--',lw=1.5,label='Initially gouged surface')
        ax.plot(xyz[:,2],xyz[:,1],color='#bd3425',lw=2.5,label='Surface at this stage')
        ax.fill_between(xyz[:,2],radius-8,xyz[:,1],color='#cbd5df',alpha=.45)
        if stage == 1:
            sphere_y = radius+case.dent.indenter_diameter/2+case.dent.clearance
            tool_nodes=model.nsets['INDENTER_NODES']
            sphere_y+=float(np.mean([frames[stage][n][1] for n in tool_nodes]))
            rr=case.dent.indenter_diameter/2; z=np.linspace(-rr,rr,200)
            ax.plot(z,sphere_y-np.sqrt(np.maximum(0,rr*rr-z*z)),color='#47617a',lw=2,label='Nominal spherical indenter profile')
        ax.set(title=titles[stage],xlim=(-span,span),ylim=(radius-4,radius+.5),
               xlabel='Axial z (mm)',ylabel='Surface height y (mm)')
        ax.grid(alpha=.2)
    handles,labels=axs.flat[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Longitudinal profile through the crown (x = 0)\nVertical axis expanded for readability; dimensions are actual millimetres',fontsize=15)
    fig.savefig(out/'stage_profiles.png',dpi=170);fig.savefig(out/'stage_profiles.pdf');plt.close(fig)
    print(out.resolve())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    main(parser.parse_args().directory)
