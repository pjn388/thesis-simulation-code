"""Headless model previews, independent of a desktop CAD/FE viewer."""
import numpy as np
from .model import HEX_FACES


def render_model(case,model,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .geometry import gouge_depth
    fig=plt.figure(figsize=(12,5),layout='constrained')
    ax=fig.add_subplot(121,projection='3d')
    polygons=[];colours=[]
    for e,face in model.surfaces['OUTER']:
        xyz=np.array([model.nodes[model.elements[e][1][i]] for i in HEX_FACES[face]])
        polygons.append(xyz[:,[2,0,1]])
        colours.append(max(0,case.pipe.od/2-np.linalg.norm(xyz[:,:2],axis=1).mean()))
    values=np.asarray(colours)/max(case.gouge.depth,1e-12)
    coll=Poly3DCollection(polygons,facecolors=plt.cm.viridis(values),edgecolors='#333333',linewidths=.12)
    ax.add_collection3d(coll)
    r=case.pipe.od/2;L=case.pipe.length
    ax.set(xlim=(-L/2,L/2),ylim=(-r,r),zlim=(-r,r),xlabel='Axial z (mm)',ylabel='x (mm)',zlabel='y (mm)')
    ax.set_box_aspect((min(L/(2*r),5),1,1));ax.view_init(elev=35,azim=-65)
    ax.set_title(f'{len(model.elsets["PIPE"]):,} pipe elements; min SJ={model.metadata["min_scaled_jacobian"]:.3f}')
    ax=fig.add_subplot(122)
    extent=max(case.gouge.length/2,case.dent.indenter_diameter)*1.15
    z=np.linspace(-extent,extent,250);s=np.linspace(-extent,extent,250)
    zz,ss=np.meshgrid(z,s);depth=gouge_depth(ss,zz,case.gouge)
    im=ax.pcolormesh(zz,ss,depth,shading='auto',cmap='viridis')
    ax.set(xlabel='Axial z (mm)',ylabel='Circumferential arc s (mm)',title=f'Initial material removal; angle {case.gouge.angle_deg:g}°',aspect='equal')
    fig.colorbar(im,ax=ax,label='Gouge depth (mm)')
    fig.suptitle(case.name+' — provisional geometry')
    fig.savefig(path,dpi=160);plt.close(fig)


def render_deformation(case,model,displacements,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(11,5),layout='constrained')
    for index,deformed in enumerate([False,True],1):
        ax=fig.add_subplot(1,2,index,projection='3d');polygons=[];values=[]
        for e,face in model.surfaces['OUTER']:
            ns=[model.elements[e][1][i] for i in HEX_FACES[face]]
            xyz=np.array([model.nodes[n] for n in ns]);u=np.array([displacements.get(n,[0,0,0]) for n in ns])
            if deformed:xyz+=u
            polygons.append(xyz[:,[2,0,1]]);values.append(float(np.linalg.norm(u,axis=1).mean()))
        values=np.array(values);colours=plt.cm.viridis(values/max(values.max(),1e-12))
        ax.add_collection3d(Poly3DCollection(polygons,facecolors=colours,edgecolors='#333333',linewidths=.12))
        r=case.pipe.od/2;length=case.pipe.length
        ax.set(xlim=(-length/2,length/2),ylim=(-r*1.2,r*1.2),zlim=(-r*1.2,r*1.2),xlabel='z (mm)',ylabel='x (mm)',zlabel='y (mm)')
        ax.set_box_aspect((min(length/(2*r),5),1,1));ax.view_init(elev=35,azim=-65)
        ax.set_title('Last converged shape, scale 1' if deformed else 'Initial shape')
    fig.suptitle(case.name+' — displacement magnitude colours; development result')
    fig.savefig(path,dpi=150);plt.close(fig)
