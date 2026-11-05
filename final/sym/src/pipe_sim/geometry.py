"""Structured Gmsh solid mesh mapped to a full cylindrical pipe.

Gmsh meshes a parametric block (arc distance, wall coordinate, axial distance).
Mapping that mesh to the pipe preserves wall layers even for oblique gouges.
The cylindrical seam is merged. The groove changes the outer radius only.
An independent OpenCASCADE sphere supplies a meshed rigid indenter.
"""
import math
import numpy as np
import gmsh
from .model import Model, HEX_LOCAL, HEX_FACES, TET_FACES


def symmetric_grid(half_length, fine_extent, fine_size, far_size):
    extent = min(half_length, max(fine_size, fine_extent))
    positive = list(np.linspace(0., extent, max(2, math.ceil(extent / fine_size)) + 1))
    step = fine_size
    while positive[-1] < half_length - 1e-9:
        step = min(step * 1.35, far_size)
        remainder = half_length - positive[-1]
        if remainder < 1.5 * step:
            positive.extend(np.linspace(positive[-1], half_length, max(1, math.ceil(remainder / step)) + 1)[1:])
        else:
            positive.append(positive[-1] + step)
    return np.array([-v for v in positive[:0:-1]] + positive)


def gouge_depth(s, z, gouge):
    s, z = np.asarray(s), np.asarray(z)
    if not gouge.enabled: return np.zeros(np.broadcast(s, z).shape)
    angle = math.radians(gouge.angle_deg)
    along = z * math.cos(angle) + s * math.sin(angle)
    across = -z * math.sin(angle) + s * math.cos(angle)
    # Smooth at both edges, finite root curvature, tapered along full length.
    u, v = 2 * along / gouge.length, 2 * across / gouge.width
    inside = (abs(u) < 1) & (abs(v) < 1)
    return np.where(inside, gouge.depth * np.cos(np.pi*u/2)**2 * np.cos(np.pi*v/2)**2, 0.)


def adaptive_block(case):
    """Extruded all-quad parameter mesh, locally refined along the defect.

    Identical transfinite seam edges merge exactly after cylindrical mapping.
    Four base patches retain an exact crown/defect-centre vertex.
    """
    p,g,d,m=case.pipe,case.gouge,case.dent,case.mesh
    geo=gmsh.model.geo;r=p.od/2;ss=[-math.pi*r,0,math.pi*r];zz=[-p.length/2,0,p.length/2]
    pts={(i,j):geo.addPoint(s,0,z) for i,s in enumerate(ss) for j,z in enumerate(zz)}
    horizontal={(i,j):geo.addLine(pts[i,j],pts[i+1,j]) for i in range(2) for j in range(3)}
    vertical={(i,j):geo.addLine(pts[i,j],pts[i,j+1]) for i in range(3) for j in range(2)}
    surfaces=[]
    for i in range(2):
        for j in range(2):
            loop=geo.addCurveLoop([horizontal[i,j],vertical[i+1,j],-horizontal[i,j+1],-vertical[i,j]])
            surface=geo.addPlaneSurface([loop]);geo.mesh.setRecombine(2,surface);surfaces.append((2,surface))
    for i in [0,2]:
        for j in range(2):geo.mesh.setTransfiniteCurve(vertical[i,j],2*math.ceil(p.length/(4*m.far_size))+1)
    geo.extrude(surfaces,0,p.thickness,0,numElements=[m.wall_layers],recombine=True)
    geo.synchronize()
    angle=math.radians(g.angle_deg);si,co=math.sin(angle),math.cos(angle)
    distances=[]
    if g.enabled:
        along=f'({si:.16g}*x+{co:.16g}*z)';across=f'({co:.16g}*x-{si:.16g}*z)'
        distances.append(f'Sqrt(Max(Abs({along})-{g.length/2},0)^2+Max(Abs({across})-{g.width/2},0)^2)')
    if d.enabled:distances.append(f'Max(Sqrt(x*x+z*z)-{d.indenter_diameter/2},0)')
    distance=distances[0] if len(distances)==1 else ('Min('+','.join(distances)+')' if distances else '1000000')
    field=gmsh.model.mesh.field.add('MathEval')
    gmsh.model.mesh.field.setString(field,'F',f'Min({m.far_size},{m.fine_size}+0.3*Max({distance}-{m.refinement_padding},0))')
    gmsh.model.mesh.field.setAsBackgroundMesh(field)
    for name in ['Mesh.MeshSizeFromPoints','Mesh.MeshSizeFromCurvature','Mesh.MeshSizeExtendFromBoundary']:gmsh.option.setNumber(name,0)
    gmsh.option.setNumber('Mesh.Algorithm',6)
    gmsh.option.setNumber('Mesh.RecombinationAlgorithm',3)
    gmsh.model.mesh.generate(3)


def build_model(case):
    case.validate()
    p, g, d, m = case.pipe, case.gouge, case.dent, case.mesh
    radius, inner = p.od/2, p.od/2-p.thickness
    angle = math.radians(g.angle_deg)
    contact_extent = d.indenter_diameter/2 + m.refinement_padding if d.enabled else m.fine_size*2
    sx = max(contact_extent, (g.length*math.sin(angle)+g.width*math.cos(angle))/2 + m.refinement_padding if g.enabled else 0)
    zx = max(contact_extent, (g.length*math.cos(angle)+g.width*math.sin(angle))/2 + m.refinement_padding if g.enabled else 0)
    axes = [symmetric_grid(math.pi*radius, sx, m.fine_size, m.far_size),
            np.linspace(0.,1.,m.wall_layers+1),
            symmetric_grid(p.length/2, zx, m.fine_size, m.far_size)]
    count = math.prod(len(a)-1 for a in axes)
    if m.strategy=='structured' and count > m.max_elements:
        raise ValueError(f"Estimated pipe mesh {count:,} elements exceeds max_elements={m.max_elements:,}")
    model = Model()
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.option.setNumber('General.NumThreads', case.analysis.threads)
    try:
        gmsh.model.add('parametric_pipe')
        if m.strategy=='adaptive_quads':adaptive_block(case)
        else:
            vol = gmsh.model.occ.addBox(0,0,0,1,1,1)
            gmsh.model.occ.synchronize()
            for dim, tag in gmsh.model.getEntities(1):
                bb = gmsh.model.getBoundingBox(dim,tag)
                axis = int(np.argmax(np.array(bb[3:])-bb[:3]))
                gmsh.model.mesh.setTransfiniteCurve(tag, len(axes[axis]))
            for _, tag in gmsh.model.getEntities(2):
                gmsh.model.mesh.setTransfiniteSurface(tag)
                gmsh.model.mesh.setRecombine(2,tag)
            gmsh.model.mesh.setTransfiniteVolume(vol)
            gmsh.model.mesh.generate(3)
        if m.order == 2:
            gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)
            gmsh.model.mesh.setOrder(2)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        xyz = np.asarray(xyz).reshape(-1,3)
        param = (np.column_stack((xyz[:,0],xyz[:,1]/p.thickness,xyz[:,2])) if m.strategy=='adaptive_quads' else
                 np.column_stack([np.interp(xyz[:,i],np.linspace(0,1,len(ax)),ax) for i,ax in enumerate(axes)]))
        s, q, z = param.T
        r = inner + q*(p.thickness-gouge_depth(s,z,g))
        physical = np.column_stack((r*np.sin(s/radius),r*np.cos(s/radius),z))
        index, old_to_new, parameters = {}, {}, {}
        for old, coord, par in zip(tags, physical, param):
            key = tuple(np.round(coord,8))
            if key not in index:
                new = len(index)+1; index[key]=new
                model.nodes[new] = coord.tolist()
                parameters[new] = par.tolist()
            old_to_new[int(old)] = index[key]
        types, etags, conn = gmsh.model.mesh.getElements(3)
        if len(types) != 1: raise ValueError('Expected one structured hex element type')
        typ = types[0]
        _, _, _, nn, local, _ = gmsh.model.mesh.getElementProperties(typ)
        local = np.asarray(local).reshape(nn,3)
        desired = HEX_LOCAL[:nn]
        permutation = [int(np.argmin(np.linalg.norm(local-c,axis=1))) for c in desired]
        kind = 'C3D20R' if m.order == 2 else 'C3D8'
        connectivity = np.asarray(conn[0]).reshape(-1,nn)[:,permutation]
        count=len(connectivity)
        if count>m.max_elements:raise ValueError('Actual mesh exceeds max_elements')
        for e, nodes in enumerate(connectivity,1):
            ns = [old_to_new[int(n)] for n in nodes]
            model.elements[e] = (kind,ns)
        model.elsets['PIPE'] = list(model.elements)
        model.nsets['PIPE_NODES'] = sorted(model.nodes)
        model.nsets['OUTER_NODES'] = [n for n,par in parameters.items() if abs(par[1]-1)<1e-8]
        model.nsets['INNER_NODES'] = [n for n,par in parameters.items() if abs(par[1])<1e-8]
        model.nsets['END_MINUS'] = [n for n,x in model.nodes.items() if abs(x[2]+p.length/2)<1e-7]
        model.nsets['END_PLUS'] = [n for n,x in model.nodes.items() if abs(x[2]-p.length/2)<1e-7]
        outer = model.nsets['OUTER_NODES']
        model.nsets['BOTTOM'] = [n for n in outer if abs(model.nodes[n][0])<1e-7 and model.nodes[n][1]<0]
        if case.analysis.support=='bottom_patch':
            half=case.analysis.support_length/2
            model.nsets['BOTTOM_SUPPORT']=[n for n in model.nsets['BOTTOM'] if abs(model.nodes[n][2])<=half+1e-8]
            if len(model.nsets['BOTTOM_SUPPORT'])<3:
                raise ValueError('Support length needs at least three bottom mesh nodes; refine axial mesh')
        def closest(target, candidates=outer):
            return min(candidates,key=lambda n:np.linalg.norm(np.array(model.nodes[n])-target))
        model.nsets['AXIAL_ANCHOR'] = [closest([0,-radius,0])]
        model.nsets['LATERAL_ANCHORS'] = [closest([0,-radius,-p.length/2]),closest([0,-radius,p.length/2]),closest([0,radius,-p.length/2])]
        model.nsets['CROWN'] = [closest([0,radius-g.depth if g.enabled else radius,0])]
        model.nsets['GAUGES'] = list(dict.fromkeys(closest([radius*math.sin(s0/radius),radius*math.cos(s0/radius),z0])
            for s0,z0 in case.probe_positions))
        model.nsets['HISTORY'] = sorted(set(model.nsets['GAUGES']+model.nsets['CROWN']))
        for name in ['INNER','OUTER','END_MINUS','END_PLUS','PIPE_CONTACT']:
            model.surfaces[name] = []
        model.elsets['DEFECT'] = []
        for e, (_,ns) in model.elements.items():
            pars = np.array([parameters[n] for n in ns[:8]])
            # Face q / z are continuous even where s crosses the merged seam.
            for face, ii in HEX_FACES.items():
                qp, zp = pars[ii,1], pars[ii,2]
                if np.all(abs(qp)<1e-8): model.surfaces['INNER'].append((e,face))
                if np.all(abs(qp-1)<1e-8):
                    model.surfaces['OUTER'].append((e,face))
                    center = np.mean([model.nodes[ns[i]] for i in ii],axis=0)
                    if center[1]>0 and abs(center[0])<contact_extent*1.5 and abs(center[2])<contact_extent*1.5:
                        model.surfaces['PIPE_CONTACT'].append((e,face))
                if np.all(abs(zp+p.length/2)<1e-8): model.surfaces['END_MINUS'].append((e,face))
                if np.all(abs(zp-p.length/2)<1e-8): model.surfaces['END_PLUS'].append((e,face))
            center = np.mean([model.nodes[n] for n in ns[:8]],axis=0)
            if center[1]>0 and abs(center[0])<sx*1.1 and abs(center[2])<zx*1.1:
                model.elsets['DEFECT'].append(e)
        # Include elements adjacent to measurement nodes even outside the defect patch.
        gauge_nodes=set(model.nsets['GAUGES'])
        model.elsets['DEFECT']=sorted(set(model.elsets['DEFECT'])|{
            e for e,(_,ns) in model.elements.items() if gauge_nodes.intersection(ns)})
        # Inspect mapped element Jacobians with Gmsh, not just the regular block.
        gmsh.model.add('mapped_pipe')
        entity = gmsh.model.addDiscreteEntity(3)
        ids = sorted(model.nodes)
        gmsh.model.mesh.addNodes(3,entity,ids,np.array([model.nodes[n] for n in ids]).ravel())
        inverse = np.argsort(permutation)
        mapped_conn = [np.array(ns)[inverse] for _,ns in model.elements.values()]
        gmsh.model.mesh.addElementsByType(entity,int(typ),list(model.elements),np.asarray(mapped_conn).ravel())
        quality = gmsh.model.mesh.getElementQualities(list(model.elements), 'minSJ')
        min_quality = float(np.min(quality))
        if min_quality <= 0: raise ValueError(f'Inverted pipe elements: minimum scaled Jacobian {min_quality:g}')
        model.metadata.update({'pipe_elements':count,'pipe_nodes':len(model.nodes),'min_scaled_jacobian':min_quality,
            'gmsh_version':gmsh.__version__,'mesh_method':m.strategy+' parameter mesh mapped onto cylindrical wall; merged seam',
            'gouge_profile':'depth*cos(pi*u/length)^2*cos(pi*v/width)^2 within footprint',
            'coordinates':'x lateral, y indenter/crown, z axial; origin at pipe centre'})
        if d.enabled:
            gmsh.option.setNumber('Mesh.MeshSizeFromPoints',1)
            gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',1)
            gmsh.model.add('sphere')
            sphere = gmsh.model.occ.addSphere(0,radius+d.indenter_diameter/2+d.clearance,0,d.indenter_diameter/2)
            gmsh.model.occ.synchronize()
            gmsh.option.setNumber('Mesh.MeshSizeMin',m.indenter_size)
            gmsh.option.setNumber('Mesh.MeshSizeMax',m.indenter_size)
            gmsh.option.setNumber('Mesh.ElementOrder',1)
            gmsh.model.mesh.generate(3)
            tags,coords,_ = gmsh.model.mesh.getNodes()
            offset = max(model.nodes)
            mapping = {int(n):i+offset+1 for i,n in enumerate(tags)}
            for n,x in zip(tags,np.asarray(coords).reshape(-1,3)): model.nodes[mapping[int(n)]]=x.tolist()
            types, es, cs = gmsh.model.mesh.getElements(3)
            model.elsets['INDENTER']=[]
            for t,ec,co in zip(types,es,cs):
                if t != 4: raise ValueError('Expected linear tetrahedral rigid indenter')
                for ns in np.asarray(co).reshape(-1,4):
                    e = len(model.elements)+1
                    model.elements[e]=('C3D4',[mapping[int(n)] for n in ns]); model.elsets['INDENTER'].append(e)
            model.nsets['INDENTER_NODES']=sorted(mapping.values())
            faces={}
            for e in model.elsets['INDENTER']:
                ns=model.elements[e][1]
                for f,ii in TET_FACES.items(): faces.setdefault(tuple(sorted(ns[i] for i in ii)),[]).append((e,f))
            model.surfaces['INDENTER_SURFACE']=[v[0] for v in faces.values() if len(v)==1]
            ref=max(model.nodes)+1; rot=ref+1
            model.nodes[ref]=[0,radius+d.indenter_diameter/2+d.clearance,0]
            model.nodes[rot]=[0,0,0]
            model.nsets['INDENTER_REF']=[ref];model.nsets['INDENTER_ROT']=[rot]
            model.nsets['HISTORY'].append(ref)
        return model
    finally:
        gmsh.finalize()
