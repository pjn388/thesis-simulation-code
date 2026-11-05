"""CalculiX input decks for the shared geometry and physical case definitions."""
from pathlib import Path


def steps(case):
    result = [('indent', -(case.dent.travel+case.dent.clearance)), ('withdraw', 0.)] if case.dent.enabled else []
    return result + [('pressurise', None)]


def write_mesh(model, path):
    with Path(path).open('w') as f:
        f.write('** Shared undeformed mesh. Units: mm, N, MPa.\n*NODE\n')
        for n,x in model.nodes.items(): f.write(f'{n}, '+', '.join(f'{v:.12g}' for v in x)+'\n')
        for kind in ['C3D20R','C3D8','C3D4']:
            rows = [(e,ns) for e,(t,ns) in model.elements.items() if t==kind]
            if not rows: continue
            f.write(f'*ELEMENT, TYPE={kind}\n')
            for e,ns in rows:
                # Maximum 16 entries on an input data line; continue 20-node bricks.
                f.write(', '.join(map(str,[e]+ns[:15])) + (',\n' if len(ns)>15 else '\n'))
                if len(ns)>15: f.write(', '.join(map(str,ns[15:]))+'\n')
        for keyword, groups, attr in [('NSET',model.nsets,'NSET'),('ELSET',model.elsets,'ELSET')]:
            for name, ids in groups.items():
                if not ids: continue
                f.write(f'*{keyword}, {attr}={name}\n')
                for start in range(0,len(ids),16): f.write(', '.join(map(str,ids[start:start+16]))+'\n')
        for name, faces in model.surfaces.items():
            if faces:
                f.write(f'*SURFACE, NAME={name}, TYPE=ELEMENT\n')
                for e,face in faces: f.write(f'{e}, S{face}\n')


def write_deck(case, model, directory, solver='calculix'):
    if solver != 'calculix': raise ValueError('Only CalculiX is supported')
    directory=Path(directory)
    write_mesh(model,directory/'mesh.inp')
    a, mat, d, p=case.analysis,case.material,case.dent,case.pipe
    lines=['*HEADING',f'{case.name} | {case.purpose} | mm N MPa',
           '** Provisional assumptions and validation status are in case.json.', '*INCLUDE, INPUT=mesh.inp',
           '*MATERIAL, NAME=PIPE_STEEL','*ELASTIC',f'{mat.young:.12g}, {mat.poisson:.12g}']
    if mat.plastic:
        lines+=['*PLASTIC']+[f'{s:.12g}, {ep:.12g}' for s,ep in mat.plastic]
    lines+=['*SOLID SECTION, ELSET=PIPE, MATERIAL=PIPE_STEEL']
    if d.enabled:
        lines+=['*SOLID SECTION, ELSET=INDENTER, MATERIAL=PIPE_STEEL']
        ref,rot=model.nsets['INDENTER_REF'][0],model.nsets['INDENTER_ROT'][0]
        lines += [f'*RIGID BODY, NSET=INDENTER_NODES, REF NODE={ref}, ROT NODE={rot}']
        lines+=['*SURFACE INTERACTION, NAME=TOOL_CONTACT']
        if d.normal_law=='compression_only':
            # CCX face-to-face LINEAR springs can retain tension on unloading.
            # TABULAR clamps pressure to zero for every opening. Keep the same
            # positive slope over 0..wall thickness; reaching the far cap is invalid.
            lines+=['*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TABULAR','0., 0.',
                    f'{d.normal_stiffness*p.thickness:.12g}, {p.thickness:.12g}']
        else:
            lines+=['*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR', f'{d.normal_stiffness:.12g}']
        if d.contact_formulation=='node_to_surface' and d.normal_law=='linear':
            # Explicit negligible tensile regularisation; avoid material-dependent defaults.
            lines[-1]+=', 1.e-6, 0.001'
        if d.friction>0:
            lines+=['*FRICTION',f'{d.friction:.12g}, {d.stick_stiffness:.12g}']
        contact_type=d.contact_formulation.replace('_',' ').upper()
        lines+=[f'*CONTACT PAIR, INTERACTION=TOOL_CONTACT, TYPE={contact_type}',
                'PIPE_CONTACT, INDENTER_SURFACE']
    lines+=['*BOUNDARY']
    if a.support in ('bottom_line','bottom_patch'):
        support='BOTTOM_SUPPORT' if a.support=='bottom_patch' else 'BOTTOM'
        lines+=[f'{support}, 2, 2, 0.', 'LATERAL_ANCHORS, 1, 1, 0.']
    else:
        lines+=['END_MINUS, 1, 2, 0.', 'END_PLUS, 1, 2, 0.']
    lines+=['AXIAL_ANCHOR, 3, 3, 0.']
    if d.enabled:
        lines+=['INDENTER_REF, 1, 3, 0.']
        lines+=['INDENTER_ROT, 1, 3, 0.']
    for name, travel in steps(case):
        lines += [f'** Stage: {name}']
        maximum = (a.pressure_max_increment if name == 'pressurise' and
                   a.pressure_max_increment is not None else a.max_increment)
        initial = min(a.initial_increment, maximum)
        lines += [f'*STEP, NLGEOM, INC={a.max_increments}',
                  ('*STATIC, SOLVER='+a.linear_solver.upper() if a.linear_solver!='default' else '*STATIC'),f'{initial:.12g}, 1., {a.min_increment:.12g}, {maximum:.12g}']
        if travel is not None:
            lines+=['*BOUNDARY',f'INDENTER_REF, 2, 2, {travel:.12g}']
        if name=='pressurise':
            if a.pressure_contact_iterations is not None:
                # Blank fields preserve delcon, alea and kscalemax. FIELD
                # acceptance tolerances and earlier steps remain unchanged.
                lines += ['*CONTROLS, PARAMETERS=CONTACT', f',,,{a.pressure_contact_iterations}']
            # Contact is inactive after retraction; avoid artificial re-contact as the pipe expands.
            if d.enabled: lines += ['*MODEL CHANGE, TYPE=CONTACT PAIR, REMOVE','PIPE_CONTACT, INDENTER_SURFACE']
            lines+=['*DLOAD']
            for e,face in model.surfaces['INNER']: lines+=[f'{e}, P{face}, {a.max_pressure:.12g}']
            if a.closed_ends:
                ri=p.od/2-p.thickness; ro=p.od/2
                end_traction=a.max_pressure*ri**2/(ro**2-ri**2)
                for surf in ('END_MINUS','END_PLUS'):
                    for e,face in model.surfaces[surf]: lines += [f'{e}, P{face}, {-end_traction:.12g}']
        freq=a.output_frequency
        lines += [f'*NODE PRINT, NSET=HISTORY, FREQUENCY={freq}', 'U, RF',
                  f'*EL PRINT, ELSET=PIPE, FREQUENCY={freq}', 'PEEQ',
                  f'*EL PRINT, ELSET=DEFECT, FREQUENCY={freq}', 'S, E, COORD',
                  f'*NODE FILE, FREQUENCY={freq}', 'U', '*EL FILE', 'S, PEEQ']
        if d.enabled:lines += [f'*CONTACT FILE, FREQUENCY={freq}','CDIS, CSTR']
        lines += ['*END STEP']
    (directory/'job.inp').write_text('\n'.join(lines)+'\n')
    return directory/'job.inp'
