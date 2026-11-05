"""Build literature cases from curated CSVs without changing the source data."""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
from pipe_sim.config import Case
from pipe_sim.materials import total_curve_to_plastic

FINAL = Path(__file__).resolve().parents[2]
DATA = FINAL/'data'
RESULTS = FINAL/'results'
CASES = FINAL/'sym/cases/literature'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(relative):
    return pd.read_csv(DATA/relative)


def save_case(case,target):
    """Regeneration must not alter the identity of an already executed case."""
    records=RESULTS/'config/runs.json'
    used={r['case_id'] for r in json.loads(records.read_text())} if records.exists() else set()
    if target.exists() and case.name in used:
        if Case.load(target).to_dict()!=case.to_dict():
            raise ValueError(f'Refusing to replace executed case {case.name}; create a new named variant.')
        return
    case.save(target)


def material(paper, young, yield_stress, policy='legacy_table_anchor'):
    relative=f'figures/{paper}/'+('fig09_material_curve.csv' if paper=='allouti2012' else 'fig11_material_curve.csv')
    source=read(relative).sort_values('point_order')
    # Allouti labels only stress/strain; true total strain is an explicit scenario.
    # Zhao explicitly labels true stress/true strain. Never apply a second true conversion.
    ep=source.x.to_numpy()-source.y.to_numpy()/young
    stress=source.y.to_numpy()
    if policy=='figure_curve':
        curve=source.groupby('x',as_index=False).y.mean().sort_values('x')
        table=total_curve_to_plastic(curve.y,curve.x,young).tolist()
        processing='Preserve Figure 9 after its elastic intersection; no tabulated-yield splice. Table/figure discrepancy remains unresolved.'
    elif policy=='legacy_table_anchor':
        table=[[float(yield_stress),0.]]
        for s,e in zip(stress,ep):
            if e>table[-1][1] and s>=table[-1][0]:table.append([float(s),float(e)])
        processing='LEGACY: yield anchored to material table; lower-stress, decreasing-stress and non-increasing-plastic-strain samples omitted. Do not treat this as the digitized curve.'
    else:raise ValueError('Unknown material import policy')
    supported=table[-1][1]
    # Explicit constant tail keeps existing solver input validation satisfied.
    # It is NOT recovered paper data or permission to claim a burst prediction.
    if supported<.8:table.append([table[-1][0],.8])
    audit={'source':str((DATA/relative).relative_to(FINAL)), 'source_sha256':digest(DATA/relative),
           'interpretation':'true total strain; plastic strain = total - stress/E',
           'interpretation_confirmed':paper in ('zhao2021','shuai2018'),
           'initial_yield_anchor_mpa':table[0][0], 'source_supported_plastic_strain_max':supported,
           'processing':processing,
           'constant_tail_to_plastic_strain':.8,
           'burst_criterion_calibrated':False}
    return table,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allouti-material',choices=['figure_curve','legacy_table_anchor'],default='figure_curve')
    args=parser.parse_args()
    CASES.mkdir(parents=True,exist_ok=True);(RESULTS/'config').mkdir(exist_ok=True)
    entries=[];materials={}
    base=Case.load(FINAL/'sym/cases/demo_oblique_stick_check.json')
    base.dent.normal_law='compression_only'
    for paper in ['allouti2012','allouti2014','zhao2021']:
        mp='allouti2012' if paper.startswith('allouti') else paper
        row=read('tables/allouti2012/table02_material.csv' if mp=='allouti2012' else 'tables/zhao2021/table01_material.csv').iloc[0]
        young=float(row.youngs_modulus_MPa) if mp=='allouti2012' else 208000.
        policy=args.allouti_material if mp=='allouti2012' else 'legacy_table_anchor'
        table,audit=material(mp,young,float(row.yield_stress_MPa),policy);materials[paper]=audit
        curve_file={'allouti2012':'fig10_load_displacement','allouti2014':'fig11_load_displacement','zhao2021':'fig13_load_displacement'}[paper]
        curves=read(f'figures/{paper}/{curve_file}.csv').fillna({'panel':''})
        exps=curves[curves.series.str.contains('experimental')]
        for (panel,series),points in exps.groupby(['panel','series'],sort=False):
            case=Case.from_dict(base.to_dict())
            if mp=='allouti2012' and policy=='legacy_table_anchor':case.dent.normal_law='linear'
            specimen=panel if paper=='zhao2021' else series.replace('_experimental','').replace('dent_','')
            case.name=f'{paper}_{specimen}'.lower()
            if mp=='allouti2012' and policy=='figure_curve':case.name+='_figure_curve'
            case.gouge.angle_deg=0
            case.material.young=young;case.material.poisson=float(row.poissons_ratio)
            case.material.name='A37' if mp=='allouti2012' else 'API_5L_X52'
            case.material.plastic=table
            case.material.source=json.dumps(audit,sort_keys=True)
            case.analysis.max_pressure=5.;case.analysis.timeout_seconds=1200
            case.mesh.fine_size=6.;case.mesh.far_size=30.;case.mesh.wall_layers=2;case.mesh.indenter_size=4.
            notes=['Preliminary literature comparison; mesh/contact convergence and fixture verification outstanding.',
                   '5 MPa pressure ramp exercises the same three-stage workflow; it is not a burst validation.',
                   'Bottom-line bilateral support and friction 0.3 are modelling assumptions requiring verification.',
                   'Material constant tail is an explicit extrapolation; data-supported strain extent is audited.']
            if case.dent.normal_law=='compression_only':
                notes+=['Compression-only contact uses the isolated CCX 2.23 TABULAR patch; see sym/docs/contact-fix.md. Validated on the elementary coupon; each pipe response still needs checks.']
            if mp=='allouti2012':
                geometry=read('tables/allouti2012/text_geometry_and_intact_test.csv').iloc[0]
                case.pipe.od=float(geometry.OD_mm);case.pipe.thickness=float(geometry.wall_thickness_mm)
                case.pipe.length=float(geometry.length_mm);case.dent.indenter_diameter=float(geometry.indenter_diameter_mm)
                case.dent.travel=float(points.x.max())
                notes+=['Allouti Figure 9 stress/strain treated as true stress/total strain, which the figure does not explicitly confirm.',
                        ('Figure 9 preserved from its approximately 249 MPa elastic intersection; Table 2 yield of 355 MPa conflicts with that curve.'
                         if policy=='figure_curve' else 'LEGACY: 355 MPa initial yield splice discards lower-stress digitized points; retained for diagnostic comparison only.'),
                        'Maximum imposed travel taken from experimental loading-loop maximum; this is a prescribed boundary condition, not a fitted prediction.',
                        'Figure percent-OD labels describe residual dents; they are not substituted for loaded tool travel.',
                        'PEEQ 0.8 is only a software placeholder here; Allouti uses different failure criteria. No Allouti burst error is computed.']
                case.gouge.enabled=paper=='allouti2014'
                if case.gouge.enabled:
                    defects=read('tables/allouti2014/table03_burst_and_fracture.csv')
                    vessel={'4_percent_OD':'V1','10_percent_OD':'V2','16_percent_OD':'V3'}[specimen]
                    defect=defects[defects.vessel==vessel].iloc[0]
                    case.gouge.length=float(defect.gouge_length_2a_mm);case.gouge.width=float(defect.gouge_width_2c_mm)
                    case.gouge.depth=float(defect.gouge_depth_mm);case.mesh.fine_size=1.;case.mesh.wall_layers=3
                    notes+=['A37 Figure 9 is borrowed from Allouti 2012; transfer to 2014 is an assumption.',
                            'Cosine gouge does not reproduce the published root radius; local strain/burst validation is not established.',
                            'Experimental dent-then-EDM sequence differs from the model gouge-then-dent sequence, as discussed by Allouti.',
                            'V1 table dent depth 9.3 mm conflicts with 4% OD; neither is silently adopted as the loaded travel.']
                else:vessel=None
                sources=['tables/allouti2012/text_geometry_and_intact_test.csv','tables/allouti2012/table02_material.csv']
                if paper=='allouti2014':sources+=['tables/allouti2014/table03_burst_and_fracture.csv']
            else:
                params=read('tables/zhao2021/text_model_parameters.csv').set_index('parameter').value
                defect=read('tables/zhao2021/table03_defects.csv').set_index('specimen').loc[specimen]
                case.pipe.od=float(params.OD);case.pipe.thickness=float(params.wall_thickness);case.pipe.length=float(params.pipe_length)
                case.dent.indenter_diameter=float(params.indenter_diameter);case.dent.travel=float(defect.dent_depth_mm)
                case.gouge.enabled=True;case.gouge.length=float(defect.gouge_length_mm);case.gouge.depth=float(defect.gouge_depth_mm)
                case.gouge.width=20.;case.mesh.fine_size=5.;case.mesh.far_size=100.;case.mesh.wall_layers=4
                case.mesh.indenter_size=6.25;case.analysis.timeout_seconds=7200
                notes+=['20 mm cosine-gouge width is provisional; original width/cross-section not resolved.',
                        'E=208 GPa follows constitutive-model text; Table 1 gives 209 GPa.',
                        'Digitized Figure 11 is used instead of silently choosing an inconsistent Ramberg–Osgood convention.',
                        'Table dent depth interpreted as nominal loaded travel; this measurement definition still needs confirmation.']
                sources=['tables/zhao2021/text_model_parameters.csv','tables/zhao2021/table03_defects.csv','tables/zhao2021/table01_material.csv']
                vessel=specimen
            case.notes=notes;case.purpose='preliminary_literature_comparison';case.validate()
            target=CASES/(case.name+'.json');save_case(case,target)
            sources += [f'figures/{paper}/{curve_file}.csv']
            entries.append({'id':case.name,'paper':paper,'specimen':specimen,'vessel':vessel,
                'case_file':str(target.relative_to(FINAL)), 'case_sha256':digest(target),
                'reference_file':f'data/figures/{paper}/{curve_file}.csv','panel':panel,
                'experimental_series':series,'numerical_series':series.replace('experimental','numerical'),
                'material':audit,'assumptions':notes,'validation_status':'preliminary_unvalidated',
                'sources':{str((DATA/s).relative_to(FINAL)):digest(DATA/s) for s in sources}})
    # Shuai supplies gauge-axis coordinates and a distinct loaded/residual dent.
    paper='shuai2018';row=read('tables/shuai2018/table01_material_geometry.csv').iloc[0]
    case=Case.from_dict(base.to_dict());case.name='shuai2018_dent'
    case.pipe.od=float(row.OD_mm);case.pipe.thickness=float(row.wall_thickness_mm);case.pipe.length=float(row.length_mm)
    case.gouge.enabled=False;case.dent.indenter_diameter=100.;case.dent.travel=67.
    case.material.young=float(row.youngs_modulus_GPa)*1000;case.material.poisson=float(row.poissons_ratio)
    table,audit=material(paper,case.material.young,float(row.yield_strength_MPa));materials[paper]=audit
    case.material.plastic=table;case.material.name='API_5L_X52';case.material.source=json.dumps(audit,sort_keys=True)
    case.mesh.fine_size=7.8;case.mesh.far_size=150;case.mesh.wall_layers=4;case.mesh.indenter_size=7.8
    case.analysis.max_pressure=10.7;case.analysis.timeout_seconds=7200
    layout=read('tables/shuai2018/fig04_gauge_layout.csv').set_index('gauge')
    gauges=[]
    for panel,gauge in [('a',1),('b',2),('c',10),('d',4),('e',12)]:
        r=layout.loc[gauge];gauges.append({'panel':panel,'gauge':gauge,
              'arc_mm':float(r.circumferential_offset_mm),'axial_mm':float(r.axial_offset_mm)})
    case.probe_positions=[[g['arc_mm'],g['axial_mm']] for g in gauges]
    case.notes=['Preliminary pressure-strain comparison, not validated.',
        '100 mm indenter and 67 mm loaded travel from Section 2.1; 48.68 mm is the measured residual dent, not commanded travel.',
        'True-stress/strain interpretation follows Section 3.2; digitized stresses below table yield are omitted.',
        'Gauge strains use nearby integration points and engineering stretch relative to full withdrawal; not surface gauge-length averages.',
        'Figure 15 panel f (point 18) excluded: coordinates not established in the extracted layout.',
        'Bottom-line support, simplified end traction, contact stiffness and full-pipe mesh require sensitivity checks.',
        'Material constant tail is explicit extrapolation; PEEQ 0.8 is not a calibrated Shuai failure criterion.']
    case.notes+=['Compression-only contact uses the isolated CCX 2.23 TABULAR patch; this geometry has not yet been validated.']
    case.purpose='preliminary_literature_comparison';case.validate();target=CASES/(case.name+'.json');save_case(case,target)
    sources=['tables/shuai2018/table01_material_geometry.csv','tables/shuai2018/fig04_gauge_layout.csv',
             'figures/shuai2018/fig15_strain_validation.csv','evidence/shuai2018/text.txt']
    entries.append({'id':case.name,'paper':paper,'specimen':'dented_pipe','vessel':None,'comparison_kind':'pressure_strain',
        'case_file':str(target.relative_to(FINAL)),'case_sha256':digest(target),
        'reference_file':'data/figures/shuai2018/fig15_strain_validation.csv','gauges':gauges,
        'material':audit,'assumptions':case.notes,'validation_status':'preliminary_unvalidated',
        'sources':{str((DATA/s).relative_to(FINAL)):digest(DATA/s) for s in sources}})
    registry=RESULTS/'config/literature_cases.json'
    if registry.exists():
        # Preserve earlier material interpretations and user-created scenarios.
        entries += [e for e in json.loads(registry.read_text()) if e['id'] not in {x['id'] for x in entries}]
    registry.write_text(json.dumps(entries,indent=2)+'\n')
    (RESULTS/'config/material_audit.json').write_text(json.dumps(materials,indent=2)+'\n')
    print(f'Prepared {len(entries)} cases in {CASES}')


if __name__=='__main__':main()
