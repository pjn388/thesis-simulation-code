"""Controlled mismatch investigation with the same pipe solver for every case."""
import argparse
import copy
import json
import os
from pathlib import Path
import pandas as pd
from pipe_sim.config import Case
from pipe_sim.materials import total_curve_to_plastic
from pipe_sim.runner import generate,run,write_json
from pipe_sim.results import extract
from prepare_cases import FINAL,RESULTS,digest

PLAN=RESULTS/'config/investigation.json'
CASE_DIR=FINAL/'sym/cases/investigation'
RUN_DIR=FINAL/'sym/runs/investigation'


def prepare():
    if PLAN.exists():raise FileExistsError('Investigation already prepared; preserve its case snapshots')
    CASE_DIR.mkdir(parents=True,exist_ok=True)
    entries=json.loads((RESULTS/'config/literature_cases.json').read_text())
    source=pd.read_csv(FINAL/'data/figures/allouti2012/fig09_material_curve.csv').sort_values('point_order')
    # Published vector path contains occasional same-x vertices: coalesce by mean
    # and enforce neither stress smoothing nor a fitted indentation response.
    curve=source.groupby('x',as_index=False).y.mean().sort_values('x')
    material=total_curve_to_plastic(curve.y,curve.x,202500.)
    supported=float(material[-1,1]);initial=float(material[0,0])
    plastic=material.tolist()+[[float(material[-1,0]),.8]]
    jobs=[]
    def add(name,base_id,changes,description,curve_material=False):
        entry=copy.deepcopy(next(e for e in entries if e['id']==base_id))
        case=Case.load(FINAL/entry['case_file']);case.name=name;original=case.to_dict()
        for group,values in changes.items():
            for key,value in values.items():setattr(getattr(case,group),key,value)
        if curve_material:
            case.material.plastic=plastic
            case.material.source='Allouti 2012 Figure 9, assumed true total strain; elastic intersection, no table-yield splice.'
            entry['material'].update(initial_yield_anchor_mpa=initial,
                processing='Curve-preserving elastic intersection; no table-yield splice.',interpretation_confirmed=False,
                source_supported_plastic_strain_max=supported)
        # Explicit numerical/physical controls kept constant except named factor.
        case.analysis.timeout_seconds=1800
        case.notes += ['Controlled investigation: '+description]
        case.notes += ['The earlier table-yield note describes the archived baseline only. This case follows Figure 9.'] if curve_material else []
        case.validate();path=CASE_DIR/(name+'.json');case.save(path)
        entry.update(id=name,case_file=str(path.relative_to(FINAL)),case_sha256=digest(path),assumptions=case.notes)
        jobs.append({'id':name,'base_id':base_id,'description':description,'case_file':str(path.relative_to(FINAL)),
                     'case_sha256':digest(path),'entry':entry,'run':str((RUN_DIR/name).relative_to(FINAL)),
                     'changes':changes,'curve_material':curve_material})
        if case.dent.normal_law=='compression_only':
            executable=FINAL/'sym/vendor/ccx_tabular_fix/ccx_2.23_tabularfix'
            if not executable.exists():raise FileNotFoundError('Build scripts/build_contact_fix.py and verify the contact coupon first')
            jobs[-1].update(solver_executable=str(executable.relative_to(FINAL)),solver_sha256=digest(executable))
    base='allouti2012_3_percent_od'
    add('a37_curve_only',base,{},'Material only: preserve Figure 9 and use its elastic intersection.',True)
    add('a37_mesh_only',base,{'mesh':{'fine_size':4.,'wall_layers':3}},'Numerical mesh only: surface size 6 to 4 mm; wall layers 2 to 3.')
    add('a37_tool_only',base,{'mesh':{'indenter_size':2.}},'Indenter surface resolution only: 4 to 2 mm.')
    add('a37_support_only',base,{'analysis':{'support':'bottom_patch','support_length':88.9}},'Support sensitivity only: central bottom span = one OD; finite support visible in Allouti 2014 Fig.5, actual span unknown.')
    add('a37_curve_8pct','allouti2012_8_percent_od',{},'Transfer check: identical Figure 9 material treatment at the deeper 8%-OD condition, without fitting to its force response.',True)
    add('a37_contact_fixed',base,{'dent':{'normal_law':'compression_only'}},'Shallow dent: Figure 9 + compression-only contact (patched CCX)',True)
    add('a37_8pct_contact_fixed','allouti2012_8_percent_od',{'dent':{'normal_law':'compression_only'}},'Deeper dent: Figure 9 + compression-only contact (patched CCX)',True)
    add('a37_frictionless_contact_check',base,{'dent':{'normal_law':'compression_only','friction':0.}},'Contact sensitivity: Figure 9, patched compression-only law, zero friction',True)
    write_json(PLAN,{'source_curve_sha256':digest(FINAL/entries[0]['material']['source']),
        'elastic_intersection_stress_mpa':initial,'source_supported_plastic_strain_max':supported,'jobs':jobs})
    print('Prepared',len(jobs),'controlled cases; curve intersection stress',initial,flush=True)


def execute(name):
    plan=json.loads(PLAN.read_text());job=next(j for j in plan['jobs'] if j['id']==name)
    path=FINAL/job['case_file']
    if digest(path)!=job['case_sha256']:raise ValueError('Investigation input changed')
    directory=generate(Case.load(path),FINAL/job['run'])
    print('Generated',directory,flush=True)
    if job.get('solver_executable'):
        executable=FINAL/job['solver_executable']
        if digest(executable)!=job['solver_sha256']:raise ValueError('Investigation solver changed')
        os.environ['CCX_COMMAND']=str(executable)
    status=run(directory);print(json.dumps(status),flush=True)
    if (directory/'job.dat').exists() and (directory/'job.dat').stat().st_size:
        print(json.dumps(extract(directory)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run']);parser.add_argument('name',nargs='?')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:execute(args.name)
