"""Numerical benchmarks and reference comparisons, separate from solver completion."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import Case
from .runner import write_json


def elastic_benchmark(directory):
    directory=Path(directory);case=Case.load(directory/'case.json')
    if case.dent.enabled or case.gouge.enabled or case.material.plastic or not case.analysis.closed_ends:
        raise ValueError('Benchmark requires an intact, purely elastic, closed-end pipe')
    status=json.loads((directory/'status.json').read_text())
    if status['state']!='completed':raise ValueError('Elastic benchmark run did not complete')
    fields=pd.read_csv(directory/'results/integration_points.csv')
    fields=fields[fields.time==fields.time.max()]
    p=case.analysis.max_pressure;ri=case.pipe.od/2-case.pipe.thickness;ro=case.pipe.od/2
    A=p*ri**2/(ro**2-ri**2);B=p*ri**2*ro**2/(ro**2-ri**2)
    radial=np.hypot(fields.x,fields.y);hoop=A+B/radial**2
    result={'reference':'Lame elastic closed-end cylinder; low strain numerical check',
            'pressure_mpa':p,'axial_stress_theory_mpa':A,
            'axial_stress_mean_mpa':float(fields.axial_stress_mpa.mean()),
            'axial_mean_relative_error':float(abs(fields.axial_stress_mpa.mean()-A)/A),
            'hoop_mean_pointwise_relative_error':float(np.mean(abs(fields.hoop_stress_mpa-hoop)/hoop)),
            'points':len(fields),'is_experimental_validation':False}
    write_json(directory/'results/elastic_benchmark.json',result)
    return result


def compare_reference(directory, specimen, reference_path):
    directory=Path(directory)
    case=Case.load(directory/'case.json')
    table=pd.read_csv(reference_path);match=table[table.specimen==specimen]
    if len(match)!=1:raise ValueError('Expected one matching reference specimen')
    row=match.iloc[0];summary=json.loads((directory/'results/summary.json').read_text())
    target=float(row.burst_pressure_mpa);predicted=summary.get('burst_pressure_mpa')
    result={'case':case.name,'specimen':specimen,'experimental_pressure_mpa':target,
            'predicted_pressure_mpa':predicted,'relative_error_percent':None if predicted is None else 100*(predicted-target)/target,
            'within_10_percent':None if predicted is None else abs(predicted-target)<=.1*target,
            'model_validation_status':'unvalidated','notes':case.notes,
            'warning':'A pressure comparison alone does not validate geometry, material, process or failure criterion.'}
    write_json(directory/'results/reference_comparison.json',result)
    return result
