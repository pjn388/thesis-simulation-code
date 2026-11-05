"""Explicit additive/multiplicative reference hypotheses for matched cases."""
import hashlib
import json
import pandas as pd


def comparison_group(case):
    d=case.to_dict()
    # Compare only physically matched cases; mesh controls deliberately excluded
    # so convergence runs can be inspected, but duplicate baselines are rejected.
    for key in ['name','purpose','notes','probe_positions','mesh']:d.pop(key,None)
    d['gouge'].pop('enabled');d['gouge'].pop('angle_deg');d['dent'].pop('enabled')
    for key in ['linear_solver','threads','timeout_seconds','initial_increment','max_increment','min_increment','max_increments','output_frequency']:
        d['analysis'].pop(key,None)
    return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()[:12]


def interaction_predictions(intact, dent, gouge):
    if intact<=0 or min(dent,gouge)<=0:raise ValueError('Burst pressures must be positive')
    return {'additive_prediction_mpa':dent+gouge-intact,
            'multiplicative_prediction_mpa':dent*gouge/intact}


def interaction_table(comparisons):
    output=[]
    for signature,group in comparisons.groupby('comparison_group'):
        valid=group[(group.status=='criterion_crossed')&group.burst_pressure_mpa.notna()]
        intact=valid[(~valid.dent)&(~valid.gouge)]
        dent=valid[valid.dent&(~valid.gouge)]
        for angle in sorted(group[group.gouge].angle_deg.unique()):
            gouge=valid[(~valid.dent)&valid.gouge&(valid.angle_deg==angle)]
            combined=valid[valid.dent&valid.gouge&(valid.angle_deg==angle)]
            row={'comparison_group':signature,'angle_deg':angle,'status':'missing_or_ambiguous_matched_burst_results'}
            if all(len(x)==1 for x in [intact,dent,gouge,combined]):
                p0,p_dent,pg,pdg=[float(x.burst_pressure_mpa.iloc[0]) for x in [intact,dent,gouge,combined]]
                predictions=interaction_predictions(p0,p_dent,pg)
                row.update(status='descriptive_comparison_only',intact_mpa=p0,dent_mpa=p_dent,gouge_mpa=pg,combined_mpa=pdg,
                           dent_change_relative_to_gouge_percent=100*(pdg/pg-1),**predictions,
                           additive_residual_mpa=pdg-predictions['additive_prediction_mpa'],
                           multiplicative_residual_mpa=pdg-predictions['multiplicative_prediction_mpa'])
            output.append(row)
    return pd.DataFrame(output)
