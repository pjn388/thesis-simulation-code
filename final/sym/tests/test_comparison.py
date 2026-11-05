import pandas as pd
import pytest
from pipe_sim.comparison import comparison_group, interaction_predictions, interaction_table
from pipe_sim.config import Case


def test_interaction_hypotheses_and_missing_results():
    predictions=interaction_predictions(20,18,12)
    assert predictions['additive_prediction_mpa']==10
    assert predictions['multiplicative_prediction_mpa']==pytest.approx(10.8)
    rows=[dict(comparison_group='a',dent=d,gouge=g,angle_deg=0,status='criterion_crossed',burst_pressure_mpa=p)
          for d,g,p in [(False,False,20),(True,False,18),(False,True,12),(True,True,9)]]
    output=interaction_table(pd.DataFrame(rows)).iloc[0]
    assert output.additive_residual_mpa==-1
    assert output.dent_change_relative_to_gouge_percent==-25
    assert interaction_table(pd.DataFrame(rows[:-1])).iloc[0].status=='missing_or_ambiguous_matched_burst_results'


def test_mismatched_materials_not_grouped():
    a=Case();b=Case.from_dict(a.to_dict());b.dent.enabled=False;b.gouge.angle_deg=45
    assert comparison_group(a)==comparison_group(b)
    b.material.young=200000
    assert comparison_group(a)!=comparison_group(b)
