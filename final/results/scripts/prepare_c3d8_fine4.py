"""Independent surface sensitivity from retained C3D8 wall4 parents."""
import copy
import json
from formulation_trials import campaign, Case, sha256, write_json, TAG

def prepare():
    plan=json.loads(campaign.PLAN.read_text()); children=[]
    for specimen in ('p1','p4'):
        parent_id=f'full_zhao2021_{specimen}_adaptive_{TAG}'
        parent=next(j for j in plan['jobs'] if j['id']==parent_id)
        name=parent_id+'_fine4'
        assert not any(j['id']==name for j in plan['jobs'])
        source=campaign.FINAL/parent['case_file'];assert sha256(source)==parent['case_sha256']
        case=Case.load(source);before=case.to_dict()
        assert case.mesh.fine_size==5 and case.mesh.wall_layers==4 and case.mesh.order==1
        case.mesh.fine_size=4;case.name=name
        case.notes.append('Independent C3D8 surface sensitivity after wall6: fine_size5to4 only, retaining wall4 parent, material, contact, loading, criterion and all other mesh settings. Not a fitted physical model or burst prediction. Compare full branches, release, common-pressure PEEQ and criterion brackets; do not infer mesh convergence from one refinement.')
        case.validate()
        for key in ('pipe','gouge','dent','material','analysis','probe_positions'):assert case.to_dict()[key]==before[key]
        assert {k:v for k,v in case.to_dict()['mesh'].items() if k!='fine_size'}=={k:v for k,v in before['mesh'].items() if k!='fine_size'}
        path=campaign.CASES/(name+'.json');assert not path.exists();case.save(path)
        child=copy.deepcopy(parent)
        child.update(id=name,parent_attempt=parent_id,case_file=str(path.relative_to(campaign.FINAL)),case_sha256=sha256(path),run='sym/runs/campaign/'+name,changes={'mesh':{'fine_size':4}},description='Independent C3D8 surface5to4 sensitivity; wall4 and physics unchanged')
        child.pop('superseded_by',None)
        child['entry'].update(id=name,case_file=child['case_file'],case_sha256=child['case_sha256'],assumptions=case.notes)
        plan['jobs'].append(child);children.append(child)
    write_json(campaign.PLAN,plan)
    work_path=campaign.FINAL/'results/config/work_plan.json';work=json.loads(work_path.read_text())
    rows=__import__('pandas').read_csv(campaign.FINAL/'results/output/c3d8_20261019/wall6/comparison.csv')
    for task in work['tasks']:
        if task['id'].endswith('_wall6'):
            row=rows[rows.case_id==task['id']].iloc[0]
            task.update(state='completed_failed_before_criterion',finding=f'Minimum-increment convergence stop at {row.last_pressure_mpa:.8g}MPa; maxPEEQ{row.max_peeq:.8g}; peak error{row.peak_error_percent:+.5f}percent. Withdrawal/contact pass; no0.8crossing. See wall6 comparison.',next='Retain missing burst. Independent fine4 surface child isolates remaining discretization uncertainty; no wall-layer sweep or unchanged rerun.')
    for child in children:
        work['tasks'].append(dict(id=child['id'],state='prepared',service='thesis-c3d8-'+('p1' if '_p1_' in child['id'] else 'p4')+'-fine4-20261019.service',run=child['run'],parent=child['parent_attempt'],budget='8hour/11000MiB native guard;12GiB service;2threads;maximum2native jobs',hypothesis='Independent surface resolution may affect C3D8 denting stiffness and strain; wall refinement changed peaks6to7percent and PEEQ10to15percent, so discretization remains unresolved.',next='Compare full branches/physical and paperFE errors, release/contact, residual crown, source strain range, common-pressure PEEQ and unchanged0.8 brackets against wall4/fine5. A missing crossing remains missing. No further mesh sweep or orientation-strength claim without an explicit terminal decision.'))
    work['current_handoff']['next']='Wall6 terminal P1/P4:26.91885/26.92655MPa,maxPEEQ.6571398/.3392451; no0.8crossings. Peak errors+62.2757/+44.4481percent; contact/withdrawal pass. Independent wall4 surface5to4 children prepared, not wall6 combined refinements. Compare physical branches and local strain before orientation selection; preserve parent provisional crossing and all failed attempts.'
    write_json(work_path,work)
    print([c['id'] for c in children])

if __name__=='__main__':prepare()
