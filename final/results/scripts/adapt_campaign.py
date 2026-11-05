"""Append memory-efficient mesh attempts without changing frozen original cases."""
import copy,json
from pipe_sim.config import Case
from pipe_sim.runner import write_json,sha256
from campaign import PLAN,CASES,FINAL
p=json.loads(PLAN.read_text())
if any(j['id'].endswith('_adaptive') for j in p['jobs']):raise ValueError('Adaptive attempts already added')
new=[]
for j in p['jobs']:
    if j['kind']!='orientation' and j['estimated_elements']<12000:continue
    c=Case.load(FINAL/j['case_file']);c.name+='_adaptive';c.mesh.strategy='adaptive_quads'
    c.notes+=['Adaptive all-quad surface mesh extruded through the original wall-layer count; same nominal local size and physical inputs. Structured meshes exceeded this qube memory budget. Convergence must still be checked.']
    path=CASES/(c.name+'.json');c.save(path);a=copy.deepcopy(j)
    a.update(id=c.name,case_file=str(path.relative_to(FINAL)),case_sha256=sha256(path),run='sym/runs/campaign/'+c.name,
             estimated_elements=None,structured_estimate=j['estimated_elements'],parent_attempt=j['id'])
    if a['entry']:a['entry'].update(id=c.name,case_file=a['case_file'],case_sha256=a['case_sha256'],assumptions=c.notes)
    j['superseded_by']=c.name;new.append(a)
p['jobs']+=new;p['orientation_combined_0']='full_zhao2021_p1_adaptive'
p['mesh_revision']='Adaptive variants preserve original attempts; no material, geometry or failure criterion fitting.'
write_json(PLAN,p)
print('Added',len(new),'adaptive attempts')
