"""Make a new registered case from a paper case using explicit parameter overrides."""
import argparse
import copy
import json
from pipe_sim.config import Case
from prepare_cases import FINAL, RESULTS, CASES, digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('base_id');p.add_argument('new_id')
    p.add_argument('--set',action='append',default=[],metavar='PARAMETER=JSON',help='For example analysis.max_pressure=30 or mesh.wall_layers=4')
    p.add_argument('--note',required=True,help='Why this variant is being run; retained on comparisons')
    args=p.parse_args();registry=RESULTS/'config/literature_cases.json';entries=json.loads(registry.read_text())
    if any(e['id']==args.new_id for e in entries):p.error('New ID already exists')
    base=next((e for e in entries if e['id']==args.base_id),None)
    if base is None:p.error('Unknown base case')
    entry=copy.deepcopy(base);data=Case.load(FINAL/base['case_file']).to_dict();changes={}
    for item in args.set:
        key,value=item.split('=',1);keys=key.split('.');target=data
        for k in keys[:-1]:target=target[k]
        if keys[-1] not in target:p.error('Unknown parameter: '+key)
        converted=json.loads(value);changes[key]={'old':target[keys[-1]],'new':converted};target[keys[-1]]=converted
    data['name']=args.new_id;data['notes']+=['Variant of '+args.base_id+': '+args.note]
    if any(k.startswith('material.') for k in changes):
        data['notes']+=['Material parameters changed: the parent paper curve is retained only as a reference; its supported strain range is not inherited.']
        entry['material'].update(custom_input=True,source_supported_plastic_strain_max=0.,interpretation_confirmed=False)
    case=Case.from_dict(data);path=CASES/(case.name+'.json')
    if path.exists():p.error('Case file already exists')
    case.save(path)
    entry.update(id=case.name,case_file=str(path.relative_to(FINAL)),case_sha256=digest(path),
                 assumptions=case.notes,parent_case=args.base_id,parameter_changes=changes)
    entries.append(entry);registry.write_text(json.dumps(entries,indent=2)+'\n');print(path)


if __name__=='__main__':main()
