"""Fast terminal PEEQ block audit while full invariant extraction continues."""
import json
from pathlib import Path
import sys
from pipe_sim.runner import sha256, write_json


def audit(angle):
    final=Path(__file__).resolve().parents[2]
    d=final/f'sym/runs/campaign/orientation_combined_{angle}_adaptive_pressure005_20261020'
    state=json.loads((d/'status.json').read_text())
    assert state['state'] in ('failed','interrupted','completed')
    model=json.loads((d/'model.json').read_text())
    expected=8*len(model['elsets']['PIPE'])
    rows=[]; current=None; maximum=0.; count=0
    def flush():
        if current is not None:
            assert count==expected,(current,count,expected)
            rows.append(dict(time=current,pressure_mpa=max(0.,min(1.,current-2.))*35.,max_peeq=maximum))
    with (d/'job.dat').open() as stream:
        for line in stream:
            if 'for set' in line and 'and time' in line:
                flush();current=None
                if line.lstrip().startswith('equivalent plastic strain') and 'for set PIPE ' in line:
                    current=float(line.split('and time')[1]);maximum=0.;count=0
            elif current is not None:
                parts=line.split()
                if len(parts)==3 and parts[0].isdigit() and parts[1].isdigit():
                    maximum=max(maximum,float(parts[2]));count+=1
    flush()
    pressure=[r for r in rows if r['time']>2.]
    crossing=next((i for i,r in enumerate(pressure) if r['max_peeq']>=.8),None)
    assert crossing is not None and crossing>0
    assert max(r['max_peeq'] for r in rows if r['time']<=2.)<.8
    bracket=pressure[crossing-1:crossing+1]
    out=final/f'results/output/orientation{angle}_pressure005_20261020'
    out.mkdir(exist_ok=True)
    write_json(out/'terminal_peeq.json',dict(case_id=d.name,solver_state=state['state'],stop_reason=state.get('stop_reason'),expected_ip_per_block=expected,complete_blocks=len(rows),bracket=bracket,endpoint=pressure[-1],maximum_peeq=max(r['max_peeq'] for r in rows),status_sha256=sha256(d/'status.json'),dat_sha256=sha256(d/'job.dat'),script_sha256=sha256(Path(__file__)),limitations='Selective complete PEEQ blocks only; full contact/history/invariant extraction remains authoritative. Criterion0.8 exceeds material support0.791057. No physical burst or ranking.'))
    print(json.dumps(dict(bracket=bracket,endpoint=pressure[-1],maximum_peeq=max(r['max_peeq'] for r in rows))))

if __name__=='__main__':audit(int(sys.argv[1]))
