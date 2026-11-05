"""Audit retained no-MPC capture against literal DAT rounding intervals."""
import json
import sys
from decimal import Decimal
from pathlib import Path
import numpy as np
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]


def half_quantum(token):
    return float(Decimal(5).scaleb(Decimal(token).as_tuple().exponent - 1))


def main(tag):
    report_path = FINAL/'results/output'/f'{tag}.json'
    original = json.loads(report_path.read_text())
    records = []
    for case in original['cases']:
        run = FINAL/case['run']
        assert all(sha256(run/f) == digest for f,digest in case['hashes'].items())
        model = Model.load(run/'model.json')
        deck = (run/'job.inp').read_text()
        assert not any(k in deck.upper() for k in ['*MPC','*EQUATION','*CONTACT','*CLOAD'])
        pins = set()
        for line in deck.split('*BOUNDARY\n')[1].split('*')[0].strip().splitlines():
            n,first,last,value = [x.strip() for x in line.split(',')]
            assert float(value) == 0
            ids = [int(n)] if n.isdigit() else model.nsets[n]
            pins.update((i,d-1) for i in ids for d in range(int(first),int(last)+1))
        frames = {}; time = None
        for line in (run/'job.dat').read_text().splitlines():
            if line.startswith(' forces '):
                time = float(line.split('time')[1]) if 'set PIPE_NODES ' in line else None
                if time is not None: frames[time] = {}
            elif time is not None:
                tokens = line.split()
                if len(tokens)==4 and tokens[0].isdigit():
                    frames[time][int(tokens[0])] = tokens[1:]
                elif line.strip(): time = None
        accepted = converged_times(run/'job.sta')
        assert len(frames)==case['accepted_frames']
        batches = {}; seen = []; assembly = None
        for line in (run/'capture.jsonl').open():
            r = json.loads(line)
            key = (r['routine'],r['istep'],r['iinc'],r['time'],r['dtime'])
            if not seen:
                metadata = key
                assembly = {n:np.zeros(3) for n in model.nodes}
            assert key==metadata
            seen.append(r['nelem'])
            for n,v in zip(r['connectivity'],np.array(r['ff']).reshape(20,3)):
                assembly[n] += v
            if len(seen)==len(model.elements):
                assert seen==sorted(model.elements)
                batches[r['time']] = assembly
                seen = []
        assert not seen
        history = []
        for time,rf in sorted(frames.items()):
            assert matches_sta(accepted,time) and set(rf)==set(model.nodes)
            matches = [t for t in batches if abs(t-time)<1e-7]
            assert len(matches)==1
            assembly = batches[matches[0]]
            errors = []; excesses = []; quanta = []; worst = None
            for n,tokens in rf.items():
                for d,token in enumerate(tokens):
                    if (n,d) in pins: continue
                    error = abs(float(token)-assembly[n][d])
                    bound = half_quantum(token)
                    excess = max(0.,error-bound)
                    errors.append(error); excesses.append(excess); quanta.append(bound)
                    if worst is None or excess>worst['excess_n']:
                        worst = dict(node=n,component=d+1,token=token,captured_n=float(assembly[n][d]),error_n=error,half_quantum_n=bound,excess_n=excess)
            # Preserve the original absolute/relative allowances AFTER accounting
            # for the known observation precision; do not claim native RF identity.
            scale = max(float(np.max(abs(v))) for v in assembly.values())
            passed = bool(max(excesses)<1e-4 and max(excesses)/scale<2e-5)
            history.append(dict(time=time,max_raw_error_n=max(errors),max_rounding_excess_n=max(excesses),max_half_quantum_n=max(quanta),passed=passed,worst_excess=worst))
        peeq = [dict(time=t,maximum=max(r[-1] for r in rows)) for kind,group,t,rows in dat_blocks(run/'job.dat') if kind=='PEEQ' and group=='PIPE']
        records.append(dict(run=case['run'],history=history,peeq=peeq,passed=all(h['passed'] for h in history)))
    output = dict(records=records,source_sha256=sha256(report_path),script_sha256=sha256(__file__),
                  limitations='Printed RF intervals only, no unrounded native internal-force capture. Exact half-unit bounds derived from each literal decimal token; unchanged 1e-4 N/2e-5 residual allowance. No MPC/contact, cap tangent/work, Abaqus equivalence or physical validation.')
    path = FINAL/'results/output'/f'{tag}_rounding.json'
    assert not path.exists()
    path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(passed=all(r['passed'] for r in records),frames=sum(len(r['history']) for r in records),max_excess_n=max(h['max_rounding_excess_n'] for r in records for h in r['history']))))


if __name__=='__main__': main(sys.argv[1])
