"""Recover accepted-frame evidence from interrupted serial captures; never alter native files."""
import argparse
import json
from pathlib import Path
import numpy as np
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_capture_rounding import half_quantum
from audit_pressure_failures import converged_times, matches_sta
from audit_end_surface_traction import face_nodes
from audit_nodal_pressure_loads import nodal_loads

FINAL = Path(__file__).resolve().parents[2]


def main(tag):
    root = FINAL/'sym/runs/investigation'/tag
    run = root/'intact_support_bounded_20261010T170002Z'
    output = FINAL/'results/output'/f'{tag}_partial.json'
    assert not output.exists()
    model = Model.load(run/'model.json')
    deck = (run/'job.inp').read_text()
    assert not any(k in deck.upper() for k in ['*MPC','*EQUATION','*CONTACT','*CLOAD'])
    pressure = json.loads((run/'case.json').read_text())['analysis']['max_pressure']
    accepted = converged_times(run/'job.sta')
    frames = {}
    for kind,group,t,rows in dat_blocks(run/'job.dat'):
        if (group=='PIPE_NODES' and kind in ('U','RF')) or (group=='PIPE' and kind=='PEEQ'):
            if matches_sta(accepted,t): frames.setdefault(t,{})[kind] = rows
    assert frames and all('U' in f and 'RF' in f and 'PEEQ' in f for f in frames.values())
    tokens = {}; time = None
    for line in (run/'job.dat').read_text().splitlines():
        if line.startswith(' forces '):
            time = float(line.split('time')[1]) if 'set PIPE_NODES ' in line else None
            if time is not None: tokens[time] = {}
        elif time is not None:
            fields = line.split()
            if len(fields)==4 and fields[0].isdigit(): tokens[time][int(fields[0])] = fields[1:]
            elif line.strip(): time = None
    pins = set()
    for line in deck.split('*BOUNDARY\n')[1].split('*')[0].strip().splitlines():
        n,first,last,value = [v.strip() for v in line.split(',')]
        assert float(value)==0
        ids = [int(n)] if n.isdigit() else model.nsets[n]
        pins.update((i,d-1) for i in ids for d in range(int(first),int(last)+1))
    faces = {}
    for line in deck.split('*DLOAD\n')[1].split('*')[0].strip().splitlines():
        e,f,p = [v.strip() for v in line.split(',')]
        faces.setdefault(int(e),[]).append((int(f[1:]),float(p)))
    # Keep the last complete cycle at each accepted time. Incomplete trailing
    # records are explicitly excluded, including a timeout-truncated JSON line.
    # Store byte offsets rather than every element record for every accepted
    # frame. Plastic iteration histories can exceed a service's memory budget.
    batches = {}; seen = []; cycles = 0; truncated = False
    capture = (run/'capture.jsonl').open('rb')
    while True:
        offset = capture.tell()
        line = capture.readline()
        if not line: break
        try: r = json.loads(line)
        except json.JSONDecodeError:
            truncated = True
            assert not line.endswith(b'\n')
            break
        key = tuple(r[k] for k in ('routine','istep','iinc','time','dtime'))
        if not seen: metadata = key; batch_start = offset
        assert key==metadata and r['routine']=='e_c3d_'
        assert r['connectivity']==model.elements[r['nelem']][1]
        seen.append(r['nelem'])
        if len(seen)==len(model.elements):
            assert seen==sorted(model.elements)
            cycles += 1
            matching_frames = [t for t in frames if abs(t-r['time'])<1e-7]
            assert len(matching_frames)<=1
            if matching_frames: batches[matching_frames[0]] = batch_start
            seen = []
    history = []
    for t,blocks in sorted(frames.items()):
        matching = [k for k in batches if abs(k-t)<1e-7]
        assert len(matching)==1
        assembled = {n:np.zeros(3) for n in model.nodes}; local_error = 0.
        capture.seek(batches[matching[0]])
        for _ in model.elements:
            r = json.loads(capture.readline())
            conn = r['connectivity']; current = np.array(r['coordinates'])+np.array(r['displacements_at_assembly'])
            native = np.array(r['ff']).reshape(20,3); expected = np.zeros((20,3))
            for f,p in faces.get(r['nelem'],[]):
                indices = [conn.index(n) for n in face_nodes(model,r['nelem'],f)]
                expected[indices] += nodal_loads(current[indices],-p*r['time'])
            local_error = max(local_error,float(np.max(abs(native-expected))))
            for n,v in zip(conn,native): assembled[n] += v
        assert local_error<1e-8
        rf = tokens[t]; u = {int(row[0]):row[1:] for row in blocks['U']}
        assert set(rf)==set(u)==set(model.nodes)
        assert max(abs(u[n][d]) for n,d in pins)==0
        excess = max(max(0.,abs(float(rf[n][d])-assembled[n][d])-half_quantum(rf[n][d])) for n in rf for d in range(3) if (n,d) not in pins)
        scale = max(float(np.max(abs(v))) for v in assembled.values())
        history.append(dict(time=t,pressure_mpa=t*pressure,max_rounding_excess_n=excess,passed=bool(excess<1e-4 and excess/scale<2e-5),local_load_error_n=local_error,maximum_peeq=max(row[-1] for row in blocks['PEEQ']),crown_inward_mm=-u[model.nsets['CROWN'][0]][1]))
    capture.close()
    report = dict(run=str(run.relative_to(FINAL)),accepted_frames=len(history),complete_cycles=cycles,excluded_trailing_records=len(seen),truncated_final_line=truncated,history=history,passed=all(h['passed'] for h in history),hashes={f:sha256(run/f) for f in ('job.inp','mesh.inp','model.json','case.json','job.sta','job.dat','capture.jsonl','solver.log')},script_sha256=sha256(__file__),limitations='Partial accepted-frame audit only. Original 1e-4 N / 2e-5 allowances retained after literal DAT rounding. Last complete assembly cycle at accepted time; serial no-MPC/contact identity assumptions unchanged. No completion, cap-work/tangent, Abaqus equivalence, substantial plastic-transfer or physical burst claim.')
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('accepted_frames','complete_cycles','excluded_trailing_records','truncated_final_line','passed')}))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    main(parser.parse_args().tag)
