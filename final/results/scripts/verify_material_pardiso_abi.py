"""Exercise unchanged native PARDISO adapter on retained additive matrices.

This is an integration prerequisite, not a nonlinear solver revision.
"""
import json
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
from audit_plastic_return_map import F, sha256


def main():
    report_path = F/'results/output/material_pardiso_abi_20261019.json'
    assert not report_path.exists()
    root = F/'sym/runs/investigation/material_pardiso_abi_v3_20261019'
    root.mkdir(exist_ok=False)
    source = F/'results/scripts/material_pardiso_abi.c'
    src = F/'sym/vendor/CalculiX/ccx_2.23/src'
    adapter = F/'sym/vendor/ccx_pardiso/pardiso.o'
    archive = F/'sym/vendor/ccx_pardiso/ccx_2.23.a'
    library = F/'sym/.venv/lib/libmkl_rt.so.2'
    exe = root/'solve'
    cmd = ['gcc', '-O2', '-Wall', '-Wextra', str(source), str(adapter),
           str(archive), str(library), '-Wl,-rpath,'+str(library.parent),
           '-l:libgfortran.so.5', '-Wl,--no-as-needed', '-lm', '-lpthread', '-o', str(exe)]
    with (root/'build.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    matrix_source = F/'sym/runs/investigation/late_local_inputs_20261019/assembly.json'
    g = json.loads(matrix_source.read_text())['global_matrix']
    n = g['neq']; nz = len(g['au'])
    jq = np.array(g['jq'], dtype='<i4'); ir = np.array(g['irow'], dtype='<i4')
    cols = np.repeat(np.arange(n), np.diff(jq)); rows = ir-1
    # A dense-in-equation-space ray exercises both triangles, independently
    # of the two-element perturbation used to generate the correction.
    direction = np.sin(np.arange(n)*0.123+0.7)
    cases = []
    inputs = [source, adapter, archive, library, matrix_source,
              src/'pardiso.c', F/'results/output/volume_material_scatter_20261019.json']
    for enabled in (0, 1):
        p = F/f'sym/runs/investigation/volume_material_scatter_20261019/enabled{enabled}.bin'
        inputs.append(p)
        data = np.frombuffer(p.read_bytes(), dtype='<f8'); assert len(data)==n+2*nz
        def action(v):
            out = data[:n]*v
            np.add.at(out, rows, data[n:n+nz]*v[cols])
            np.add.at(out, cols, data[n+nz:]*v[rows])
            return out
        rhs = action(direction)
        payload = (struct.pack('<ii', n, nz)+jq.tobytes()+ir.tobytes()+
                   data.tobytes()+rhs.astype('<f8').tobytes())
        inp = root/f'case{enabled}.in'; out = root/f'case{enabled}.out'
        inp.write_bytes(payload)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_THREADING_LAYER='GNU',
                   CCX_NPROC_EQUATION_SOLVER='1')
        with (root/f'case{enabled}.log').open('w') as log:
            subprocess.run([str(exe), str(inp), str(out)], stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=180, env=env)
        solution = np.frombuffer(out.read_bytes(), dtype='<f8'); assert len(solution)==n
        residual = float(np.linalg.norm(action(solution)-rhs)/np.linalg.norm(rhs))
        error = float(np.linalg.norm(solution-direction)/np.linalg.norm(direction))
        cases.append(dict(enabled=bool(enabled), relative_residual=residual,
                          relative_solution_error=error, passed=residual<2e-7 and error<2e-7))
    report = dict(checks_passed=all(c['passed'] for c in cases), cases=cases,
                  relative_limit=2e-7, equations=n, offdiagonal_per_triangle=nz,
                  run=str(root.relative_to(F)), build_command=cmd,
                  sources={str(p.relative_to(F)):sha256(p) for p in [*inputs,Path(__file__),exe]},
                  limitations='Unchanged native PARDISO inputformat=1/symmetryflag=2 on retained parent and localized material-corrected matrices. This tests sparse adapter orientation and solve ABI, not full-ray native FD, live material integration, residual/state restoration, nonlinear convergence or physical burst.',
                  next='Implement isolated opt-in runtime material correction with disabled-parent identity and state restoration; full-ray native FD and coupons still required before pressure successor.')
    with report_path.open('x') as f:
        json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','build_command')},indent=2),flush=True)
    assert report['checks_passed']


if __name__=='__main__':
    main()
