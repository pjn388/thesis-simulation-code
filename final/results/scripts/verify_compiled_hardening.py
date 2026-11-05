"""Execute preserved CCX Fortran objects to check the Python resampling audit.

This is a routine-level check, not a nonlinear material or pipe validation.
No compiler installation or native solver/model modification is needed.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_hardening_resampling import FINAL, resample
from pipe_sim.runner import sha256


def main():
    out = FINAL / 'results/output/compiled_hardening_verification'
    out.mkdir(exist_ok=True)
    archive = FINAL / 'sym/vendor/ccx_pardiso/ccx_2.23.a'
    objects = []
    for name in ('plcopy', 'plinterpol'):
        target = out / (name + '.o')
        target.write_bytes(subprocess.check_output(['ar', 'p', str(archive), name + '.o']))
        objects.append(target)
    driver = out / 'driver.c'
    driver.write_text('''#include <stdio.h>
#include <stdlib.h>
extern void plcopy_(double*,int*,double*,int*,int*,int*,int*,int*,int*);
int main(void) {
 int n,j,nt=1,imat=1,itemp=1,nelem=1,kin=0;
 if(scanf("%d",&n)!=1 || n<2) return 2;
 double *p=calloc(2*n+1,sizeof(double)), loc[802]={0};
 int counts[2]={1,n};
 for(j=1;j<=n;j++) if(scanf("%lf %lf",p+2*j-1,p+2*j)!=2) return 3;
 plcopy_(p,counts,loc,&n,&nt,&imat,&itemp,&nelem,&kin);
 int m=(int)loc[800];
 printf("%d\\n",m);
 for(j=0;j<m;j++) printf("%.17g %.17g\\n",loc[2*j+1],loc[2*j]);
 free(p); return 0;
}
''')
    exe = out / 'driver'
    command = ['gcc', '-O2', str(driver), *map(str, objects),
               '/usr/lib/x86_64-linux-gnu/libgfortran.so.5', '-o', str(exe)]
    subprocess.run(command, check=True, capture_output=True)
    records = []
    for case_id in ('orientation_combined_75_adaptive',
                    'orientation_combined_90_adaptive',
                    'full_allouti2012_3_percent_od_figure_curve'):
        run = FINAL / 'sym/runs/campaign' / case_id
        manifest = json.loads((run / 'manifest.json').read_text())
        assert all(sha256(run / n) == h for n, h in manifest['files'].items())
        table = json.loads((run / 'case.json').read_text())['material']['plastic']
        # Also exercise the <=200 direct-copy branch on an unchanged subset.
        for label, data in [('full', table), ('direct_copy', table[:20])]:
            stdin = str(len(data)) + '\n' + '\n'.join(
                f'{stress:.17g} {strain:.17g}' for stress, strain in data) + '\n'
            result = subprocess.check_output([str(exe)], input=stdin, text=True)
            lines = result.splitlines()
            count = int(lines[0])
            actual = np.array([[float(x) for x in line.split()] for line in lines[1:]])
            expected = resample(data)
            assert count == len(expected)
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
            records.append(dict(case_id=case_id, branch=label,
                                input_points=len(data), output_points=count,
                                max_absolute_difference=float(np.max(abs(actual-expected))),
                                case_sha256=sha256(run / 'case.json')))
    sources = [FINAL / f'sym/vendor/CalculiX/ccx_2.23/src/{n}.f'
               for n in ('plcopy', 'plinterpol')]
    evidence = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    records=records, compile_command=command,
                    sha256={str(p.relative_to(FINAL)): sha256(p)
                            for p in [archive, *objects, *sources, driver, exe, Path(__file__),
                                      Path(__file__).with_name('audit_hardening_resampling.py')]},
                    limitation='Executed objects extracted from preserved PARDISO archive; source-to-object rebuild not performed because gfortran is unavailable. Single temperature isotropic table copy/resampling only; no return-mapping, tangent convergence, failure causation, burst or experimental validation established.')
    (out / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
