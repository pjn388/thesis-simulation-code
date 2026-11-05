"""Isolated experimental build: opt-in existing line search for contact-free statics.

Keeps native material, loads, convergence tolerances and default behavior intact.
Not a production replacement or an exact-tangent implementation.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'vendor/CalculiX/ccx_2.23/src'
PARENT = ROOT/'vendor/ccx_pardiso'
OUT = ROOT/'vendor/ccx_static_linesearch_v3_20261019'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parent = json.loads((PARENT/'build.json').read_text())
    assert sha(PARENT/'ccx_2.23_pardiso') == parent['solver_sha256']
    original = SRC/'nonlingeo.c'
    assert sha(original) == parent['compiled_sources']['nonlingeo.c']
    text = original.read_text()
    old = 'if((*mortar==1)&&(iit!=1)&&(*ne-ne0>0)&&(*nmethod!=4)){'
    new = ('if(((*mortar==1)&&(iit!=1)&&(*ne-ne0>0)&&(*nmethod!=4)) || '
           '((getenv("THESIS_STATIC_LINESEARCH") != NULL) && '
           '(strcmp(getenv("THESIS_STATIC_LINESEARCH"),"1")==0) && '
           '(*nmethod==1) && (*ithermal<2) && (iit!=1) && '
           '(*ne==ne0) && (*mortar<=1) && (*nmpc==0))){')
    assert text.count(old) == 2
    OUT.mkdir(exist_ok=False)
    modified = OUT/'nonlingeo.c'
    modified.write_text(text.replace(old, new))
    # Both branches must change together: save old residual, then damp/update.
    assert modified.read_text().replace(new, old) == text
    include = Path(sys.prefix)/'include'; lib = Path(sys.prefix)/'lib'
    commands = [
        ['gcc','-O2','-std=gnu99','-Wno-error=return-mismatch','-DPARDISO','-DSPOOLES',
         '-DARPACK','-DMATRIXSTORAGE','-DNETWORKOUT','-DARCH=Linux','-I',str(SRC),
         '-I',str(ROOT/'vendor/spooles'),'-I',str(include),'-c',str(modified),'-o',str(OUT/'nonlingeo.o')],
        ['ar','r',str(OUT/'ccx_2.23.a'),str(OUT/'nonlingeo.o')],
        ['gcc','-O2','-o',str(OUT/'ccx_2.23_linesearch'),str(SRC/'ccx_2.23.o'),
         str(OUT/'ccx_2.23.a'),str(ROOT/'vendor/spooles/spooles.a'),'-l:libarpack.so.2',
         '-L',str(lib),'-Wl,-rpath,'+str(lib),'-l:libmkl_rt.so.2','-l:liblapack.so.3','-l:libblas.so.3',
         '-l:libgfortran.so.5','-l:libquadmath.so.0',
         '-lpthread','-lm','-ldl','-fopenmp']]
    shutil.copy2(PARENT/'ccx_2.23.a', OUT/'ccx_2.23.a')
    for index, command in enumerate(commands):
        with (OUT/f'build_{index}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    report = dict(executable=str(OUT/'ccx_2.23_linesearch'),
                  solver_sha256=sha(OUT/'ccx_2.23_linesearch'), parent_solver_sha256=parent['solver_sha256'],
                  source_sha256=sha(original), modified_sha256=sha(modified),
                  builder_sha256=sha(__file__), commands=commands, verified=False,
                  source_reference='nonlingeo.c:2211,2753-2757,3210-3290,3484: allocation, residual save, existing secant line search and repeated results evaluation, free.',
                  scope='Opt-in THESIS_STATIC_LINESEARCH=1, static mechanical, no added contact elements, no MPC. Only two branch conditions changed. Native contact branch and disabled behavior retained. No material, load, increment, tolerance or failure-criterion change. Experimental convergence method; physical acceptance remains separate.')
    (OUT/'build.json').write_text(json.dumps(report, indent=2)+'\n')
    print(OUT/'ccx_2.23_linesearch')


if __name__ == '__main__':
    main()
