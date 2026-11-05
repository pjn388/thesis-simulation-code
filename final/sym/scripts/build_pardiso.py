"""Isolated CCX 2.23 build with the contact array fix and oneMKL PARDISO."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'vendor/CalculiX/ccx_2.23/src'
FIX=ROOT/'vendor/ccx_tabular_fix'
OUT=ROOT/'vendor/ccx_pardiso'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True);venv=Path(sys.prefix);include=venv/'include';lib=venv/'lib'
    if not (include/'mkl_service.h').exists():raise RuntimeError('Run uv sync --group solver first')
    files=[p for p in SRC.glob('*.c') if '#ifdef PARDISO' in p.read_text() or '#if defined(PARDISO)' in p.read_text() or '#elif defined(PARDISO)' in p.read_text()]
    # Include every translation unit whose preprocessor depends on PARDISO.
    import re
    files=[p for p in SRC.glob('*.c') if re.search(r'^\s*#.*PARDISO',p.read_text(),re.M)]
    def compile(p):
        cmd=['gcc','-O2','-std=gnu99','-Wno-error=return-mismatch','-DPARDISO','-DSPOOLES','-DARPACK','-DMATRIXSTORAGE','-DNETWORKOUT','-DARCH=Linux','-I',str(SRC),'-I',str(ROOT/'vendor/spooles'),'-I',str(include),'-c',str(p),'-o',str(OUT/(p.stem+'.o'))]
        result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (OUT/(p.stem+'.log')).write_text(result.stdout)
        if result.returncode:raise RuntimeError(result.stdout[-3000:])
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(compile,files))
    shutil.copy2(FIX/'ccx_2.23.a',OUT/'ccx_2.23.a')
    subprocess.run(['ar','r',str(OUT/'ccx_2.23.a')]+[str(OUT/(p.stem+'.o')) for p in files],check=True)
    exe=OUT/'ccx_2.23_pardiso'
    subprocess.run(['gfortran','-O2','-o',str(exe),str(SRC/'ccx_2.23.o'),str(OUT/'ccx_2.23.a'),str(ROOT/'vendor/spooles/spooles.a'),'/usr/lib/x86_64-linux-gnu/libarpack.so',
      '-L',str(lib),'-Wl,-rpath,'+str(lib),'-l:libmkl_rt.so.2','-llapack','-lblas','-lpthread','-lm','-ldl','-fopenmp'],check=True)
    metadata={'executable':str(exe),'solver_sha256':sha(exe),'contact_patch_build':json.loads((FIX/'build.json').read_text()),
      'mkl_runtime_sha256':sha(lib/'libmkl_rt.so.2'),'compiled_sources':{p.name:sha(p) for p in files},'uv_lock_sha256':sha(ROOT/'uv.lock'),'verified':False}
    (OUT/'build.json').write_text(json.dumps(metadata,indent=2)+'\n');print(exe)
if __name__=='__main__':main()
