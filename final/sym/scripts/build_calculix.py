"""Build official CalculiX 2.23 locally on Debian; never invokes apt or sudo.

Install the system packages documented in SYSTEM_PACKAGES.md first.
Downloads are checksum-pinned to archives inspected during implementation.
"""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
ARCHIVES=[
    ('https://www.dhondt.de/ccx_2.23.src.tar.bz2','9c88385c10fb04f5dc6c4e98027a51bebdd8aee3920e05190d6c1dd08357d6e7',''),
    ('https://www.netlib.org/linalg/spooles/spooles.2.2.tgz','a84559a0e987a1e423055ef4fdf3035d55b65bbe4bf915efaa1a35bef7f8c5dd','spooles'),
    ('https://www.dhondt.de/ccx_2.23.htm.tar.bz2','ed14b31b51972d5492209a42fcd52a062e36cbc7843bcd358617cb3aee0da736','')]


def main():
    for tool in ['gcc','gfortran','make','ar']:
        if not shutil.which(tool):raise RuntimeError('Missing '+tool+'; see SYSTEM_PACKAGES.md')
    vendor=ROOT/'vendor';vendor.mkdir(exist_ok=True)
    docs=ROOT/'docs';docs.mkdir(exist_ok=True)
    for url,digest,target in ARCHIVES:
        archive=vendor/url.rsplit('/',1)[1]
        if not archive.exists():urllib.request.urlretrieve(url,archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise RuntimeError('Checksum mismatch: '+str(archive))
        dest=vendor/target;dest.mkdir(exist_ok=True)
        with tarfile.open(archive) as tf:tf.extractall(dest,filter='data')
    p=vendor/'spooles/Make.inc'
    p.write_text(p.read_text().replace('CC = /usr/lang-4.0/bin/cc','CC = gcc').replace('CFLAGS = $(OPTLEVEL)\n','CFLAGS = $(OPTLEVEL) -std=gnu89 -fcommon\n'))
    source=vendor/'CalculiX/ccx_2.23/src'
    multiarch=subprocess.check_output(['gcc','-print-multiarch'],text=True).strip()
    arpack='/usr/lib/'+multiarch+'/libarpack.so'
    if not Path(arpack).exists():raise RuntimeError('Missing libarpack2-dev')
    p=source/'Makefile'
    p.write_text(p.read_text().replace('../../../SPOOLES.2.2','../../../spooles')
        .replace('../../../ARPACK/libarpack_INTEL.a',arpack)
        .replace('-lpthread -lm -lc','-llapack -lblas -lpthread -lm -lc')
        .replace('FFLAGS = -Wall -O2 -cpp','FFLAGS = -Wall -O2 -cpp -fallow-argument-mismatch')
        .replace('-DNETWORKOUT','-DNETWORKOUT -std=gnu99 -Wno-error=return-mismatch')
        .replace('ccx_2.23.a  $(LIBS)','ccx_2.23.a'))
    for name,folder,cmd in [('spooles',vendor/'spooles',['make','lib']),('ccx',source,['make','-j4'])]:
        print('Building '+name,flush=True)
        with (docs/(name+'-build.log')).open('w') as log:
            subprocess.run(cmd,cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
    exe=source/'ccx_2.23'
    subprocess.run([str(exe),'-v'],check=True)
    (docs/'ccx-ldd.txt').write_text(subprocess.check_output(['ldd',str(exe)],text=True))
    metadata={'version':'2.23','platform':platform.platform(),
              'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
              'archives':[{'url':u,'sha256':h} for u,h,_ in ARCHIVES],
              'compiler':subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
              'notes':'GCC14 return-mismatch warning downgraded in upstream readnewmesh.c; no solver physics modifications.'}
    (docs/'solver-build.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':main()
