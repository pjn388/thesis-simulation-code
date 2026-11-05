"""Capture native local inputs/returns at the retained late trial, without solver edits."""
from pathlib import Path
base=Path(__file__).with_name('capture_pipe_solver_gdb.py').read_text()
start=base.index('class IP(gdb.Breakpoint):')
end=base.index('class ResidualDone',start)
replacement="class LocalDone(gdb.FinishBreakpoint):\n    def __init__(self,p,row,offset,ns):\n        super().__init__(gdb.newest_frame(),internal=True)\n        self.ptr={i:p(i) for i in (3,10,27)};self.row=row;self.offset=offset;self.ns=ns\n    def stop(self):\n        self.row.update(current=read(self.ptr[3]+8*self.offset,self.ns,'d'),stress=read(self.ptr[10],6,'d'),depvisc=read(self.ptr[27],1,'d')[0])\n        return False\nclass IP(gdb.Breakpoint):\n    def stop(self):\n        p=args();element=read(p(19),1,'i')[0]\n        if element not in (1189,2669):return False\n        ip=read(p(20),1,'i')[0];ns=read(p(21),1,'i')[0];mi=read(p(22),3,'i')\n        offset=ns*((element-1)*mi[0]+ip-1)\n        row=dict(element=element,ip=ip,elconloc=read(p(1),2,'d'),plconloc=read(p(2),802,'d'),strain=read(p(6),6,'d'),previous=read(p(4)+8*offset,ns,'d'),beta=read(p(9),6,'d'),vj=read(p(11),1,'d')[0],kode=read(p(12),1,'i')[0],ielas=read(p(13),1,'i')[0],icmd=read(p(8),1,'i')[0],temperature=read(p(15),1,'d')[0],dtime=read(p(16),1,'d')[0])\n        ips.append(row);LocalDone(p,row,offset,ns)\n        return False\n"
base=base[:start]+replacement+base[end:]
exec(compile(base,str(Path(__file__).with_name('capture_pipe_solver_gdb.py')),'exec'))
