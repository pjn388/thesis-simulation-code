"""Output-only first-IP incplas calls; immutable parent deck, serial GDB."""
import gdb, json, os, struct
from pathlib import Path
records=[]
def read(a,n,code):
    return list(struct.unpack('<'+code*n,gdb.selected_inferior().read_memory(a,struct.calcsize(code)*n).tobytes()))
def reg(s): return int(gdb.parse_and_eval(s))
class Done(gdb.FinishBreakpoint):
    def __init__(self,record,addresses):
        super().__init__(gdb.newest_frame(),internal=True)
        self.record,self.addresses=record,addresses
    def stop(self):
        for key,a,n in self.addresses:
            self.record[key]=read(a,n,'d')
        records.append(self.record)
        return False
class Entry(gdb.Breakpoint):
    def stop(self):
        stack=reg('$rsp')
        def p(i):
            return reg(['$rdi','$rsi','$rdx','$rcx','$r8','$r9'][i-1]) if i<=6 else read(stack+8*(i-6),1,'Q')[0]
        iel,iint=read(p(19),1,'i')[0],read(p(20),1,'i')[0]
        icmd=read(p(8),1,'i')[0]
        if iel!=1 or iint!=1 or icmd==3:return False
        nstate=read(p(21),1,'i')[0]
        assert nstate>=13
        record=dict(icmd=icmd,iel=iel,iint=iint,emec=read(p(6),6,'d'),previous_state=read(p(4),nstate,'d'),elastic=read(p(1),2,'d'),time=read(p(17),1,'d')[0],ttime=read(p(18),1,'d')[0],dtime=read(p(16),1,'d')[0])
        Done(record,[('stress',p(10),6),('stiffness',p(5),21),('current_state',p(3),nstate)])
        return False
Entry('*incplas_',internal=True)
gdb.execute('run')
assert records
Path(os.environ['THESIS_TANGENT_OUTPUT']).write_text(json.dumps(dict(records=records,limitation='First IP, stiffness-requested calls only; native local stress/stiffness, not assembled residual or equilibrium acceptance.'),indent=2)+'\n')
