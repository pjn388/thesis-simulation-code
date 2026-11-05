"""Isolated native incplas call with retained pipe history; no pipe equilibrium."""
import gdb,json,os,struct
from pathlib import Path
state=json.loads(Path(os.environ['THESIS_STATE']).read_text());result={};finished=False

def read(a,n,c):return list(struct.unpack('<'+c*n,gdb.selected_inferior().read_memory(a,struct.calcsize(c)*n).tobytes()))
def write(a,v):gdb.selected_inferior().write_memory(a,struct.pack('<'+'d'*len(v),*v))
class Done(gdb.FinishBreakpoint):
    def __init__(self,p):
        super().__init__(gdb.newest_frame(),internal=True);self.p={i:p(i) for i in (3,5,10)}
    def stop(self):
        global finished
        result.update(stress=read(self.p[10],6,'d'),stiffness=read(self.p[5],21,'d'),current=read(self.p[3],len(state['previous']),'d'));finished=True;return True
class Entry(gdb.Breakpoint):
    def stop(self):
        stack=int(gdb.parse_and_eval('$rsp'))
        def p(i):return int(gdb.parse_and_eval(['$rdi','$rsi','$rdx','$rcx','$r8','$r9'][i-1])) if i<=6 else read(stack+8*(i-6),1,'Q')[0]
        if read(p(8),1,'i')[0]==3:return False
        assert read(p(19),1,'i')[0]==1 and read(p(20),1,'i')[0]==1
        assert read(p(21),1,'i')[0]==len(state['previous'])
        write(p(4),state['previous']);write(p(3),state['previous']);write(p(6),state['strain'])
        result.update(previous=read(p(4),len(state['previous']),'d'),strain=read(p(6),6,'d'),elastic=read(p(1),2,'d'),icmd=read(p(8),1,'i')[0])
        Done(p);return False
Entry('*incplas_',internal=True);gdb.execute('run');assert finished;gdb.execute('kill')
Path(os.environ['THESIS_OUTPUT']).write_text(json.dumps(result,indent=2)+'\n')
