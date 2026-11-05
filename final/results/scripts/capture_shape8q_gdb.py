"""Disposable native shape8q replay; stop before altered element assembly."""
import gdb,json,os,struct,re
from pathlib import Path
cycle=0;selected=False;inside=False;records=[]
source=Path(os.environ['THESIS_ELEMENT_SOURCE']).read_text()
names=re.sub(r'\s|&','',source.split('subroutine e_c3d(',1)[1].split(')',1)[0]).split(',')
def read(a,n,c='d'):
    return list(struct.unpack('<'+c*n,gdb.selected_inferior().read_memory(a,struct.calcsize(c)*n).tobytes()))
def args():
    stack=int(gdb.parse_and_eval('$rsp'))
    return lambda i:int(gdb.parse_and_eval(['$rdi','$rsi','$rdx','$rcx','$r8','$r9'][i-1])) if i<=6 else read(stack+8*(i-6),1,'Q')[0]
class Results(gdb.Breakpoint):
    def stop(self):
        global selected,cycle
        p=args()
        if read(p(36),1,'i')[0]!=-1:return False
        selected=cycle==2;cycle+=1
        return False
class Element(gdb.Breakpoint):
    def stop(self):
        global inside
        if selected:
            a=args();p={n:a(i+1) for i,n in enumerate(names)}
            assert read(p['nelem'],1,'i')==[1]
            assert read(p['mass'],1,'i')==[0] and read(p['rhsi'],1,'i')==[1]
            gdb.selected_inferior().write_memory(p['mass'],struct.pack('<i',1))
            gdb.selected_inferior().write_memory(p['rhsi'],struct.pack('<i',0))
            inside=True;self.enabled=False
        return False
class Done(gdb.FinishBreakpoint):
    def __init__(self,p,record):
        super().__init__(gdb.newest_frame(),internal=True);self.p=p;self.record=record
    def stop(self):
        self.record.update(shp=read(self.p[5],56),xs=read(self.p[4],21),area=read(self.p[3],3))
        records.append(self.record)
        return len(records)==4
class Shape(gdb.Breakpoint):
    def stop(self):
        if not inside:return False
        a=args();p=[a(i) for i in range(1,8)]
        if read(p[6],1,'i')[0]!=3:return False
        original=read(p[2],24);imposed=json.loads(os.environ['THESIS_SHAPE_COORDS'])
        if imposed is not None:gdb.selected_inferior().write_memory(p[2],struct.pack('<'+'d'*24,*imposed))
        record=dict(r=read(p[0],1)[0],s=read(p[1],1)[0],original_coordinates=original,coordinates=read(p[2],24),iflag=3)
        Done(p,record);return False
Results('*results',internal=True);Element('*e_c3d_',internal=True);Shape('*shape8q_',internal=True)
gdb.execute('run');assert len(records)==4
gdb.execute('kill');Path(os.environ['THESIS_OUTPUT']).write_text(json.dumps(records)+'\n')
