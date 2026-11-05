"""Separate output-only plastic-state capture; existing field capture unchanged."""
import gdb, json, os, struct
records=[]
stream=open(os.environ['THESIS_INTERNAL_BINARY'],'xb')
def read(address,count,code):
    return struct.unpack('<'+code*count,gdb.selected_inferior().read_memory(address,struct.calcsize(code)*count).tobytes())
def pointer(index):
    if index<=6:return int(gdb.parse_and_eval(['$rdi','$rsi','$rdx','$rcx','$r8','$r9'][index-1]))
    return read(int(gdb.parse_and_eval('$rsp'))+8*(index-6),1,'Q')[0]
class Frame(gdb.Breakpoint):
    def stop(self):
        mi=read(pointer(18),3,'i');nk=read(pointer(31),1,'i')[0];ne=read(pointer(32),1,'i')[0]
        assert mi[1]==3 and nk>0 and ne>0
        nstate=read(pointer(19),1,'i')[0];assert nstate>=13
        record=dict(time=read(pointer(24),1,'d')[0]+read(pointer(42),1,'d')[0],nodes=nk,elements=ne,ips=mi[0],states=nstate,fields={})
        for name,index,count in [('u',9,4*nk),('stress',14,6*mi[0]*ne),('strain',15,6*mi[0]*ne),('energy',17,2*mi[0]*ne),('state',16,nstate*mi[0]*ne)]:
            offset=stream.tell();stream.write(gdb.selected_inferior().read_memory(pointer(index),8*count).tobytes())
            record['fields'][name]=dict(offset=offset,count=count)
        stream.flush();records.append(record)
        return False
Frame('*printout_',internal=True)
gdb.execute('run');stream.close()
with open(os.environ['THESIS_INTERNAL_INDEX'],'x') as out:json.dump(dict(records=records,layout='little endian float64; node4, IP6 stress/strain, IP2 energy, IP nstate; Fortran contiguous',limitations='Output-only states do not expose intermediate material integration trajectory.'),out,indent=2)
