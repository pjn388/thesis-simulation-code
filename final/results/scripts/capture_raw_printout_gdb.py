"""Serial x86-64 SysV output-only capture; isolated diagnostics, never attach.

Argument indices follow printout.f and printoutelem.f. Energy is the change
in the cumulative set total on element return, preserving native integration.
"""
import gdb
import json
import os
import struct

records = []
current = None

def read(address, count, code):
    return list(struct.unpack('<'+code*count, gdb.selected_inferior().read_memory(address, struct.calcsize(code)*count).tobytes()))

def pointer(index):
    if index <= 6:
        return int(gdb.parse_and_eval(['$rdi','$rsi','$rdx','$rcx','$r8','$r9'][index-1]))
    return read(int(gdb.parse_and_eval('$rsp'))+8*(index-6),1,'Q')[0]

class Frame(gdb.Breakpoint):
    def stop(self):
        global current
        stride = read(pointer(18),3,'i')[1]+1
        nodes = read(pointer(31),1,'i')[0]
        elements = read(pointer(32),1,'i')[0]
        assert stride == 4 and nodes > 0 and elements > 0
        current = dict(time=read(pointer(24),1,'d')[0]+read(pointer(42),1,'d')[0],
            nodes=nodes, elements=elements,
            u=[read(pointer(9)+8*(n*stride+1),3,'d') for n in range(nodes)],
            energy=[])
        records.append(current)
        if endpoint_only and current["time"] >= 1.-1e-12:
            element_breakpoint.enabled = True
        return False

class EnergyReturn(gdb.FinishBreakpoint):
    def __init__(self, address, before, element, frame):
        super().__init__(internal=True)
        self.address,self.before,self.element,self.record=address,before,element,frame
    def stop(self):
        self.record['energy'].append([self.element,read(self.address,1,'d')[0]-self.before])
        return False

class Element(gdb.Breakpoint):
    def stop(self):
        ii = read(pointer(8),1,'i')[0]
        label = gdb.selected_inferior().read_memory(pointer(1)+6*(ii-1),6).tobytes()
        if label[:4] == b'ELSE':
            assert current is not None
            address=pointer(10)
            EnergyReturn(address,read(address,1,'d')[0],read(pointer(9),1,'i')[0],current)
        return False

Frame('*printout_',internal=True)
endpoint_only = os.environ.get('THESIS_RAW_ENDPOINT_ONLY') == '1'
element_breakpoint = Element('*printoutelem_',internal=True)
element_breakpoint.enabled = not endpoint_only
gdb.execute('run')
with open(os.environ['THESIS_RAW_OUTPUT'],'x') as stream:
    json.dump(dict(records=records, limitation='Serial ABI output capture; accepted times and literal rounding must be verified separately. Energy by cumulative difference can suffer cancellation for large sets.'),stream,indent=2)
