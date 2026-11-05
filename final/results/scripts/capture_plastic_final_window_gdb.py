"""GDB-only x86-64 SysV diagnostic capture; never attach to production processes.

Function entry arguments follow retained Fortran signatures. Array snapshots are
captured on function return, before the caller eliminates constrained DOFs.
This prototype deliberately does not establish nonlinear assembly identity.
"""
import gdb
import json
import os
import struct

records = []
stream_path = os.environ.get('THESIS_CAPTURE_JSONL')
stream = open(stream_path, 'x') if stream_path else None
record_count = 0
SIGNATURES = {
    'e_c3d_': dict(ff=11, nelem=12, nmethod=13, vold=29, xload=33, dtime=45, mi=47, ttime=54, time=55, istep=56, iinc=57, reltime=60, ipkon=73),
    'e_c3d_rhs_': dict(ff=10, nelem=11, nmethod=12, vold=16, xload=20, ttime=23, time=24, istep=25, iinc=26, dtime=27, reltime=29, mi=38),
}


def ptr(expr):
    return int(gdb.parse_and_eval(expr))

def read(address, count, code):
    size = struct.calcsize(code)
    return list(struct.unpack('<' + code * count,
                gdb.selected_inferior().read_memory(address, size * count).tobytes()))

returns = {}

class ReturnCapture(gdb.Breakpoint):
    def __init__(self, address):
        super().__init__("*" + hex(address), internal=True)
        self.invocations = {}
    def stop(self):
        key = (gdb.selected_thread().global_num, ptr('$rsp'))
        invocation = self.invocations.pop(key, None)
        if invocation is not None:
            routine, ff, connectivity, metadata = invocation
            global record_count
            record = dict(routine=routine, connectivity=connectivity, ff=read(ff, 60, 'd'), **metadata)
            record_count += 1
            if stream is None:
                records.append(record)
            else:
                stream.write(json.dumps(record)+'\n')
        return False

class EntryCapture(gdb.Breakpoint):
    def __init__(self, routine, ff_stack_offset, kon_register):
        super().__init__('*' + routine, internal=True)
        self.routine, self.offset, self.kon_register = routine, ff_stack_offset, kon_register
    def stop(self):
        stack = ptr('$rsp')
        ff = read(stack + self.offset, 1, 'Q')[0]
        args = {name: read(stack + 8*(index-6), 1, 'Q')[0]
                for name, index in SIGNATURES[self.routine].items()}
        metadata = {name: read(args[name], 1, 'i')[0]
                    for name in ('nelem', 'nmethod', 'istep', 'iinc')}
        metadata.update({name: read(args[name], 1, 'd')[0]
                         for name in ('time', 'ttime', 'dtime', 'reltime')})
        if metadata['time'] < 0.9875 - 1e-8:
            return False
        offset = read(args['ipkon'] + (metadata['nelem']-1)*4, 1, 'i')[0] if self.routine == 'e_c3d_' else 0
        assert offset >= 0
        connectivity = read(ptr(self.kon_register) + offset*4, 20, 'i')
        metadata['connectivity_offset'] = offset
        stride = read(args['mi'], 3, 'i')[1] + 1
        co = ptr('$rdi')
        metadata['coordinates'] = [read(co + (n-1)*24, 3, 'd') for n in connectivity]
        metadata['displacements_at_assembly'] = [read(args['vold'] + ((n-1)*stride+1)*8, 3, 'd') for n in connectivity]
        metadata['xload'] = read(args['xload'], 2, 'd')
        address = read(stack, 1, 'Q')[0]
        if address not in returns:
            returns[address] = ReturnCapture(address)
        key = (gdb.selected_thread().global_num, stack+8)
        assert key not in returns[address].invocations
        returns[address].invocations[key] = (self.routine, ff, connectivity, metadata)
        return False

EntryCapture('e_c3d_', 40, '$rsi')
EntryCapture('e_c3d_rhs_', 32, '$rdx')
gdb.execute('run')
if stream is not None:
    stream.close()
with open(os.environ['THESIS_CAPTURE_OUTPUT'], 'w') as stream:
    json.dump(dict(records=records, record_count=record_count, streamed_jsonl=stream_path, limitation='ABI prototype; no accepted-frame or assembly identity mapping'), stream, indent=2)
