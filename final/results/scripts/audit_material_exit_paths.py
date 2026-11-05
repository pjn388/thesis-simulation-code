"""Static exit-edge/call-selection prerequisite; does not run the solver."""
import json
import re
import subprocess
from pathlib import Path
from audit_plastic_return_map import F, sha256


def main():
    build = F/'sym/vendor/ccx_volume_material_debug_20261019'
    exe = build/'ccx_2.23_material'
    prior = F/'results/output/material_instruction_map_20261019.json'
    assert sha256(exe) == json.loads(prior.read_text())['sources'][str(exe.relative_to(F))]
    def dis(name):
        return subprocess.check_output(['objdump', '-d', '--no-show-raw-insn',
                                       '--disassemble='+name, str(exe)], text=True)
    raw = dis('incplas_')
    instructions = []
    for line in raw.splitlines():
        m = re.match(r'\s*([0-9a-f]+):\s+(\S+)(?:\s+(.*))?$', line)
        if m:
            instructions.append((int(m[1], 16), m[2], m[3] or ''))
    assert len(instructions) > 2000
    # Enumerate every direct branch and ordinary fallthrough into the three
    # exit blocks. This is edge coverage, not all-path or live-value proof.
    targets = (0x47a760, 0x478a00, 0x478a8a)
    incoming = {hex(t): [] for t in targets}
    indirect = []
    for k, (pc, op, args) in enumerate(instructions):
        if op.startswith('j'):
            m = re.match(r'([0-9a-f]+)\s', args)
            if not m:
                indirect.append(hex(pc))
            elif int(m[1], 16) in targets:
                incoming[hex(int(m[1], 16))].append(dict(pc=hex(pc), edge=op))
        if k+1 < len(instructions) and instructions[k+1][0] in targets:
            if op not in ('jmp', 'ret', 'retq', 'ud2'):
                incoming[hex(instructions[k+1][0])].append(dict(pc=hex(pc), edge='fallthrough'))
    assert incoming['0x47a760'] == [dict(pc='0x476561', edge='ja'), dict(pc='0x47656c', edge='ja'), dict(pc='0x47a75d', edge='fallthrough')]
    # Padding falls through syntactically but has no incoming branch and follows
    # an unconditional jump. Retain it explicitly rather than hide an edge.
    by_pc = {pc: (op, args) for pc, op, args in instructions}
    assert by_pc[0x47a75d][0] == 'nopl'
    assert by_pc[0x47a758][0] == 'jmp'
    assert not re.search(r'\bj\w*\s+47a75d\s', raw)
    assert incoming['0x478a00'] == [dict(pc='0x4781ad', edge='ja'), dict(pc='0x4781b8', edge='ja')]
    assert incoming['0x478a8a'] == [dict(pc='0x478a81', edge='fallthrough'), dict(pc='0x47a7e1', edge='jmp')]
    assert not indirect, 'Indirect control flow needs separate inspection'
    mech = dis('mechmodel_')
    results = dis('resultsmech_')
    assert re.search(r'3cfe7b:\s+call\s+475540 <incplas_>', mech)
    calls = [line.strip() for line in results.splitlines() if '<mechmodel_>' in line]
    assert calls
    header = subprocess.check_output(['readelf', '-h', str(exe)], text=True)
    assert 'DYN (Position-Independent Executable file)' in header
    source = (build/'resultsmech.f').read_text()
    assert 'time,ttime,i,jj,nstate_,mi(1)' in source
    source2 = (build/'mechmodel.f').read_text()
    assert 'time,ttime,iel,iint,nstate_,mi,eloc' in source2
    out = dict(
        state='static_exit_edges_passed_live_selection_blocked', native_runs_launched=0,
        matched_pair_authorized=False, pressure_authorized=False,
        incoming_exit_edges=incoming, indirect_jumps=indirect,
        unreachable_padding_edge='0x47a75d falls through syntactically; no direct incoming branch, preceding instruction 0x47a758 jumps away.',
        incplas_symbol_pc='0x475540',
        symbol_relative_capture_offsets={hex(t): hex(t-0x475540) for t in
                                        (0x47654c, 0x4781a0, 0x478a8a)},
        resultsmech_calls=calls, mechmodel_incplas_call='0x3cfe7b',
        finding='All direct incoming edges to the two scalar convergence exits and common update are enumerated. Source passes resultsmech i/jj through mechmodel iel/iint to incplas; binary confirms the finite-strain incplas call. Binary is PIE: earlier PCs are link-time addresses, not fixed runtime breakpoints.',
        required_live_selection='Resolve incplas_ in the inferior and add symbol-relative offsets; select the exact residual-evaluation time/cycle and resultsmech i/jj, confirm mechmodel caller return site by relocated symbol offset, then compare entry arguments, old history and captured geometry with the retained frame. Element/IP alone is insufficient because calls repeat across iterations and residual evaluations.',
        blocker='No inferior stopped at these sites exists in retained evidence. Static incoming edges and source argument forwarding do not demonstrate live frame selection, register/stack identity or path-specific residual operands. fu remains optimized out. The explicit prerequisite for a matched pair is unmet.',
        decision='Stop before native replay. Preserve global FD failure and existing baseline. Do not repeat static maps or launch the matched pair from link-time PCs. Further capture requires a concrete read-only call-selection/identity mechanism; no amplitude sweep, changed optimization, rollback or pressure successor.',
        limitations='Static direct-edge audit only; not complete feasible-path dataflow proof, native local branch identity, global derivative acceptance, convergence benefit or physical validation.',
        sources={str(p.relative_to(F)): sha256(p) for p in
                 (Path(__file__), exe, prior, build/'resultsmech.f', build/'mechmodel.f')})
    path = F/'results/output/material_exit_paths_20261019.json'
    assert not path.exists(), 'Preserve prior evidence'
    path.write_text(json.dumps(out, indent=2)+'\n')
    print(out['state'])


if __name__ == '__main__':
    main()
