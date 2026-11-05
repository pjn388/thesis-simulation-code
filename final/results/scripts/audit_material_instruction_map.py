"""Static instruction evidence only; never authorize live capture from this map."""
import json, subprocess
from pathlib import Path
from audit_plastic_return_map import F, sha256

exe=F/'sym/vendor/ccx_volume_material_debug_20261019/ccx_2.23_material'
prior=json.loads((F/'results/output/material_debug_alignment_20261019.json').read_text())
assert sha256(exe)==prior['sources'][str(exe.relative_to(F))]
regions=[('newton',0x476548,0x476589),('bisect',0x478176,0x4781c3),
         ('newton_exit',0x47a760,0x47a7e6),('update',0x478a8a,0x478ac1)]
raw={name:subprocess.check_output(['objdump','-d',f'--start-address={a}',f'--stop-address={b}',str(exe)],text=True) for name,a,b in regions}
for name,tokens in {
 'newton':['476548:','subsd  %xmm0,%xmm3','47655d:','476561:','47a760','47656c:'],
 'bisect':['478198:','subsd  %xmm6,%xmm4','4781a9:','4781ad:','478a00','4781b8:'],
 'newton_exit':['47a7e1:','478a8a'],
 'update':['478a8a:','mulsd  %xmm3,%xmm14','478aa5:','478ab2:']}.items():
    for token in tokens: assert token in raw[name],(name,token)
commands=['info line *0x47654c','info line *0x476559','info line *0x4781a9','info line *0x478a8a','x/gf 0x57f290']
gdb=subprocess.check_output(['gdb','--batch',str(exe)]+[v for c in commands for v in ('-ex',c)],text=True)
assert '0.0001' in gdb
out=dict(state='static_candidate_map_live_alignment_unproved',native_runs_launched=0,
 matched_pair_authorized=False,pressure_authorized=False,
 finding='The single info-line address 0x4764ce is not the unique post-loop update. Newton test and bisection test have distinct optimized exits which converge at 0x478a8a. Register dataflow gives candidate read-only capture sites; no native local identity demonstrated.',
 candidates=[
 dict(pc='0x47654c',meaning='Immediately after cop=cop-dcop; xmm3 is updated cop and xmm0 retains signed dcop until andpd at 0x476554. Source/DWARF agree on short live ranges. Newton comparisons at 0x47655d/0x476567 branch to 0x47a760 then 0x478a8a.'),
 dict(pc='0x4781a0',meaning='Bisection half-width is xmm0 before absolute-value mask; midpoint cop is xmm3. Comparisons at 0x4781a9/0x4781b3 branch through 0x478a00 to common update.'),
 dict(pc='0x478a8a',meaning='Common post-loop update consumes xmm3 as cop. xmm14*c​op then epini addition stored to ep at rsp+0x470. DCOP and fu availability cannot be inferred at this merged site.'.replace('c​op','cop'))],
 exact_blocker='All predecessor paths, relocated runtime addresses, element/IP/call selection and stack/register identity at live candidate sites have not been demonstrated. fu is optimized out; relevant branch residual operands require path-specific recovery. Static map alone cannot authorize matched pair.',
 next='Complete predecessor/path and resultsmech-to-incplas call-selection audit using this unchanged binary. Only if exact read-only capture can be demonstrated, perform the already bounded matched +/-1e-6 pair. If unavailable retain blocker. No changed optimization, amplitude sweep, rollback or pressure successor.',
 limitations='Read-only disassembly/source-line audit only. No local branch cause, global FD acceptance, native rollback, convergence benefit or physical validation.',
 disassembly=raw,source_lines=gdb,
 sources={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),exe,F/'results/output/material_debug_alignment_20261019.json')})
p=F/'results/output/material_instruction_map_20261019.json'
assert not p.exists()
p.write_text(json.dumps(out,indent=2)+'\n')
print(out['state'])
