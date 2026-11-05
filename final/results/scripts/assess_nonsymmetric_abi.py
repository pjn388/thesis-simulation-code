"""Record source-interface assessment, without altering or validating a native binary."""
import json
from datetime import datetime, timezone
from audit_plastic_return_map import F, sha256


def main():
    src = F/'sym/vendor/CalculiX/ccx_2.23/src'
    checks = {
        'incplas.f': ['stiff(21)'],
        'resultsmech.f': ['xstiff(27,mi(1),*)', 'do m1=1,21', 'xstiff(m1,jj,i)=stiff(m1)'],
        'materialdata_me.f': ['stiff(j)=xstiff(j,iint,iel)'],
        'e_c3d.f': ['stiff(21)', 'do ii=1,jj', 'call anisonl'],
        'mafillsm.f': ['do ll=jj,ndof*nope', 'call add_sm_st('],
        'mafillsmas.f': ["if(lakon(i)(1:2).ne.'ES') cycle", 'call add_sm_st_as('],
        'mafillsmasmain.c': ['cpypardou(&au[nzs[2]],au'],
        'nonlingeo.c': ['NNEW(xstiff,double,(long long)27*mi[0]**ne)', 'RENEW(au,double,2*nzs[1])', 'symmetryflag=2;', 'inputformat=1;'],
        'add_sm_st_as.f': ['ioffset=nzs(3)', 'ad(i)=ad(i)+2.d0*value'],
        'pardiso.c': ['mtype=1;', 'au[j+*nzs3-1]'],
    }
    evidence = {}
    for name, snippets in checks.items():
        p = src/name
        lines = p.read_text().splitlines()
        matches = []
        for snippet in snippets:
            hits = [i+1 for i, line in enumerate(lines) if snippet in line]
            assert hits, (name, snippet)
            matches.append(dict(snippet=snippet, lines=hits))
        evidence[str(p.relative_to(F))] = dict(sha256=sha256(p), matches=matches)
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(),
        source_trace_passed=True, native_implementation_verified=False,
        decision='A nasym-only switch is insufficient. Preserve production ABI; assess a separate opt-in additive correction path before implementing a native revision.',
        findings=[
            '21 packed material entries survive resultsmech storage and materialdata_me retrieval; 27 allocated slots do not constitute a 36-entry general material tangent.',
            'e_c3d large-displacement loop computes one node-pair triangle; mafillsm scatters one local triangle. Widening incplas alone is insufficient.',
            'Existing mechanical mafillsmas only visits ES contact elements. Ordinary C3D20R material correction needs a separate volume-element contribution.',
            'mafillsmasmain copies the symmetric lower values into the upper half first. Insert additive full correction after this copy, not before it; otherwise the copy destroys the distinct upper entries.',
            'add_sm_st_as uses nzs(3) upper offset; its diagonal doubles values when local index arguments differ. A full correction scatter must use equal neutral local indices for each diagonal entry.',
            'PARDISO inputformat1 supports separate lower/upper values with shared graph and mtype1. Existing interface capacity is not proof that a revised matrix is correct.'
        ],
        proposed_scope='Isolated static C3D20R isotropic plastic no-MPC/no-contact/no-thermal experiment only. Add full exact-minus-native material correction to copied symmetric storage. Follower-load derivative remains separate and must not be double-counted.',
        gates=['First assess full Newton-direction retained assembly with actual native hardening table; localized two-element success does not establish global correction coverage.',
               'Before pressure solve, verify isolated native lower/upper scatter and matrix action against independent finite differences; require unchanged2e-7 gate and disabled-parent identity.',
               'Verify residual/state restoration and unchanged elastic/material outputs; no production promotion from coupons alone.'],
        next='One retained full Newton-direction assembly assessment under900s/4GiB service. It tests full-direction native reconstruction and correction magnitude, not pressure improvement or physical stability.',
        sources=evidence,
        limitations='Source inspection only; no compiled nonsymmetric solver, solver ABI execution, convergence benefit, stability or burst acceptance.')
    out=F/'results/output/nonsymmetric_abi_assessment_20261019.json'
    with out.open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    print(out)

if __name__=='__main__': main()
