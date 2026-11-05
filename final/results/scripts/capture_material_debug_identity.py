"""One bounded baseline capture; no perturbed replay until identity passes."""
import json
import capture_volume_material as capture

def main():
    F = capture.F
    coupons = F/'results/output/material_debug_coupons_20261019.json'
    assert json.loads(coupons.read_text())['checks_passed']
    # Debug information must also preserve enabled outputs against v5 itself.
    for label in ('material','elastic','resampled_material'):
        for variant in ('disabled','enabled'):
            name = label+'_'+variant+'/job.dat'
            assert (F/'sym/runs/investigation/material_debug_coupons_20261019'/name).read_bytes() == (F/'sym/runs/investigation/volume_material_integration_v5_20261019'/name).read_bytes()
    capture.BUILD = F/'sym/vendor/ccx_volume_material_debug_20261019'
    capture.ROOT = F/'sym/runs/investigation/material_debug_identity_20261019'
    capture.OUT = F/'results/output/material_debug_identity_20261019.json'
    capture.main()
    out = json.loads(capture.OUT.read_text())
    try:
        actual = json.loads((capture.ROOT/'assembly.json').read_text())
        reference = json.loads((F/'sym/runs/investigation/volume_material_capture_v5_20261019/assembly.json').read_text())
        keys = ('time','cycle','previous_state','previous_stress','displacement_before','displacement_after','residual')
        identities = {key: actual[key] == reference[key] for key in keys}
        out.update(retained_identity=identities,debug_v5_coupon_byte_identity=True,
                   checks_passed=all(identities.values()),
                   next='If identities pass, establish exact source-call/local availability then capture only matched +/-1e-6 pair. No amplitude sweep or pressure successor.')
        assert out['checks_passed'], 'Debug trajectory differs from retained v5'
    except Exception as exc:
        out.update(state='failed',checks_passed=False,error=str(exc)); raise
    finally:
        out['sources'][str(coupons.relative_to(F))] = capture.sha256(coupons)
        out['sources']['results/scripts/capture_material_debug_identity.py'] = capture.sha256(__file__)
        capture.OUT.write_text(json.dumps(out,indent=2)+'\n')

if __name__ == '__main__':
    main()
