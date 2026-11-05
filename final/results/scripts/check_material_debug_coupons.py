"""Reuse unchanged nine-case integration gates for isolated debug executable."""
import json
from pathlib import Path
import check_volume_material_integration as checks

def main():
    checks.BUILD = checks.F/'sym/vendor/ccx_volume_material_debug_20261019'
    build = json.loads((checks.BUILD/'build.json').read_text())
    assert build['state'] == 'built_code_identity_passed'
    assert all(row['text_byte_identity'] for row in build['objects'])
    checks.ROOT = checks.F/'sym/runs/investigation/material_debug_coupons_20261019'
    checks.OUT = checks.F/'results/output/material_debug_coupons_20261019.json'
    checks.main()

if __name__ == '__main__':
    main()
