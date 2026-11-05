import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_elastic_restraint_fields import lame_components


def test_lame_surface_tractions_and_closed_end_force():
    inner, outer, pressure = 46., 50., 1.
    xyz = np.array([[inner, 0, 0], [0, outer, 0]])
    radial, hoop, axial = lame_components(xyz, pressure, outer, inner)
    np.testing.assert_allclose(radial, [-pressure, 0], atol=1e-14)
    np.testing.assert_allclose(axial * np.pi * (outer**2-inner**2), pressure*np.pi*inner**2)
    np.testing.assert_allclose(hoop[0]-radial[0], 2*pressure*outer**2/(outer**2-inner**2))
