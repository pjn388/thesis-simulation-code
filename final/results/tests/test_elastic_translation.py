import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_elastic_translation import decompose


def test_constant_translation_removed_without_fitting_expansion():
    angles = np.arange(16)*2*np.pi/16
    xyz = np.column_stack([50*np.cos(angles), 50*np.sin(angles), np.zeros(16)])
    expected = xyz*.001
    shift = np.array([.03, -.04, .02])
    mask = np.ones(16, dtype=bool)
    result = decompose(xyz, expected+shift, expected, mask)
    np.testing.assert_allclose(result['translation_mm'], shift)
    assert result['radial_errors']['raw']['mean'] > .5
    assert result['radial_errors']['translation_removed']['maximum'] < 1e-14
    # A real expansion discrepancy must survive the diagnostic.
    result = decompose(xyz, expected*1.1+shift, expected, mask)
    np.testing.assert_allclose(result['translation_mm'], shift)
    np.testing.assert_allclose(result['radial_errors']['translation_removed']['mean'], .1)
