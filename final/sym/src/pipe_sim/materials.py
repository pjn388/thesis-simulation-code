"""Explicit material conversions; never invent reference material coefficients."""
import numpy as np


def engineering_to_plastic(stress, strain, young):
    """Uniform uniaxial deformation only; NOT a post-necking correction."""
    stress, strain = np.asarray(stress, float), np.asarray(strain, float)
    true_stress = stress * (1 + strain)
    plastic_strain = np.log1p(strain) - true_stress / young
    return np.column_stack((true_stress, plastic_strain))


def ramberg_osgood(stress, young, yield_stress, alpha, exponent, convention="normalised"):
    """Return true stress / plastic strain; caller must handle the initial yield point.

    normalised: eps/(sy/E) = s/sy + alpha*(s/sy)**n
    offset:     eps = s/E + alpha*(s/sy)**n
    These coefficients are NOT interchangeable. Neither is fitted automatically.
    """
    stress = np.asarray(stress, float)
    if min(young, yield_stress, alpha, exponent) <= 0:
        raise ValueError("Material coefficients must be positive")
    if convention not in ("normalised", "offset"):
        raise ValueError("Specify normalised or offset convention")
    factor = yield_stress / young if convention == "normalised" else 1.0
    return np.column_stack((stress, factor * alpha * (stress / yield_stress)**exponent))


def total_curve_to_plastic(stress, strain, young, convention='true_total'):
    """Preserve an ordered monotonic curve after its intersection with sigma=E*eps.

    No independent table yield is spliced into the curve. The first plastic point
    is the interpolated elastic intersection, followed by the measured samples.
    Engineering conversion is valid only before necking; the caller must establish
    applicability and document digitization uncertainty. No tail is added here.
    """
    stress=np.asarray(stress,dtype=float);strain=np.asarray(strain,dtype=float)
    if len(stress)!=len(strain) or len(stress)<2 or young<=0:
        raise ValueError('Need a stress/strain curve and positive modulus')
    if not np.isfinite(stress).all() or not np.isfinite(strain).all():
        raise ValueError('Curve values must be finite')
    if convention=='engineering_total':
        if (strain<=-1).any():raise ValueError('Invalid engineering strain')
        stress,strain=stress*(1+strain),np.log1p(strain)
    elif convention!='true_total':raise ValueError('Specify true_total or engineering_total')
    if (np.diff(strain)<=0).any() or (np.diff(stress)<0).any():
        raise ValueError('Curve must have strictly increasing strain and nondecreasing stress')
    plastic=strain-stress/young
    crossing=np.flatnonzero((plastic[:-1]<=0)&(plastic[1:]>0))
    if len(crossing)!=1:raise ValueError('Need one resolved elastic intersection; supply an explicit material model otherwise')
    i=int(crossing[0]);weight=-plastic[i]/(plastic[i+1]-plastic[i])
    yield_stress=float(stress[i]+weight*(stress[i+1]-stress[i]))
    result=[[yield_stress,0.]]
    for s,e in zip(stress[i+1:],plastic[i+1:]):
        if e<=result[-1][1]:raise ValueError('Non-increasing plastic strain after elastic intersection')
        result.append([float(s),float(e)])
    return np.asarray(result)
