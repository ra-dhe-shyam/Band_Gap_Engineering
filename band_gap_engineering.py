#!/usr/bin/env python3
"""Band-Gap Engineering Explorer: interactive local Python teaching model.

Install: python3 -m pip install numpy scipy matplotlib
Run:     python3 band_gap_engineering.py

References and limits are listed in README.md and below. Numerical results are
trend-level estimates, not calibrated predictions for a specific sample.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss

KB = 8.617333262e-5       # eV/K
HBAR2_2M0 = 0.0380998212  # eV nm^2
# Representative Varshni parameters: (Eg(0) eV, alpha eV/K, beta K).
# Diamond is a special case handled by a Bose-Einstein-type temperature fit in
# varshni(); its tuple below is only a fallback reference, not the active model.
# Sources and limitations are documented in README.md.
VARSHNI = {
    "Si": (1.170, 4.73e-4, 636.0),
    "Ge": (0.744, 4.774e-4, 235.0),
    "GaAs": (1.519, 5.405e-4, 204.0),
    "InAs": (0.417, 2.76e-4, 93.0),
    "InP": (1.4236, 4.9e-4, 327.0),
    "InSb": (0.235, 3.20e-4, 170.0),
    "GaSb": (0.812, 4.17e-4, 140.0),
    "GaN": (3.507, 9.09e-4, 830.0),
    "AlN": (6.230, 1.799e-3, 1462.0),
    "4H-SiC": (3.285, 3.30e-4, 1300.0),
    # Diamond is handled separately below; do not use a constant-gap entry here.
    "Diamond": (5.480, 0.0, 204.0),
}

# Approximate scalar near-edge electron and hole masses in units of m0.
# Actual masses can be anisotropic and valley-dependent.
MASSES = {
    "Si": (0.26, 0.49), "Ge": (0.12, 0.29), "GaAs": (0.067, 0.50),
    "InAs": (0.023, 0.41), "InP": (0.079, 0.60), "InSb": (0.0135, 0.43),
    "GaSb": (0.039, 0.40), "GaN": (0.20, 1.00), "AlN": (0.30, 1.50),
    "4H-SiC": (0.29, 0.66), "Diamond": (0.36, 0.80),
}

# Representative hydrostatic deformation potentials (ac, av), in eV.
# Simplified scalar values; anisotropic/wurtzite strain physics is not resolved.
DEFORMATION = {
    "Si": (-8.60, 2.10), "Ge": (-8.97, 2.00), "GaAs": (-7.17, 1.16),
    "InAs": (-5.08, 1.00), "InP": (-6.00, 0.60), "InSb": (-6.94, 0.36),
    "GaSb": (-7.50, 0.80), "GaN": (-7.21, 1.00), "AlN": (-11.80, 1.50),
    "4H-SiC": (-8.50, 1.20), "Diamond": (-8.00, 1.00),
}

# Approximate 300 K Nc, Nv [cm^-3], donor binding [eV], dielectric constant.
# Composition-, sample-, and polytype-dependent values vary in the literature.
FERMI_DATA = {
    "Si": (2.80e19, 1.04e19, 0.045, 11.7),
    "Ge": (1.04e19, 6.00e18, 0.012, 16.0),
    "GaAs": (4.70e17, 7.00e18, 0.006, 12.9),
    "AlAs": (2.20e18, 1.80e19, 0.030, 10.1),
    "InAs": (8.60e16, 6.60e18, 0.001, 15.1),
    "InP": (5.70e17, 1.10e19, 0.006, 12.4),
    "InSb": (5.80e16, 7.30e18, 0.001, 16.8),
    "GaSb": (4.70e17, 1.40e19, 0.003, 15.7),
    "GaN": (2.30e18, 4.60e19, 0.030, 9.5),
    "AlN": (6.30e18, 4.80e19, 0.060, 8.5),
    "4H-SiC": (1.60e19, 2.50e19, 0.080, 9.7),
    "Diamond": (1.00e20, 1.00e20, 0.570, 5.7),
}

MATERIALS_MENU = (
    "Si", "Ge", "GaAs", "AlGaAs", "InAs", "InP", "InSb", "GaSb",
    "InGaAs", "GaN", "AlN", "AlGaN", "4H-SiC", "Diamond"
)
MATERIAL_TYPE = {
    "Si": "indirect", "Ge": "indirect", "GaAs": "direct",
    "AlGaAs": "composition-dependent Γ/X crossover",
    "InAs": "direct", "InP": "direct", "InSb": "direct", "GaSb": "direct",
    "InGaAs": "direct", "GaN": "direct", "AlN": "direct",
    "AlGaN": "direct (approximate)", "4H-SiC": "indirect", "Diamond": "indirect"
}


def _parse_material_name(material):
    """Support legacy labels such as 'AlGaAs(0.3)' and plain material names."""
    if "(" in material and material.endswith(")"):
        name, value = material.split("(", 1)
        return name, float(value[:-1])
    return material, None


def varshni(material, T, x=0.30):
    """Temperature-dependent gap for binaries and approximate alloy equations.

    Diamond special case:
      Eg(T) = Eg(0) - a*Theta / (exp(Theta/T) - 1)
      Eg(0) = 5.480 eV, a = 0.490 meV/K, Theta = 1308 K.

    This Bose-Einstein-type temperature law uses the temperature dependence
    fitted by Passler to older diamond exciton-gap data, with the exciton
    binding-energy offset added to estimate the fundamental indirect band gap.
    The low-temperature anchor Eg(0)=5.480 +/- 0.004 eV is consistent with
    high-resolution electronic-grade diamond data measured over 10-620 K.
    Treat this combined fit as approximate; above 620 K it is extrapolated.
    """
    material, named_x = _parse_material_name(material)
    if named_x is not None:
        x = named_x
    T = float(T)

    if material == "Diamond":
        eg0 = 5.480                 # eV; low-T fundamental indirect gap
        a = 0.490e-3                # eV/K; Passler fit parameter
        theta = 1308.0              # K; characteristic temperature
        if T <= 0.0:
            return eg0
        thermal_shift = (a*theta) / np.expm1(theta/T)
        return eg0 - thermal_shift

    if material in VARSHNI:
        eg0, alpha, beta = VARSHNI[material]
        return eg0 - alpha*T*T/(T+beta)
    if material == "AlGaAs":
        x = float(np.clip(x, 0.0, 1.0))
        eg300 = min(1.424 + 1.247*x, 1.900 + 0.125*x + 0.143*x*x)
        gaas300 = varshni("GaAs", 300.0)
        alas = lambda temp: 3.099 - 5.8e-4*temp*temp/(temp+408.0)
        temp_shift = (1-x)*(varshni("GaAs", T)-gaas300) + x*(alas(T)-alas(300.0))
        return eg300 + temp_shift
    if material == "InGaAs":
        # x is the indium fraction; representative bowing parameter is 0.477 eV.
        x = float(np.clip(x, 0.0, 1.0))
        return (1-x)*varshni("GaAs", T) + x*varshni("InAs", T) - 0.477*x*(1-x)
    if material == "AlGaN":
        # x is the aluminium fraction; representative bowing parameter is 0.55 eV.
        x = float(np.clip(x, 0.0, 1.0))
        return (1-x)*varshni("GaN", T) + x*varshni("AlN", T) - 0.55*x*(1-x)
    raise ValueError(f"No band-gap parameters available for {material!r}")


def masses(material, x):
    """Approximate scalar electron and hole effective masses, in units of m0."""
    if material in MASSES:
        return MASSES[material]
    if material == "AlGaAs":
        return ((1-x)*MASSES["GaAs"][0] + x*0.15,
                (1-x)*MASSES["GaAs"][1] + x*0.76)
    if material == "InGaAs":
        return ((1-x)*MASSES["GaAs"][0] + x*MASSES["InAs"][0],
                (1-x)*MASSES["GaAs"][1] + x*MASSES["InAs"][1])
    if material == "AlGaN":
        return ((1-x)*MASSES["GaN"][0] + x*MASSES["AlN"][0],
                (1-x)*MASSES["GaN"][1] + x*MASSES["AlN"][1])
    raise ValueError(f"No effective-mass parameters available for {material!r}")


def _fermi_data(material, x):
    """Return Nc300, Nv300 [cm^-3], donor binding [eV], relative permittivity."""
    if material in FERMI_DATA:
        return FERMI_DATA[material]
    if material == "AlGaAs":
        a, b = FERMI_DATA["GaAs"], FERMI_DATA["AlAs"]
    elif material == "InGaAs":
        a, b = FERMI_DATA["GaAs"], FERMI_DATA["InAs"]
    elif material == "AlGaN":
        a, b = FERMI_DATA["GaN"], FERMI_DATA["AlN"]
    else:
        raise ValueError(f"No carrier-statistics parameters available for {material!r}")
    return tuple((1-x)*a[i] + x*b[i] for i in range(4))


def strain_shift(material, x, hydrostatic_strain):
    """Approximate hydrostatic gap shift using scalar deformation potentials."""
    if material in DEFORMATION:
        ac, av = DEFORMATION[material]
    elif material == "AlGaAs":
        gaas, alas = DEFORMATION["GaAs"], (-5.64, 2.47)
        ac, av = (1-x)*gaas[0]+x*alas[0], (1-x)*gaas[1]+x*alas[1]
    elif material == "InGaAs":
        gaas, inas = DEFORMATION["GaAs"], DEFORMATION["InAs"]
        ac, av = (1-x)*gaas[0]+x*inas[0], (1-x)*gaas[1]+x*inas[1]
    elif material == "AlGaN":
        gan, aln = DEFORMATION["GaN"], DEFORMATION["AlN"]
        ac, av = (1-x)*gan[0]+x*aln[0], (1-x)*gan[1]+x*aln[1]
    else:
        return 0.0
    return (ac-av)*hydrostatic_strain


def bgn(doping):
    """Illustrative, capped high-doping narrowing; not a universal material law."""
    return min(0.012*np.sqrt(max(doping, 0.0)/1e18), 0.15)


# Gauss-Legendre nodes used for the normalized Fermi-Dirac integral F_(1/2).
_GL_NODES, _GL_WEIGHTS = leggauss(160)


def fermi_half(eta):
    """Normalized F_(1/2)(eta), with stable non-degenerate and degenerate limits."""
    eta = float(eta)
    if eta < -10.0:
        return float(np.exp(max(eta, -745.0)))
    if eta > 20.0:
        # Sommerfeld expansion for large positive eta.
        # F_1/2(eta) = 4/(3 sqrt(pi))*eta^(3/2) *
        # [1 + pi^2/(8 eta^2) + 7 pi^4/(640 eta^4) + ...]
        correction = (1.0 + np.pi**2/(8.0*eta**2)
                      + 7.0*np.pi**4/(640.0*eta**4))
        return float(4.0/(3.0*np.sqrt(np.pi))*eta**1.5*correction)

    # Normalized integral: 2/sqrt(pi) * integral sqrt(x)/(1+exp(x-eta)) dx.
    xmax = max(50.0, eta + 40.0)
    xx = 0.5*xmax*(_GL_NODES + 1.0)
    ww = 0.5*xmax*_GL_WEIGHTS
    z = np.clip(xx - eta, -700.0, 700.0)
    return float((2.0/np.sqrt(np.pi)) *
                 np.sum(ww*np.sqrt(xx)/(1.0 + np.exp(z))))


def fermi_estimate(eg, T, doping, material, x):
    """
    Approximate n-type Fermi level relative to the valence-band maximum (eV).

    Two clearly separated regimes are used:
      1) Below the estimated Mott density: donor freeze-out, with one isolated
         donor level and charge neutrality.
      2) At/above the estimated Mott density: donors are treated as fully
         ionised because isolated-donor physics is no longer appropriate.

    The Mott transition is not sharp in real materials. This piecewise switch
    is a transparent approximation, not a calibrated transition model.
    Assumptions: uncompensated n-type material, one donor species, effective-
    mass density of states, and representative donor binding energies.
    Returns (EF in eV from Ev, rough intrinsic concentration, regime label).
    """
    T = max(float(T), 0.5)
    doping = max(float(doping), 0.0)
    me, mh = masses(material, x)

    nc300, nv300, donor_binding, epsr = _fermi_data(material, x)
    # Use the material/alloy DOS values directly; do not apply an additional
    # effective-mass correction that would double-count composition dependence.
    nc = nc300*(T/300.0)**1.5
    nv = nv300*(T/300.0)**1.5

    kT = KB*T
    ec = float(eg)
    ed = ec - donor_binding
    donor_degeneracy = 2.0

    # Mott criterion estimate: n_Mott^(1/3) * a*_B ≈ 0.25.
    a_star_nm = 0.0529177*epsr/max(me, 0.01)
    mott_density = (0.25/a_star_nm)**3 * 1.0e21  # cm^-3

    def carrier_densities(ef):
        eta_n = (ef-ec)/kT
        n = nc*fermi_half(eta_n)
        # Non-degenerate hole approximation, reasonable for this n-type model.
        p = nv*np.exp(np.clip(-ef/kT, -700.0, 100.0))
        return n, p

    if doping <= 0:
        # Intrinsic, non-degenerate expression avoids numerical underflow as T -> 0:
        # EF = (EC+EV)/2 + (kT/2)*ln(NV/NC), with EV=0 and EC=Eg.
        # At T=0 this tends to midgap, as expected for an ideal intrinsic semiconductor.
        ef = 0.5*eg + 0.5*kT*np.log(max(nv, 1e-300)/max(nc, 1e-300))
        regime = "intrinsic (non-degenerate)"
    elif doping < mott_density:
        # Isolated donor levels: incomplete ionisation / freeze-out.
        def charge_balance(ef):
            n, p = carrier_densities(ef)
            exponent = np.clip((ef-ed)/kT, -700.0, 700.0)
            donor_ionised = doping/(1.0 + donor_degeneracy*np.exp(exponent))
            return n - p - donor_ionised
        ef = brentq(charge_balance, -2.0, eg+2.0)
        regime = "donor freeze-out approximation"
    else:
        # Above the estimated Mott density, use a fully ionised donor continuum.
        # Do not mix a fixed fraction of this branch into the freeze-out regime.
        ef = brentq(lambda e: carrier_densities(e)[0] -
                    carrier_densities(e)[1] - doping, -2.0, eg+2.0)
        regime = "high-density full-ionisation approximation"

    ni = float(np.sqrt(nc*nv)*np.exp(np.clip(-eg/(2*kT), -700.0, 0.0)))
    return float(ef), ni, regime


def well_levels(width_nm, depth_eV, mass_ratio, finite=True, count=4):
    """1D finite-difference square well; energies measured from well bottom (eV).

    A single constant effective mass is used inside and outside the well.
    This simple solver does not implement the position-dependent-mass
    BenDaniel-Duke operator.
    """
    if width_nm <= 0.1 or depth_eV <= 0: return np.array([])
    dx = 0.05
    half = max(8.0, 2*width_nm)
    xx = np.arange(-half, half+dx/2, dx)
    inside = np.abs(xx) <= width_nm/2
    barrier = depth_eV if finite else 100.0
    potential = np.where(inside, 0.0, barrier)
    c = HBAR2_2M0/(max(mass_ratio, 0.01)*dx*dx)
    diagonal = 2*c+potential
    off_diagonal = np.full(len(xx)-1, -c)
    vals = eigh_tridiagonal(diagonal, off_diagonal, select="i", select_range=(0, min(count+1, len(xx)-1)))[0]
    return vals[vals < barrier-1e-5][:count]


def main():
    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(left=.24, bottom=.36, hspace=.38, wspace=.28)
    ax_gap, ax_band, ax_levels, ax_well = axs.flat
    ax_mat = plt.axes([.025, .51, .17, .46])
    mat_radio = RadioButtons(ax_mat, MATERIALS_MENU, active=3)
    for _txt in mat_radio.labels: _txt.set_fontsize(8)
    ax_mat.set_title("Material", fontsize=10)
    ax_mode = plt.axes([.025, .35, .16, .12])
    mode_radio = RadioButtons(ax_mode, ("Finite well", "Infinite well"), active=0)
    ax_mode.set_title("Quantum well", fontsize=9)
    for _txt in mode_radio.labels: _txt.set_fontsize(8)
    sliders = []
    def add_slider(y, label, lo, hi, start, step):
        ax = plt.axes([.25, y, .65, .025])
        s = Slider(ax, label, lo, hi, valinit=start, valstep=step)
        sliders.append(s)
        return s
    sT = add_slider(.25, "Temperature (K)", 4, 800, 300, 1)
    sx = add_slider(.21, "Alloy fraction x", 0, 1, .30, .01)
    sw = add_slider(.17, "Well width (nm)", 2, 30, 8, .5)
    ss = add_slider(.13, "Hydrostatic strain (%)", -1, 1, 0, .05)
    sd = add_slider(.09, "Donor density N_D (cm⁻³; zero + logarithmic)", 0, 100, 80, 1)
    sv = add_slider(.05, "Illustrative well depth (eV)", .05, 1.5, .5, .05)
    status = fig.text(.24, .305, "", fontsize=8.0)

    def update(_=None):
        material, T, x = mat_radio.value_selected, sT.val, sx.val
        CURRENT_X = x
        if material == "AlGaAs":
            label = f"Al$_{{{x:.2f}}}$Ga$_{{{1-x:.2f}}}$As"
        elif material == "InGaAs":
            label = f"In$_{{{x:.2f}}}$Ga$_{{{1-x:.2f}}}$As"
        elif material == "AlGaN":
            label = f"Al$_{{{x:.2f}}}$Ga$_{{{1-x:.2f}}}$N"
        else:
            label = material
        eg_bulk = varshni(material, T, x)
        strain = strain_shift(material, x, ss.val/100)
        # Slider position 0 represents intrinsic material (N_D = 0).
        # Positions 1..100 map logarithmically from 10^0 to 10^20 cm^-3.
        doping = 0.0 if sd.val < 0.5 else 10.0**(20.0*(sd.val-1.0)/99.0)
        sd.valtext.set_text("0 (intrinsic)" if doping == 0 else f"{doping:.2e}")
        narrowing = bgn(doping)
        eg = max(.05, eg_bulk+strain-narrowing)
        me, mh = masses(material, x)
        ef, ni, fermi_regime = fermi_estimate(eg, T, doping, material, x)
        finite = mode_radio.value_selected == "Finite well"
        depth, width = sv.val, sw.val

        ax_gap.clear()
        ts = np.linspace(4, 800, 180)
        ax_gap.plot(ts, [varshni(material, tt, x) for tt in ts])
        ax_gap.scatter([T], [eg_bulk], zorder=3)
        ax_gap.axvline(T, ls="--", lw=1)
        ax_gap.set(title="Bulk gap vs temperature", xlabel="Temperature (K)", ylabel="Band gap (eV)")
        ax_gap.grid(alpha=.25)

        # Stable band-edge plot: energy is measured from the midgap, and both
        # axes use fixed limits so changing parameters moves the bands, not the axes.
        ax_band.clear()
        dk = np.linspace(-1.1, 1.1, 401)  # local wave-vector offset from each extremum
        ec_midgap = +0.5*eg + HBAR2_2M0/me*dk*dk
        ev_midgap = -0.5*eg - HBAR2_2M0/mh*dk*dk
        ef_midgap = ef - 0.5*eg  # ef is returned relative to the valence-band maximum
        ax_band.plot(dk, ec_midgap, label=r"Conduction edge $E_C$")
        ax_band.plot(dk, ev_midgap, label=r"Valence edge $E_V$")
        ax_band.axhline(ef_midgap, ls="--", lw=1.5, label=r"Fermi level $E_F$")
        ax_band.axhline(0.0, ls=":", lw=1.0, label="Midgap reference")
        ax_band.axvline(0.0, ls=":", lw=1.0, label=r"Band extrema ($\Delta k=0$)")
        # Mark the fundamental gap at the band extrema.
        ax_band.annotate("", xy=(-0.16, +0.5*eg), xytext=(-0.16, -0.5*eg),
                         arrowprops={"arrowstyle": "<->", "linewidth": 1.2})
        ax_band.text(-0.13, 0.0, rf"$E_g={eg:.3f}$ eV", fontsize=8,
                     ha="left", va="center",
                     bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.5})
        ax_band.set_title(f"Parabolic band-edge sketch: {label} ({MATERIAL_TYPE[material]})", fontsize=9.5)
        ax_band.set_xlabel(r"Wave-vector offset from each band extremum, $\Delta k$ (nm$^{-1}$)")
        ax_band.set_ylabel(r"Energy relative to midgap (eV)")
        ax_band.set_xlim(-1.1, 1.1)
        ax_band.set_ylim(-3.6, 3.6)
        ax_band.set_xticks(np.arange(-1.0, 1.01, 0.5))
        ax_band.set_yticks(np.arange(-3.0, 3.01, 1.0))
        ax_band.grid(alpha=.22)
        # Direct labels avoid a legend box hiding the parabolas.
        k_label = 0.67
        ec_label_y = 0.5*eg + HBAR2_2M0/me*k_label*k_label
        ev_label_y = -0.5*eg - HBAR2_2M0/mh*k_label*k_label
        ax_band.annotate("Conduction band", (k_label, ec_label_y),
                         xytext=(6, 4), textcoords="offset points", fontsize=7.2)
        ax_band.annotate("Valence band", (k_label, ev_label_y),
                         xytext=(6, -12), textcoords="offset points", fontsize=7.2)
        ax_band.annotate("Fermi level", (-0.98, ef_midgap),
                         xytext=(0, 5), textcoords="offset points", fontsize=7.2)
        ax_band.text(-0.98, 0.08, "Midgap", fontsize=7.0, va="bottom")
        ax_band.text(.02, .02,
                     "Local scalar-mass sketch only; not a full E(k) or valley model.",
                     transform=ax_band.transAxes, fontsize=7.2, va="bottom")

        el = well_levels(width, depth, me, finite)
        hl = well_levels(width, depth, mh, finite)
        ax_levels.clear()
        if len(el):
            ax_levels.scatter(el, np.arange(1, len(el)+1), label="Electron levels")
            for i, v in enumerate(el, 1): ax_levels.annotate(f"{v:.3f}", (v, i), xytext=(3, 3), textcoords="offset points", fontsize=8)
        if len(hl):
            ax_levels.scatter(hl, -np.arange(1, len(hl)+1), marker="s", label="Hole levels")
            for i, v in enumerate(hl, 1): ax_levels.annotate(f"{v:.3f}", (v, -i), xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax_levels.set(title=f"{'Finite' if finite else 'Infinite'} well energy levels", xlabel="Energy above well edge (eV)", ylabel="Level index (+ electron, − hole)")
        ax_levels.text(.02, .02, "Well depth is illustrative, not a calculated band offset", transform=ax_levels.transAxes, fontsize=8)
        ax_levels.grid(alpha=.25); ax_levels.legend(fontsize=8)

        ax_well.clear()
        pos = np.linspace(-15, 15, 400)
        inside = np.abs(pos) <= width/2
        barrier = depth if finite else 1.2
        ax_well.plot(pos, np.where(inside, eg, eg+barrier), label="Conduction edge")
        ax_well.plot(pos, np.where(inside, 0, -barrier), label="Valence edge")
        ax_well.axhline(ef, ls="--", label="Fermi level (charge neutrality)")
        for v in el:
            if v < barrier: ax_well.hlines(eg+v, -width/2, width/2, ls=":")
        for v in hl:
            if v < barrier: ax_well.hlines(-v, -width/2, width/2, ls=":")
        ax_well.set(title="Schematic quantum-well diagram", xlabel="Position (nm)", ylabel="Energy (eV; well valence edge = 0)")
        ax_well.grid(alpha=.25); ax_well.legend(fontsize=8)
        diamond_note = " | diamond fit extrapolated above 620 K" if material == "Diamond" and T > 620 else ""
        status.set_text(f"{label} | Eg={eg:.3f} eV | EF={ef:.3f} eV from Ev | N_D={doping:.1e} cm⁻³ | strain shift={strain:+.3f} eV | BGN={narrowing:.3f} eV | me*={me:.3f} m0 | mh*={mh:.3f} m0{diamond_note}")
        fig.canvas.draw_idle()

    mat_radio.on_clicked(update); mode_radio.on_clicked(update)
    for s in sliders: s.on_changed(update)
    fig.suptitle("Band-Gap Engineering Explorer — interactive local Python model", fontsize=14)
    update()
    plt.show()

if __name__ == "__main__":
    main()
