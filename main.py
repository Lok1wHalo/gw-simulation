"""
Glavna izvršna skripta za simulaciju gravitacijskih valova GW150914. Pokreće
cijeli 'pipeline': generiranje valnog oblika, orbitalni raspad, spektralnu
analizu, studije parametara (masa, udaljenost, inklinacija) i usporedbu
teorije s opažanjem. Generira slike fig1–fig8 u results/. 
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from config import PARAMS, MASS_VARIANTS, DISTANCE_VARIANTS, INCLINATION_VARIANTS, M_SUN, MPC
from physics.orbital   import (evolve_frequency, orbital_separation,
                                chirp_mass, isco_frequency, merger_time)
from physics.waveform  import generate_waveform, peak_strain
from physics.spectrum  import compute_spectrogram
from analysis.comparison import measure_chirp_mass
from plots.visualise   import (plot_orbital_decay, plot_waveform, plot_spectrogram,
                                plot_mass_variation, plot_distance_variation,
                                plot_chirp_mass_panel, plot_inclination_effects,
                                plot_parameter_comparison)


def run():
    print("=" * 60)
    print("  Binary Black Hole GW Simulation")
    print("=" * 60)
    p = PARAMS

    print(f"\n[1] Generating waveform for {p['name']} ...")
    t, f, h_plus, h_cross, phase = generate_waveform(
        m1_msun    = p["m1"],
        m2_msun    = p["m2"],
        d_mpc      = p["d_mpc"],
        iota_deg   = p["iota_deg"],
        f0_hz      = p["f0_hz"],
        sample_rate= p["sample_rate"],
    )
    sep = orbital_separation(t, f, p["m1"], p["m2"])

    Mc_true  = chirp_mass(p["m1"], p["m2"]) / M_SUN
    Mc_recov = measure_chirp_mass(t, f)
    f_isco   = isco_frequency(p["m1"], p["m2"])
    h_peak   = peak_strain(p["m1"], p["m2"], p["d_mpc"])

    print(f"  Duration in band:  {t[-1] - t[0]:.3f} s")
    print(f"  Frequency range:   {f[0]:.1f} – {f[-1]:.1f} Hz")
    print(f"  ISCO frequency:    {f_isco:.1f} Hz")
    print(f"  True chirp mass:   {Mc_true:.2f} M_sun")
    print(f"  Recovered M_c:     {Mc_recov:.2f} M_sun  (error: {abs(Mc_true-Mc_recov)/Mc_true*100:.2f}%)")
    print(f"  Peak strain:       {h_peak:.3e}")

    plot_orbital_decay(t, f, sep, label=p["label"])
    plot_waveform(t, h_plus, h_cross, label=p["label"])
    plot_spectrogram(t, h_plus, p["sample_rate"], label=p["label"])

    print("\n[2] Mass variation ...")
    mass_data = []
    for v in MASS_VARIANTS:
        t_v, f_v, hp_v, _, _ = generate_waveform(
            m1_msun=v["m1"], m2_msun=v["m2"],
            d_mpc=v["d_mpc"], iota_deg=v["iota_deg"],
            f0_hz=v["f0_hz"], sample_rate=p["sample_rate"],
        )
        Mc_v = chirp_mass(v["m1"], v["m2"]) / M_SUN
        print(f"  {v['label']:40s}  M_c = {Mc_v:.2f} M_sun  duration = {t_v[-1]-t_v[0]:.2f} s")
        mass_data.append((t_v, hp_v, v["label"]))
    plot_mass_variation(mass_data)

    print("\n[3] Distance variation ...")
    dist_list, strain_list = [], []
    for v in DISTANCE_VARIANTS:
        hs = peak_strain(v["m1"], v["m2"], v["d_mpc"])
        dist_list.append(v["d_mpc"])
        strain_list.append(hs)
        print(f"  d = {v['d_mpc']:5.0f} Mpc   h_peak = {hs:.3e}")
    plot_distance_variation(dist_list, strain_list)

    print("\n[4] Chirp mass panel ...")
    freq_data, mc_vals = [], []
    for v in MASS_VARIANTS:
        t_v, f_v, _, _, _ = generate_waveform(
            m1_msun=v["m1"], m2_msun=v["m2"],
            d_mpc=v["d_mpc"], iota_deg=v["iota_deg"],
            f0_hz=v["f0_hz"], sample_rate=p["sample_rate"],
        )
        Mc_v = chirp_mass(v["m1"], v["m2"]) / M_SUN
        mc_vals.append(Mc_v)
        label = v["label"] + f"  $\\mathcal{{M}}_c={Mc_v:.1f}M_\\odot$"
        freq_data.append((t_v, f_v, label))
    plot_chirp_mass_panel(mc_vals, freq_data)

    print("\n[5] Inclination variation ...")
    iota_arr, hplus_max_arr, hcross_max_arr = [], [], []
    for v in INCLINATION_VARIANTS:
        _, _, hp_v, hc_v, _ = generate_waveform(
            m1_msun=v["m1"], m2_msun=v["m2"],
            d_mpc=v["d_mpc"], iota_deg=v["iota_deg"],
            f0_hz=v["f0_hz"], sample_rate=p["sample_rate"],
        )
        iota_arr.append(v["iota_deg"])
        hplus_max_arr.append(np.max(np.abs(hp_v)))
        hcross_max_arr.append(np.max(np.abs(hc_v)))
        print(f"  iota={v['iota_deg']:6.1f}°   |h+|_max={np.max(np.abs(hp_v)):.3e}   "
              f"|hx|_max={np.max(np.abs(hc_v)):.3e}")
    plot_inclination_effects(iota_arr, hplus_max_arr, hcross_max_arr)

    print("\n[6] Theory vs observation (GW150914) ...")
    theory_dict = {
        r"$\mathcal{M}_c\,[M_\odot]$":  round(Mc_true,   1),
        r"$f_\mathrm{ISCO}\,[\mathrm{Hz}]$": round(f_isco, 1),
        r"$h_\mathrm{peak}\,[\times10^{-21}]$": round(h_peak * 1e21, 2),
    }
    observed_dict = {
        r"$\mathcal{M}_c\,[M_\odot]$":  28.3,
        r"$f_\mathrm{ISCO}\,[\mathrm{Hz}]$": 67.6,
        r"$h_\mathrm{peak}\,[\times10^{-21}]$": 1.00,
    }
    for k in theory_dict:
        print(f"  {k:40s}  theory={theory_dict[k]}   observed={observed_dict[k]}")
    plot_parameter_comparison(theory_dict, observed_dict)

    print("\n[Done] All figures saved to results/")
    print("=" * 60)

    return dict(
        Mc_true   = Mc_true,
        Mc_recov  = Mc_recov,
        f_isco    = f_isco,
        h_peak    = h_peak,
        duration  = t[-1] - t[0],
        f_start   = f[0],
        f_end     = f[-1],
    )


if __name__ == "__main__":
    results = run()
