"""
Slike publiciranog kvaliteta za simulaciju GW150914. Generira osam slika
(fig1–fig8) pohranjenih u results/. Koristi matplotlib sa serifnim fontovima
i izlaznom rezolucijom 150 dpi.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  1.4,
})


def _save(fig, name):
    """Pohrani figuru u results/ i zatvori je.

    Args:
        fig:  matplotlib Figure objekt.
        name: Naziv izlazne datoteke (npr. "fig1_orbital_decay.png").

    Returns:
        Apsolutna putanja do pohranjene datoteke.
    """
    path = os.path.join(RESULTS, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")
    return path


def plot_orbital_decay(t, f, sep_m, label="GW150914"):
    """Prikazuje orbitalnu separaciju i frekvenciju GW nasuprot vremenu do spajanja.

    Dvo-plošna slika. Gornja plosa: orbitalna separacija a(t) [km]. Donja
    plosa: frekvencija GW f(t) [Hz]. Vremenska os referira se na trenutak
    spajanja (t = 0). Pohranjuje fig1_orbital_decay.png.

    Args:
        t:     Polje vremena [s].
        f:     Polje frekvencije GW [Hz].
        sep_m: Polje orbitalne separacije [m].
        label: Oznaka za naslov slike.

    Returns:
        Putanja do pohranjene slike.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(f"Orbitalna evolucija — {label}", fontsize=13)

    t_ms = (t - t[-1]) * 1e3
    ax1.plot(t_ms, sep_m / 1e3, color="steelblue")
    ax1.set_ylabel("Orbitalni razmak [km]")
    ax1.set_xlim(t_ms[0], 20)

    ax2.plot(t_ms, f, color="darkorange")
    ax2.set_ylabel("Frekvencija GW [Hz]")
    ax2.set_xlabel("Vrijeme do spajanja [ms]")
    ax2.set_xlim(t_ms[0], 20)

    fig.tight_layout()
    return _save(fig, "fig1_orbital_decay.png")


def plot_waveform(t, h_plus, h_cross, label="GW150914"):
    """Prikazuje dva polarizacijska straina gravitacijskih valova h₊(t) i h×(t).

    Dvo-plošna slika. Strain skaliran na jedinice 10⁻²¹. Vremenska os referira
    se na spajanje. Pohranjuje fig2_waveform.png.

    Args:
        t:       Polje vremena [s].
        h_plus:  Plus polarizacijski strain h₊(t) [bezdimenzionalno].
        h_cross: Križni polarizacijski strain h×(t) [bezdimenzionalno].
        label:   Oznaka za naslov slike.

    Returns:
        Putanja do pohranjene slike.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig.suptitle(f"Naprezanje gravitacijskog vala — {label}", fontsize=13)

    t_ms = (t - t[-1]) * 1e3

    axes[0].plot(t_ms, h_plus * 1e21, color="steelblue", lw=0.8)
    axes[0].set_ylabel(r"$h_+\;[\times10^{-21}]$")
    axes[0].set_xlim(-800, 30)

    axes[1].plot(t_ms, h_cross * 1e21, color="darkorange", lw=0.8)
    axes[1].set_ylabel(r"$h_\times\;[\times10^{-21}]$")
    axes[1].set_xlabel("Vrijeme do spajanja [ms]")
    axes[1].set_xlim(-800, 30)

    fig.tight_layout()
    return _save(fig, "fig2_waveform.png")


def plot_spectrogram(t, h, sample_rate, label="GW150914"):
    """Prikazuje vremensko-frekvencijski spektrogram od h₊ koristeći STFT.

    Skala boja: log₁₀(spektralna gustoća snage). Frekvencijski raspon
    10–500 Hz. Prikazuje rastuć chirp trag od ~20 Hz do ~150 Hz karakterističan
    za ulazak u spiralu. Pohranjuje fig3_spectrogram.png.

    Args:
        t:           Polje vremena valnog oblika [s] (koristi se za postavljanje
                     osi).
        h:           Plus polarizacijski strain [bezdimenzionalno].
        sample_rate: Brzina uzorkovanja [Hz].
        label:       Oznaka za naslov slike.

    Returns:
        Putanja do pohranjene slike.
    """
    from physics.spectrum import compute_spectrogram

    t_s, f_s, Sxx = compute_spectrogram(h, sample_rate, nperseg=256, noverlap=248)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle(f"Spektrogram — {label}", fontsize=13)

    t_ms = (t_s - t_s[-1]) * 1e3
    pcm = ax.pcolormesh(
        t_ms, f_s,
        np.log10(Sxx + 1e-60),
        shading="gouraud", cmap="inferno",
    )
    ax.set_ylabel("Frekvencija [Hz]")
    ax.set_xlabel("Vrijeme do spajanja [ms]")
    ax.set_xlim(-700, 30)
    ax.set_ylim(10, 500)
    cb = fig.colorbar(pcm, ax=ax, label=r"$\log_{10}$ Power")
    fig.tight_layout()
    return _save(fig, "fig3_spectrogram.png")


def plot_mass_variation(variants_data):
    """Prikazuje h₊(t) za svaku varijantu mase na zasebnim plohama.

    Ilustrira kako ukupna masa sustava kontrolira trajanje valnog oblika i
    frekvencijski raspon: teži sustavi spajaju se ranije i pri nižim
    frekvencijama. Pohranjuje fig4_mass_variation.png.

    Args:
        variants_data: Lista (t, h_plus, label) torki, jedna po varijanti mase.

    Returns:
        Putanja do pohranjene slike.
    """
    fig, axes = plt.subplots(len(variants_data), 1,
                             figsize=(10, 2.5 * len(variants_data)),
                             sharex=False)
    fig.suptitle("Učinak ukupne mase na valne oblike", fontsize=13)

    colors = ["steelblue", "darkorange", "seagreen", "crimson"]

    for ax, (t, hp, label), color in zip(axes, variants_data, colors):
        t_ms = (t - t[-1]) * 1e3
        ax.plot(t_ms, hp * 1e21, color=color, lw=0.8)
        ax.set_xlim(max(t_ms[0], -1000), 30)
        ax.set_ylabel(r"$h_+\;[\times10^{-21}]$")
        ax.text(0.02, 0.88, label, transform=ax.transAxes,
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    axes[-1].set_xlabel("Vrijeme do spajanja [ms]")
    fig.tight_layout()
    return _save(fig, "fig4_mass_variation.png")


def plot_distance_variation(distances_mpc, peak_strains):
    """Prikazuje vršni strain h_peak nasuprot luminoznoj udaljenosti na log-log osiima.

    Prekriva teorijski trend h ∝ 1/d isprekidanom linijom i anotira točku
    koja odgovara GW150914. Pohranjuje fig5_distance_variation.png.

    Args:
        distances_mpc: Lista luminoznih udaljenosti [Mpc].
        peak_strains:  Lista vršnih straina pri ISCO-u [bezdimenzionalno].

    Returns:
        Putanja do pohranjene slike.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(distances_mpc, np.array(peak_strains) * 1e21, "o-", color="steelblue", ms=5)
    ax.set_xlabel("Luminozitetna udaljenost [Mpc]")
    ax.set_ylabel(r"Vršno naprezanje $h_\mathrm{peak}\;[\times10^{-21}]$")
    ax.set_title("Vršno naprezanje prema udaljenosti")
    ax.set_xscale("log")
    ax.set_yscale("log")

    idx = np.argmin(np.abs(np.array(distances_mpc) - 440))
    ax.annotate("GW150914",
                xy=(distances_mpc[idx], peak_strains[idx] * 1e21),
                xytext=(distances_mpc[idx] * 1.2, peak_strains[idx] * 1e21 * 1.4),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9)

    d_range = np.linspace(min(distances_mpc), max(distances_mpc), 200)
    h_ref   = peak_strains[0] * distances_mpc[0]
    ax.plot(d_range, h_ref / d_range * 1e21, "k--", alpha=0.4, label=r"$h\propto d^{-1}$")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "fig5_distance_variation.png")


def plot_chirp_mass_panel(mc_values_msun, freq_data):
    """Prikazuje evoluciju frekvencije GW f(t) za više chirp masa na jednoj plosi.

    Ilustrira kako teži sustavi (veća chirp masa) spajaju se brže i pri nižim
    frekvencijama — strmeji nagib f(t) krivulje odgovara brže rastućoj
    frekvenciji. Pohranjuje fig6_chirp_mass_panel.png.

    Args:
        mc_values_msun: Lista chirp masa [sunčeve mase].
        freq_data:      Lista (t, f, label) torki, jedna po varijanti.

    Returns:
        Putanja do pohranjene slike.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Evolucija frekvencije za različite čirp-mase", fontsize=13)
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]

    for (t, f, label), c in zip(freq_data, colors):
        t_ms = (t - t[-1]) * 1e3
        ax.plot(t_ms, f, color=c, label=label, lw=1.4)

    ax.set_xlabel("Vrijeme do spajanja [ms]")
    ax.set_ylabel("Frekvencija GW [Hz]")
    ax.set_xlim(-800, 20)
    ax.set_ylim(0, 500)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "fig6_chirp_mass_panel.png")


def plot_inclination_effects(iota_deg_arr, hplus_max_arr, hcross_max_arr):
    """Prikazuje maksimalnu amplitudu straina h₊ i h× kao funkciju kuta inklinacije ι.

    Ilustrira geometrijsku ovisnost: h₊ je maksimalan pri ι = 0° (lice-prema-
    promatraču) i jednak nuli pri ι = 90° × sqrt(2) faktoru; h× iščezava pri
    ι = 0° i 180°. Pohranjuje fig7_inclination_effects.png.

    Args:
        iota_deg_arr:   Lista kutova inklinacije [stupnjevi].
        hplus_max_arr:  Lista maksimalnih vrijednosti |h₊| [bezdimenzionalno].
        hcross_max_arr: Lista maksimalnih vrijednosti |h×| [bezdimenzionalno].

    Returns:
        Putanja do pohranjene slike.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(iota_deg_arr, np.array(hplus_max_arr) * 1e21, "o-",
            color="steelblue", label=r"$|h_+|_\mathrm{max}$")
    ax.plot(iota_deg_arr, np.array(hcross_max_arr) * 1e21, "s--",
            color="darkorange", label=r"$|h_\times|_\mathrm{max}$")
    ax.set_xlabel(r"Inklinacija $\iota$ [deg]")
    ax.set_ylabel(r"Vršno naprezanje $[\times10^{-21}]$")
    ax.set_title("Amplituda naprezanja prema kutu nagiba")
    ax.legend()
    ax.set_xlim(0, 180)
    fig.tight_layout()
    return _save(fig, "fig7_inclination_effects.png")


def plot_parameter_comparison(theory_dict, observed_dict):
    """Grupirani stupčani dijagram koji uspoređuje simulacijom predviđene nasuprot LIGO opaženim parametrima.

    Uspoređuje M_c, f_isco i h_peak. Postotna razlika između simulacije i
    opažanja anotirana je iznad svake grupe stupaca. Pohranjuje
    fig8_parameter_comparison.png.

    Args:
        theory_dict:   Rječnik {oznaka_parametra: vrijednost} za simulaciju.
        observed_dict: Rječnik {oznaka_parametra: vrijednost} za LIGO opažanje.
                       Isti ključevi kao theory_dict.

    Returns:
        Putanja do pohranjene slike.
    """
    keys   = list(theory_dict.keys())
    t_vals = [theory_dict[k]  for k in keys]
    o_vals = [observed_dict[k] for k in keys]

    x = np.arange(len(keys))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_t = ax.bar(x - w/2, t_vals, w, label="Ovaj rad (simulacija)",
                    color="steelblue", alpha=0.8)
    bars_o = ax.bar(x + w/2, o_vals, w, label="LIGO/Virgo observacija",
                    color="darkorange", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel("Vrijednost")
    ax.set_title("Usporedba parametara: simulacija i opažanja GW150914")
    ax.legend()

    for i, (tv, ov) in enumerate(zip(t_vals, o_vals)):
        diff = abs(tv - ov) / ov * 100 if ov != 0 else 0
        ax.text(i, max(tv, ov) * 1.02, f"{diff:.1f}%",
                ha="center", fontsize=8, color="black")

    fig.tight_layout()
    return _save(fig, "fig8_parameter_comparison.png")
