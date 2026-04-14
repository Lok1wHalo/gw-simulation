"""
Opcionalni modul za preuzimanje i usporedbu sa stvarnim LIGO strain podacima
putem javne GWOSC arhive (Gravitational Wave Open Science Center). Zahtijeva
vanjske pakete gwosc i gwpy koji nisu u standardnom projektnom okruženju —
instalirati zasebno po potrebi. Vidi odjeljak 3.6 rada.
"""

import numpy as np


def fetch_gw150914(detector="H1", duration=4, sample_rate=4096):
    """Preuzima strain podatke GW150914 iz javne GWOSC arhive.

    Zahtijeva pakete gwosc i gwpy koji nisu u standardnom projektnom okruženju.

    Dohvaća vremenski segment trajanja duration sekundi centriran oko GPS
    okidnog vremena GW150914 (1126259462,4 s). Vraća gwpy TimeSeries objekt.

    Args:
        detector:    Identifikator LIGO detektora: "H1" (Hanford) ili
                     "L1" (Livingston). Zadano "H1".
        duration:    Ukupno trajanje podatkovnog segmenta [s]. Zadano 4.
        sample_rate: Izlazna brzina uzorkovanja [Hz]. Zadano 4096.

    Returns:
        strain: gwpy TimeSeries sirovih strain podataka.
        gps:    GPS okidno vrijeme GW150914 [s].
    """
    from gwosc.datasets import event_gps
    from gwpy.timeseries import TimeSeries

    gps    = event_gps("GW150914")
    start  = gps - duration / 2
    end    = gps + duration / 2
    strain = TimeSeries.fetch_open_data(detector, start, end,
                                        sample_rate=sample_rate)
    return strain, gps


def bandpass_filter(strain, flow=20, fhigh=400):
    """Primjenjuje pojasno-propusni filtar na LIGO strain podatke.

    Zadržava signal u frekvencijskom pojasu [flow, fhigh] Hz i prigušuje
    sve izvan njega. Time se potiskuje seizmički zid na niskim frekvencijama
    i šum snimke na visokim frekvencijama, čime chirp gravitacijskih valova
    postaje vidljiv u vremenskoj domeni.

    Args:
        strain: gwpy TimeSeries sirovih LIGO strain podataka.
        flow:   Donja frekvencija pojasnog propusnika [Hz]. Zadano 20.
        fhigh:  Gornja frekvencija pojasnog propusnika [Hz]. Zadano 400.

    Returns:
        Pojasno-filtrirani gwpy TimeSeries.
    """
    return strain.bandpass(flow, fhigh)


def plot_comparison(strain_data, h_theory, t_theory, gps, output_path="results/fig_gwosc_comparison.png"):
    """Prekriva LIGO opaženi strain s teorijskim valnim oblikom.

    Generira jednoplohu vremensku usporedbu. LIGO podaci prikazani su sivom
    bojom; Newtonov ulazni valni oblik preklopljen je plavom bojom. Oboje su
    skalirani na jedinice 10⁻²¹ radi čitljivosti. Vremenska os referira se
    na spajanje (t = 0 na kraju h_theory).

    Napomena: Newtonov valni oblik ne uključuje ringdown nakon spajanja, pa
    se slaganje očekuje samo za vrijeme ulaznog chirpa, a ne nakon t ≈ 0.

    Args:
        strain_data: Pojasno-filtrirani gwpy TimeSeries (H1 podaci).
        h_theory:    Teorijski strain h₊(t) kao numpy polje [bezdimenzionalno].
        t_theory:    Polje vremena za teorijski valni oblik [s].
        gps:         GPS okidno vrijeme GW150914; koristi se za poravnanje
                     LIGO vremenske osi [s].
        output_path: Putanja za pohranu slike. Zadano
                     results/fig_gwosc_comparison.png.

    Returns:
        Putanja do pohranjene slike.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    t_ligo  = strain_data.times.value - gps

    ax.plot(t_ligo, strain_data.value * 1e21,
            color="gray", lw=0.5, alpha=0.7, label="LIGO H1 (bandpassed)")

    t_th = t_theory - t_theory[-1]          # align merger to t = 0
    ax.plot(t_th, h_theory * 1e21,
            color="steelblue", lw=1.5, label="Theoretical waveform (Newtonian)")

    ax.set_xlim(-0.5, 0.1)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlabel("Time from merger [s]")
    ax.set_ylabel(r"Strain $[\times 10^{-21}]$")
    ax.set_title("GW150914: LIGO Data vs Theoretical Waveform")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {output_path}")
    return output_path


def run():
    """Pokreće cjeloviti GWOSC tijek usporedbe: preuzimanje → filtriranje → valni oblik → slika."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from physics.waveform import generate_waveform
    from config import PARAMS

    print("[GWOSC] Downloading GW150914 data...")
    strain, gps = fetch_gw150914()

    print("[GWOSC] Conditioning data...")
    strain_bp = bandpass_filter(strain)

    print("[GWOSC] Generating theoretical waveform...")
    p = PARAMS
    t, f, h_plus, h_cross, _ = generate_waveform(
        m1_msun    = p["m1"],
        m2_msun    = p["m2"],
        d_mpc      = p["d_mpc"],
        iota_deg   = p["iota_deg"],
        f0_hz      = p["f0_hz"],
        sample_rate= p["sample_rate"],
    )

    print("[GWOSC] Plotting comparison...")
    plot_comparison(strain_bp, h_plus, t, gps)
    print("[GWOSC] Done.")
