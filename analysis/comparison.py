"""
Analiza valnih oblika i funkcije usporedbe
"""

import numpy as np
from physics.waveform import generate_waveform, chirp_mass_from_fdot
from physics.spectrum import overlap_integral
from physics.orbital import chirp_mass, isco_frequency, freq_deriv
from config import G, C, M_SUN


def measure_chirp_mass(t, f):
    """Oporavlja chirp masu M_c iz simulirane evolucije frekvencije f(t)

    Args:
        t: Polje vremena [s]
        f: Polje frekvencije GW [Hz]

    Returns:
        Oporavljena chirp masa [sunčeve mase]
    """
    n = len(f)
    if n < 8:
        i0, i1 = 0, n
    else:
        i0 = n // 4
        i1 = 3 * n // 4
    t_mid = t[i0:i1]
    f_mid = f[i0:i1]
    if len(t_mid) < 2:
        t_mid, f_mid = t, f

    fdot = np.gradient(f_mid, t_mid)
    mid = len(f_mid) // 2
    f_c = f_mid[mid]
    fd_c = fdot[mid]

    return chirp_mass_from_fdot(f_c, fd_c) / M_SUN


def waveform_snr(h_plus, h_cross, sample_rate, iota_deg=0.0):
    """Procjenjuje pojednostavljeni omjer signal/šum za plus polarizaciju

    Args:
        h_plus:      Plus polarizacijski strain [0-dimenzionalno].
        h_cross:     Križni polarizacijski strain [0-dimenzionalno] — nekorišten
        sample_rate: Brzina uzorkovanja [Hz].
        iota_deg:    Kut inklinacije [stupnjevi] — nekorišten

    Returns:
        Procjena SNR-a [0-dimenzionalno]
    """
    from physics.spectrum import compute_fft
    f, Hf = compute_fft(h_plus, sample_rate)
    df = sample_rate / len(h_plus)
    snr_sq = 4.0 * df * np.sum(np.abs(Hf)**2)
    return np.sqrt(snr_sq)


def compare_two_waveforms(params1, params2, sample_rate=4096):
    """Izračunava normalizirano preklapanje između dva valna oblika

    Args:
        params1:     Rječnik parametara za valni oblik 1; ključevi: m1_msun,
                     m2_msun, d_mpc, iota_deg, f0_hz.
        params2:     Rječnik parametara za valni oblik 2; isti ključevi
        sample_rate: Brzina uzorkovanja za oba valna oblika [Hz]

    Returns:
        Normalizirani match ∈ [0, 1] [0-dimenzionalno]
    """
    _, _, h1, _, _ = generate_waveform(**params1, sample_rate=sample_rate)
    _, _, h2, _, _ = generate_waveform(**params2, sample_rate=sample_rate)
    _, match = overlap_integral(h1, h2, sample_rate)
    return match


def fdot_at_frequency(f_hz, m1_msun, m2_msun):
    """Analitički izračunava df/dt pri zadanoj frekvenciji GW

    Args:
        f_hz:     Frekvencija GW pri kojoj se evaluira df/dt [Hz]
        m1_msun:  Masa primarne crne rupe [sunčeve mase]
        m2_msun:  Masa sekundarne crne rupe [sunčeve mase]

    Returns:
        df/dt [Hz/s]
    """
    Mc = chirp_mass(m1_msun, m2_msun)
    return freq_deriv(0, [f_hz], Mc)[0]
