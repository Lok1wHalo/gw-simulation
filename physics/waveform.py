"""
Generiranje gravitacijskog valnog oblika u kvadrupolnoj aproksimaciji.
Izračunava amplitudu straina, evoluciju faze i dva polarizacijska stanja
h₊ i h×.  
"""

import numpy as np
from config import G, C, MPC, M_SUN
from physics.orbital import evolve_frequency, chirp_mass


def strain_amplitude(f_gw_hz, m1_msun, m2_msun, d_mpc):
    """Izračunava amplitudu straina gravitacijskih valova A(f) u kvadrupolnoj aproksimaciji.

    U kvadrupolnoj aproksimaciji, karakteristična amplituda straina kružnog
    binarnog sustava na luminoznoj udaljenosti d_L iznosi:

        A(f) = (4/d_L) × (G M_c / c²) × (π G M_c f / c³)^(2/3)

    Ovo je omotač amplitude; stvarni polarizacijski straini h₊ i h× dobivaju
    se množenjem A s faktorima ovisnim o inklinaciji u generate_waveform.

    Args:
        f_gw_hz:  Frekvencija gravitacijskih valova [Hz]. Može biti skalar ili
                  polje.
        m1_msun:  Masa primarne crne rupe [sunčeve mase].
        m2_msun:  Masa sekundarne crne rupe [sunčeve mase].
        d_mpc:    Luminozna udaljenost [Mpc].

    Returns:
        Amplituda straina A(f) [bezdimenzionalno]. Isti oblik kao f_gw_hz.
    """
    Mc = chirp_mass(m1_msun, m2_msun)
    dL = d_mpc * MPC
    return (4.0 / dL) * (G * Mc / C**2) * (np.pi * G * Mc * f_gw_hz / C**3) ** (2.0 / 3.0)


def generate_waveform(m1_msun, m2_msun, d_mpc, iota_deg, f0_hz=20.0, sample_rate=4096):
    """Generira vremenski valni oblik gravitacijskih valova (h₊, h×) za kružni binarni ulazak u spiralu.

    Koraci:
    1. Integrira Petersovu ODE za dobivanje f(t) od f0 do f_isco.
    2. Akumulira GW fazu: φ(t) = 2π Σ f(tᵢ) Δt  (diskretni integral).
    3. Izračunava omotač amplitude A(f) u svakom vremenskom koraku.
    4. Primjenjuje faktore inklinacije za dva polarizacijska stanja:
           h₊(t) = A(t) × (1 + cos²ι) / 2 × cos φ(t)
           h×(t) = A(t) × cos ι         × sin φ(t)
       gdje je ι kut inklinacije između orbitalne kutne količine gibanja i
       pravca gledanja.  

    Za ι = 0° (lice-prema-promatraču), h₊ je pri maksimalnoj amplitudi i oba
    polarizacijska stanja su jednaka po magnitudin. Za ι = 90° (rub-prema-
    promatraču), h× = 0 i h₊ je smanjen za faktor 1/2.

    Args:
        m1_msun:     Masa primarne crne rupe [sunčeve mase].
        m2_msun:     Masa sekundarne crne rupe [sunčeve mase].
        d_mpc:       Luminozna udaljenost [Mpc].
        iota_deg:    Kut inklinacije ι [stupnjevi]. 0 = lice-prema-promatraču,
                     90 = rub-prema-promatraču. Najbolja procjena za GW150914: 153°.
        f0_hz:       Početna frekvencija GW [Hz]. Zadano 20 Hz.
        sample_rate: Brzina uzorkovanja [Hz]. Zadano 4096 Hz.

    Returns:
        t:       Polje vremena [s].
        f:       Polje frekvencije GW [Hz].
        h_plus:  Plus polarizacijski strain h₊(t) [bezdimenzionalno].
        h_cross: Križni polarizacijski strain h×(t) [bezdimenzionalno].
        phase:   Akumulirana GW faza φ(t) [radijani].
    """
    t, f = evolve_frequency(m1_msun, m2_msun, f0_hz, sample_rate=sample_rate)

    dt    = 1.0 / sample_rate
    iota  = np.radians(iota_deg)

    phase = 2.0 * np.pi * np.cumsum(f) * dt  # akumulacija GW faze: φ(t) = 2π ∫f(t')dt', diskretizirana kao kumulativna suma

    A = strain_amplitude(f, m1_msun, m2_msun, d_mpc)

    h_plus  = A * (1.0 + np.cos(iota)**2) / 2.0 * np.cos(phase)  # plus polarizacija: h₊ = A(1 + cos²ι)/2 · cos φ  (vidi odjeljak 3.3)
    h_cross = A *  np.cos(iota)            * np.sin(phase)         # križna polarizacija: h× = A cosι · sin φ  (vidi odjeljak 3.3)

    return t, f, h_plus, h_cross, phase


def peak_strain(m1_msun, m2_msun, d_mpc):
    """Izračunava vršni strain gravitacijskih valova pri ISCO-u.

    Amplituda straina A(f) monotono raste za vrijeme ulaska u spiralu.
    Njezina maksimalna vrijednost u okviru kvadrupolne aproksimacije dostiže
    se pri ISCO-u, pa je h_peak = A(f_isco). Pruža karakterizaciju jačine
    signala jednim brojem.  

    Args:
        m1_msun: Masa primarne crne rupe [sunčeve mase].
        m2_msun: Masa sekundarne crne rupe [sunčeve mase].
        d_mpc:   Luminozna udaljenost [Mpc].

    Returns:
        Vršni strain pri ISCO-u [bezdimenzionalno].
    """
    from physics.orbital import isco_frequency
    f_isco = isco_frequency(m1_msun, m2_msun)
    return strain_amplitude(f_isco, m1_msun, m2_msun, d_mpc)


def chirp_mass_from_fdot(f_hz, fdot_hz_per_s):
    """Oporavlja chirp masu M_c iz opažene frekvencije GW i njezine derivacije po vremenu.

    Invertira Petersovu formulu df/dt = (96/5) π (π G M_c / c³)^(5/3) f^(11/3)
    po M_c:

        M_c = (c³/G) × (5 / (96 π^(8/3)))^(3/5) × f^(-11/5) × (df/dt)^(3/5)

    Standardni je estimator chirp mase usklađenog filtra koji se koristi u
    analizi GW podataka. Vidi odjeljak 3.6 rada.

    Args:
        f_hz:          Frekvencija GW u točki mjerenja [Hz].
        fdot_hz_per_s: Derivacija frekvencije GW po vremenu df/dt [Hz/s].

    Returns:
        Kirp masa [kg].
    """
    coeff = (5.0 / (96.0 * np.pi ** (8.0 / 3.0))) ** 0.6
    return (C**3 / G) * coeff * f_hz ** (-11.0 / 5.0) * fdot_hz_per_s ** (3.0 / 5.0)
