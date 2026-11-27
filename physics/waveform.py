"""
Generiranje gravitacijskog valnog oblika
"""

import numpy as np
from config import G, C, MPC, M_SUN
from physics.orbital import evolve_frequency, chirp_mass


def strain_amplitude(f_gw_hz, m1_msun, m2_msun, d_mpc):
    """Izračunava amplitudu straina gravitacijskih valova A(f) u kvadrupolnoj aproksimaciji

    Args:
        f_gw_hz:  Frekvencija gravitacijskih valova [Hz]; skalar ili polje.
        m1_msun:  Masa primarne crne rupe [sunčeve mase]
        m2_msun:  Masa sekundarne crne rupe [sunčeve mase]
        d_mpc:    Luminozna udaljenost [Mpc]

    Returns:
        Amplituda straina A(f) [0-dimenzionalno]; isti oblik kao f_gw_hz
    """
    Mc = chirp_mass(m1_msun, m2_msun)
    dL = d_mpc * MPC
    return (4.0 / dL) * (G * Mc / C**2) * (np.pi * G * Mc * f_gw_hz / C**3) ** (2.0 / 3.0)


def generate_waveform(m1_msun, m2_msun, d_mpc, iota_deg, f0_hz=20.0, sample_rate=4096):
    """Generira vremenski valni oblik gravitacijskih valova (h₊, h×) za kružni binarni ulazak u spiralu.

    Args:
        m1_msun:     Masa primarne crne rupe [sunčeve mase]
        m2_msun:     Masa sekundarne crne rupe [sunčeve mase]
        d_mpc:       Luminozna udaljenost [Mpc]
        iota_deg:    Kut inklinacije ι [stupnjevi]
                     0 = lice-prema-promatraču, 90 = rub-prema-promatraču
                     Najbolja procjena za GW150914: 153°
        f0_hz:       Početna frekvencija GW [Hz]; zadano 20 Hz
        sample_rate: Brzina uzorkovanja [Hz]; zadano 4096 Hz

    Returns:
        t:       Polje vremena [s]
        f:       Polje frekvencije GW [Hz]
        h_plus:  Plus polarizacijski strain h₊(t) [0-dimenzionalno]
        h_cross: Križni polarizacijski strain h×(t) [0-dimenzionalno]
        phase:   Akumulirana GW faza φ(t) [rad]
    """
    t, f = evolve_frequency(m1_msun, m2_msun, f0_hz, sample_rate=sample_rate)

    dt = 1.0 / sample_rate
    iota = np.radians(iota_deg)

    phase = 2.0 * np.pi * np.cumsum(f) * dt  

    A = strain_amplitude(f, m1_msun, m2_msun, d_mpc)

    h_plus  = A * (1.0 + np.cos(iota)**2) / 2.0 * np.cos(phase)  
    h_cross = A * np.cos(iota) * np.sin(phase)         

    return t, f, h_plus, h_cross, phase


def peak_strain(m1_msun, m2_msun, d_mpc):
    """Izračunava vršni strain gravitacijskih valova pri ISCO-u

    Args:
        m1_msun: Masa primarne crne rupe [sunčeve mase]
        m2_msun: Masa sekundarne crne rupe [sunčeve mase]
        d_mpc:   Luminozna udaljenost [Mpc]

    Returns:
        Vršni strain pri ISCO-u [0-dimenzionalno]
    """
    from physics.orbital import isco_frequency
    f_isco = isco_frequency(m1_msun, m2_msun)
    return strain_amplitude(f_isco, m1_msun, m2_msun, d_mpc)


def chirp_mass_from_fdot(f_hz, fdot_hz_per_s):
    """Oporavlja chirp masu M_c iz opažene frekvencije GW i njezine derivacije po vremenu

    Args:
        f_hz:          Frekvencija GW u točki mjerenja [Hz]
        fdot_hz_per_s: Derivacija frekvencije GW po vremenu df/dt [Hz/s]

    Returns:
        Chirp masa [kg]
    """
    coeff = (5.0 / (96.0 * np.pi ** (8.0 / 3.0))) ** 0.6
    return (C**3 / G) * coeff * f_hz ** (-11.0 / 5.0) * fdot_hz_per_s ** (3.0 / 5.0)
