"""
Orbitalna mehanika binarnog sustava u spiralnom ulasku
"""

import numpy as np
from scipy.integrate import solve_ivp
from config import G, C, M_SUN


def chirp_mass(m1_msun, m2_msun):
    """Izračunava chirp masu M_c u kilogramima

    Args:
        m1_msun: Masa primarne crne rupe [sunčeve mase]
        m2_msun: Masa sekundarne crne rupe [sunčeve mase]

    Returns:
        Chirp masa [kg]
    """
    m1 = m1_msun * M_SUN
    m2 = m2_msun * M_SUN
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def total_mass(m1_msun, m2_msun):
    """Izračunava ukupnu masu sustava M = m1 + m2 u kilogramima

    Args:
        m1_msun: Masa primarne crne rupe [sunčeve mase]
        m2_msun: Masa sekundarne crne rupe [sunčeve mase]

    Returns:
        Ukupna masa [kg]
    """
    return (m1_msun + m2_msun) * M_SUN


def isco_frequency(m1_msun, m2_msun):
    """Izračunava frekvenciju gravitacijskih valova pri ISCO-u u [Hz]
   
    Args:
        m1_msun: Masa primarne crne rupe [sunčeve mase]
        m2_msun: Masa sekundarne crne rupe [sunčeve mase]

    Returns:
        Frekvencija gravitacijskih valova pri ISCO [Hz]
    """
    M = total_mass(m1_msun, m2_msun)
    return C**3 / (6**1.5 * np.pi * G * M)


def separation_from_frequency(f_gw_hz, m1_msun, m2_msun):
    """Izračunava orbitalnu separaciju iz frekvencije gravitacijskih valova f_GW
    
    Args:
        f_gw_hz:  Frekvencija gravitacijskih valova [Hz]; može biti skalar ili polje
        m1_msun:  Masa primarne crne rupe [sunčeve mase]
        m2_msun:  Masa sekundarne crne rupe [sunčeve mase]

    Returns:
        Orbitalna separacija [m]; Isti oblik kao f_gw_hz
    """
    M  = total_mass(m1_msun, m2_msun)
    f_orb = f_gw_hz / 2.0
    omega = 2 * np.pi * f_orb
    return (G * M / omega**2) ** (1.0 / 3.0)


def freq_deriv(t, f, Mc):
    """Desna strana ODE: brzina promjene frekvencije GW df/dt [Hz/s]

    Args:
        t:  Vrijeme [s] 
        f:  Trenutna frekvencija GW kao lista ili polje duljine 1 [Hz]
        Mc: Chirp masa [kg]

    Returns:
        df/dt kao numpy polje [Hz/s]
    """
    coeff = (96.0 / 5.0) * np.pi * (np.pi * G * Mc / C**3) ** (5.0 / 3.0) 
    return coeff * np.array(f) ** (11.0 / 3.0)


def evolve_frequency(m1_msun, m2_msun, f0_hz, f_max_hz=None, sample_rate=4096):
    """Integrira Petersovu ODE evolucije frekvencije od f0 do f_isco.

    Args:
        m1_msun:     Masa primarne crne rupe [sunčeve mase]
        m2_msun:     Masa sekundarne crne rupe [sunčeve mase]
        f0_hz:       Početna frekvencija GW [Hz]; za GW150914: 20 Hz
        f_max_hz:    Opcionalna gornja granica frekvencije [Hz]; ako je zadana, integracija staje pri min(f_isco, f_max_hz)
        sample_rate: Izlazna brzina uzorkovanja [Hz]; određuje razmak vremenske mreže

    Returns:
        t: Polje vremena [s], počinje od 0
        f: Polje frekvencije GW [Hz], završava na ili ispod f_isco
    """
    Mc = chirp_mass(m1_msun, m2_msun)
    f_isco = isco_frequency(m1_msun, m2_msun)
    if f_max_hz is not None:
        f_isco = min(f_isco, f_max_hz)

    dt = 1.0 / sample_rate

    pi_G_Mc_over_c3 = np.pi * G * Mc / C**3
    t_coal = (5.0 / (256.0 * np.pi)) * pi_G_Mc_over_c3**(-5.0/3.0) * f0_hz**(-8.0/3.0)

    t_span = (0.0, t_coal * 1.05)
    t_eval = np.arange(0.0, t_coal * 1.02, dt)
    if len(t_eval) == 0:
        t_eval = np.array([0.0, dt])

    def hit_isco(t, f, Mc): return f[0] - f_isco
    hit_isco.terminal  = True
    hit_isco.direction = +1

    sol = solve_ivp(
        freq_deriv, t_span, [f0_hz],
        args=(Mc,), t_eval=t_eval,
        events=hit_isco, max_step=dt * 10,
        rtol=1e-8, atol=1e-10,
    )

    t = sol.t
    f = sol.y[0]

    mask = f <= f_isco * 1.001
    return t[mask], f[mask]


def orbital_separation(t_arr, f_arr, m1_msun, m2_msun):
    """Izračunava orbitalnu separaciju a(t) iz polja frekvencije f(t).

    Args:
        t_arr:   Polje vremena [s] - nekorišteno
        f_arr:   Polje frekvencije GW [Hz]
        m1_msun: Masa primarne crne rupe [sunčeve mase]
        m2_msun: Masa sekundarne crne rupe [sunčeve mase]

    Returns:
        Polje orbitalne separacije [m]; iste duljine kao f_arr
    """
    return separation_from_frequency(f_arr, m1_msun, m2_msun)


def merger_time(m1_msun, m2_msun, a0_m):
    """Analitičko vrijeme do spajanja od početne orbitalne separacije a0

    Args:
        m1_msun: Masa primarne crne rupe [sunčeve mase]
        m2_msun: Masa sekundarne crne rupe [sunčeve mase]
        a0_m:    Početna orbitalna separacija [m]

    Returns:
        Vrijeme do spajanja [s]
    """
    m1 = m1_msun * M_SUN
    m2 = m2_msun * M_SUN
    M  = m1 + m2
    return (5.0 * C**5 / (256.0 * G**3)) * a0_m**4 / (m1 * m2 * M)
