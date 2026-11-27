"""
Spektralne analize
"""

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram


def compute_fft(h, sample_rate):
    """Izračunava jednostrani realni FFT vremenskog niza straina

    Args:
        h:           Vremenski niz straina [0-bimenzionalno]; duljina N
        sample_rate: Brzina uzorkovanja [Hz]

    Returns:
        f:  Polje frekvencija [Hz]; duljina N//2 + 1
        hf: Kompleksni FFT koeficijenti [bezdimenzionalno × s]; duljina N//2 + 1.
    """
    n  = len(h)
    hf = np.fft.rfft(h)
    f  = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return f, hf


def compute_spectrogram(h, sample_rate, nperseg=256, noverlap=240, fmin=10, fmax=600):
    """Izračunava spektrogram kratkovremene Fourierove transformacije (STFT) straina

    Args:
        h:           Vremenski niz straina [0-bimenzionalno]
        sample_rate: Brzina uzorkovanja [Hz]
        nperseg:     Duljina FFT segmenta [uzorci]; zadano 256
                     Kraće -> bolja vremenska rezolucija; dulje -> bolja frekvencijska rezolucija
        noverlap:    Preklapanje između segmenata [uzorci]; zadano 240
        fmin:        Donja granica frekvencije na izlazu [Hz]; zadano 10
        fmax:        Gornja granica frekvencije na izlazu [Hz]; zadano 600

    Returns:
        t_s: Polje vremena stupaca spektrograma [s]
        f_s: Polje frekvencija (fmin–fmax) [Hz]
        Sxx: Matrica spektralne gustoće snage [0-dimenzionalno^2 / Hz]; oblik (len(f_s), len(t_s)).
    """
    f_s, t_s, Sxx = scipy_spectrogram(
        h, fs=sample_rate, nperseg=nperseg,
        noverlap=noverlap, scaling='density',
    )
    mask = (f_s >= fmin) & (f_s <= fmax)
    return t_s, f_s[mask], Sxx[mask]


def overlap_integral(h1, h2, sample_rate, Sn_func=None):
    """Izračunava šumom vagani integral preklapanja usklađenog filtra <h1|h2>

    Args:
        h1:          Prvi vremenski niz straina [0-dimenzionalno]
        h2:          Drugi vremenski niz straina [0-dimenzionalno]
        sample_rate: Brzina uzorkovanja [Hz]
        Sn_func:     Opcionalna funkcija: f → Sn(f) [0-dimenzionalno^2 / Hz]
                     Ako je None, pretpostavlja se ravni (bijeli) šum

    Returns:
        inner: Sirovi unutarnji produkt <h1|h2> [0-dimenzionalno]
        match: Normalizirano preklapanje u [0, 1] [0-dimenzionalno]
    """
    n  = min(len(h1), len(h2))
    h1 = h1[:n];  h2 = h2[:n]

    df = sample_rate / n
    f  = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    H1 = np.fft.rfft(h1)
    H2 = np.fft.rfft(h2)

    if Sn_func is None:
        Sn = np.ones_like(f)
    else:
        Sn = Sn_func(f)
        Sn = np.where(Sn > 0, Sn, 1e-100)

    integrand = np.real(H1 * np.conj(H2)) / Sn         
    inner     = 4.0 * df * np.sum(integrand)             

    norm1 = 4.0 * df * np.sum(np.abs(H1)**2 / Sn)
    norm2 = 4.0 * df * np.sum(np.abs(H2)**2 / Sn)

    match = inner / np.sqrt(norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0
    return inner, match
