"""
Spektralne analize: FFT, spektralna gustoća snage (Welchova metoda),
spektrogram kratkovremene Fourierove transformacije i integral preklapanja
za usklađeni filtar.  
"""

import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram


def compute_fft(h, sample_rate):
    """Izračunava jednostrani realni FFT vremenskog niza straina.

    Args:
        h:           Vremenski niz straina [bezdimenzionalno], duljina N.
        sample_rate: Brzina uzorkovanja [Hz].

    Returns:
        f:  Polje frekvencija [Hz], duljina N//2 + 1.
        hf: Kompleksni FFT koeficijenti [bezdimenzionalno × s],
            duljina N//2 + 1.
    """
    n  = len(h)
    hf = np.fft.rfft(h)
    f  = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    return f, hf


def compute_spectrogram(h, sample_rate, nperseg=256, noverlap=240, fmin=10, fmax=600):
    """Izračunava spektrogram kratkovremene Fourierove transformacije (STFT) straina.

    Koristi scipy.signal.spectrogram sa skaliranjem spektralne gustoće snage.
    Spektrogram prikazuje kako se frekvencijski sadržaj signala mijenja u
    vremenu — rastuć chirp vidljiv je kao dijagonalni trag od ~20 Hz do
    ~150 Hz.  

    Args:
        h:           Vremenski niz straina [bezdimenzionalno].
        sample_rate: Brzina uzorkovanja [Hz].
        nperseg:     Duljina FFT segmenta [uzorci]. Zadano 256.
                     Kraće → bolja vremenska rezolucija; dulje → bolja
                     frekvencijska rezolucija.
        noverlap:    Preklapanje između segmenata [uzorci]. Zadano 240.
        fmin:        Donja granica frekvencije na izlazu [Hz]. Zadano 10.
        fmax:        Gornja granica frekvencije na izlazu [Hz]. Zadano 600.

    Returns:
        t_s: Polje vremena stupaca spektrograma [s].
        f_s: Polje frekvencija (fmin–fmax) [Hz].
        Sxx: Matrica spektralne gustoće snage [bezdimenzionalno² / Hz],
             oblik (len(f_s), len(t_s)).
    """
    f_s, t_s, Sxx = scipy_spectrogram(
        h, fs=sample_rate, nperseg=nperseg,
        noverlap=noverlap, scaling='density',
    )
    mask = (f_s >= fmin) & (f_s <= fmax)
    return t_s, f_s[mask], Sxx[mask]


def overlap_integral(h1, h2, sample_rate, Sn_func=None):
    """Izračunava šumom vagani integral preklapanja usklađenog filtra <h1|h2>.

    Standardni integral unutarnjeg produkta usklađenog filtra definiran je kao:

        <h1|h2> = 4 Re ∫₀^∞ H1(f) H2*(f) / Sn(f) df

    gdje su H1, H2 jednostrani FFT-ovi, a Sn(f) je jednostrana šumna PSD.
    U odsutnosti modela šuma (Sn_func=None), pretpostavlja se Sn = 1
    (ravni spektar).

    Također izračunava normalizirani match (preklapanje) ∈ [0, 1]:

        match = <h1|h2> / sqrt(<h1|h1> <h2|h2>)

    Match od 1,0 znači da su valni oblici identični do ukupnog mjernog faktora.
    Vidi odjeljak 3.6 rada.

    Args:
        h1:          Prvi vremenski niz straina [bezdimenzionalno].
        h2:          Drugi vremenski niz straina [bezdimenzionalno].
        sample_rate: Brzina uzorkovanja [Hz].
        Sn_func:     Opcionalna funkcija: f → Sn(f) [bezdimenzionalno²/Hz].
                     Ako je None, pretpostavlja se ravni (bijeli) šum.

    Returns:
        inner: Sirovi unutarnji produkt <h1|h2> [bezdimenzionalno].
        match: Normalizirano preklapanje u [0, 1] [bezdimenzionalno].
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

    integrand = np.real(H1 * np.conj(H2)) / Sn          # integrand usklađenog filtra: Re[H1 H2*] / Sn(f)
    inner     = 4.0 * df * np.sum(integrand)              # faktor 4 iz konvencije jednostranog spektra: <h1|h2> = 4 Re ∫ df H1 H2*/Sn

    norm1 = 4.0 * df * np.sum(np.abs(H1)**2 / Sn)
    norm2 = 4.0 * df * np.sum(np.abs(H2)**2 / Sn)

    match = inner / np.sqrt(norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0
    return inner, match
