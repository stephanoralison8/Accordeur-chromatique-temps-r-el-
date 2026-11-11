"""Extraction de la fréquence fondamentale à partir d'une trame audio.

Chaîne de traitement :
  1. suppression de la composante continue + seuil de silence (RMS) ;
  2. fenêtrage de Hann (réduit la fuite spectrale) ;
  3. TFD (FFT) avec zero-padding (échantillonnage plus fin du spectre) ;
  4. produit spectral harmonique (HPS) : évite les erreurs d'octave quand
     la fondamentale est plus faible que ses harmoniques ;
  5. interpolation parabolique du pic (précision de l'ordre du centième de bin).
"""
import numpy as np
from .dft import spectre_amplitude

FREQ_MIN = 50.0     # Hz (en dessous du Mi grave d'une basse)
FREQ_MAX = 2000.0   # Hz
SEUIL_RMS = 0.01    # en dessous : silence
NB_HARMONIQUES = 5  # nombre de spectres comprimés dans le HPS


def _interpolation_parabolique(valeurs, i):
    """Affine la position d'un pic (indice fractionnaire) à partir de ses 2 voisins."""
    if i <= 0 or i >= len(valeurs) - 1:
        return float(i)
    a, b, c = valeurs[i - 1], valeurs[i], valeurs[i + 1]
    denom = a - 2.0 * b + c
    if denom == 0:
        return float(i)
    return i + 0.5 * (a - c) / denom


def calculer_spectre(trame, freq_ech, facteur_zero_padding=4):
    """Retourne (frequences, amplitudes) du spectre fenêtré de la trame."""
    trame = np.asarray(trame, dtype=float)
    trame = trame - np.mean(trame)
    fenetre = np.hanning(len(trame))
    n_fft = 1 << int(np.ceil(np.log2(len(trame) * facteur_zero_padding)))
    amplitudes = spectre_amplitude(trame * fenetre, n_fft)
    frequences = np.arange(len(amplitudes)) * freq_ech / n_fft
    return frequences, amplitudes


def produit_spectral_harmonique(amplitudes, nb_harmoniques=NB_HARMONIQUES):
    """HPS : produit du spectre et de ses versions compressées d'un facteur 2, 3, ..."""
    hps = amplitudes.copy()
    for h in range(2, nb_harmoniques + 1):
        comprime = amplitudes[::h]
        hps[: len(comprime)] *= comprime
        hps[len(comprime):] = 0.0
    return hps


def detecter_frequence_fondamentale(trame, freq_ech, seuil_rms=SEUIL_RMS):
    """Retourne la fréquence fondamentale en Hz, ou None si silence / pas de pitch."""
    trame = np.asarray(trame, dtype=float)
    if np.sqrt(np.mean((trame - np.mean(trame)) ** 2)) < seuil_rms:
        return None

    frequences, amplitudes = calculer_spectre(trame, freq_ech)
    pas = frequences[1] - frequences[0]
    hps = produit_spectral_harmonique(amplitudes)

    i_min = int(FREQ_MIN / pas)
    i_max = min(int(FREQ_MAX / pas), len(hps) - 1)
    zone = hps[i_min:i_max]
    if zone.size == 0 or np.max(zone) <= 0:
        return None
    i_pic = i_min + int(np.argmax(zone))

    # Correction d'erreur d'octave : si la sous-octave est nettement présente
    # dans le spectre, on la préfère (la fondamentale est alors à f/2).
    i_demi = i_pic // 2
    if i_demi - 2 >= i_min:
        voisinage = amplitudes[i_demi - 2: i_demi + 3]
        if voisinage.size and np.max(voisinage) > 0.5 * amplitudes[i_pic]:
            i_pic = i_demi - 2 + int(np.argmax(voisinage))

    # Interpolation parabolique sur le spectre (échelle log) pour affiner le pic
    log_spectre = np.log(amplitudes + 1e-12)
    i_affine = _interpolation_parabolique(log_spectre, i_pic)
    return float(i_affine * pas)
