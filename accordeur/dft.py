"""Transformée de Fourier Discrète (TFD) : implémentations pédagogiques.

Définition (N échantillons x[n]) :

    X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n / N),   k = 0..N-1

Deux implémentations sont fournies :
  * ``tfd_directe``  : application littérale de la formule, complexité O(N^2) ;
  * ``tfd_rapide``   : algorithme de Cooley-Tukey radix-2 (FFT), complexité O(N log N).
C'est ``tfd_rapide`` qui est utilisée en temps réel.
"""
import numpy as np


def tfd_directe(x):
    """TFD par produit matriciel (O(N^2)). Réservée aux petits signaux / aux tests."""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    matrice_fourier = np.exp(-2j * np.pi * k * n / N)
    return matrice_fourier @ x


def tfd_rapide(x):
    """FFT radix-2 itérative (Cooley-Tukey). La longueur doit être une puissance de 2."""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    if N == 0 or N & (N - 1) != 0:
        raise ValueError("La longueur doit être une puissance de 2.")

    # 1) Permutation « bit-reversal »
    nb_bits = N.bit_length() - 1
    indices = np.arange(N)
    inverses = np.zeros(N, dtype=int)
    for b in range(nb_bits):
        inverses |= ((indices >> b) & 1) << (nb_bits - 1 - b)
    X = x[inverses].copy()

    # 2) Papillons étage par étage
    taille = 2
    while taille <= N:
        moitie = taille // 2
        twiddle = np.exp(-2j * np.pi * np.arange(moitie) / taille)
        X = X.reshape(-1, taille)
        pair = X[:, :moitie].copy()
        impair = X[:, moitie:] * twiddle
        X[:, :moitie] = pair + impair
        X[:, moitie:] = pair - impair
        X = X.reshape(-1)
        taille *= 2
    return X


def spectre_amplitude(signal, n_fft):
    """Spectre d'amplitude unilatéral (fréquences 0..Fe/2) avec zero-padding à n_fft."""
    signal = np.asarray(signal, dtype=float)
    if n_fft > signal.shape[0]:
        signal = np.concatenate([signal, np.zeros(n_fft - signal.shape[0])])
    X = tfd_rapide(signal)
    return np.abs(X[: n_fft // 2 + 1])
