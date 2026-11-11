import numpy as np
import pytest
from accordeur.dft import tfd_directe, tfd_rapide
from accordeur.notes import analyser_note
from accordeur.detection import detecter_frequence_fondamentale

FE = 44100
N = 8192


def synthese(f0, harmoniques=(1.0, 0.5, 0.33, 0.25), bruit=0.005, amplitude_fond=1.0):
    t = np.arange(N) / FE
    poids = list(harmoniques)
    poids[0] *= amplitude_fond
    s = sum(a * np.sin(2 * np.pi * (h + 1) * f0 * t) for h, a in enumerate(poids))
    s = 0.3 * s / np.max(np.abs(s))
    return s + bruit * np.random.default_rng(0).standard_normal(N)


def test_tfd_rapide_egale_tfd_directe():
    x = np.random.default_rng(1).standard_normal(256)
    assert np.allclose(tfd_directe(x), tfd_rapide(x))
    assert np.allclose(np.fft.fft(x), tfd_rapide(x))


def test_tfd_longueur_invalide():
    with pytest.raises(ValueError):
        tfd_rapide(np.zeros(100))


@pytest.mark.parametrize("f0", [82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 440.0, 523.25, 880.0])
def test_detection_precision(f0):
    f = detecter_frequence_fondamentale(synthese(f0), FE)
    assert f is not None
    assert abs(1200 * np.log2(f / f0)) < 3.0  # moins de 3 cents


def test_fondamentale_faible_pas_d_erreur_octave():
    f0 = 196.0
    f = detecter_frequence_fondamentale(synthese(f0, amplitude_fond=0.3), FE)
    assert abs(1200 * np.log2(f / f0)) < 5.0


def test_silence():
    assert detecter_frequence_fondamentale(np.zeros(N), FE) is None


def test_analyse_note():
    nom, octave, f_th, cents = analyser_note(440.0)
    assert (nom, octave) == ("La", 4) and abs(cents) < 1e-9
    nom, octave, _, cents = analyser_note(442.0)
    assert nom == "La" and 7 < cents < 9
    assert analyser_note(261.63)[:2] == ("Do", 4)
