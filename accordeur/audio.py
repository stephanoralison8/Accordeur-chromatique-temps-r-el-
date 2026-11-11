"""Capture audio temps réel (microphone) avec un tampon circulaire."""
import threading
import time
import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - environnement sans audio
    sd = None


class CaptureAudio:
    """Écoute le micro et conserve les ``taille_trame`` derniers échantillons."""

    def __init__(self, freq_ech=44100, taille_trame=8192, peripherique=None):
        if sd is None:
            raise RuntimeError("Le module 'sounddevice' est indisponible (pip install sounddevice).")
        self.freq_ech = freq_ech
        self.taille_trame = taille_trame
        self._tampon = np.zeros(taille_trame, dtype=np.float32)
        self._verrou = threading.Lock()
        self._flux = sd.InputStream(
            samplerate=freq_ech, channels=1, dtype="float32",
            blocksize=1024, device=peripherique, callback=self._rappel,
        )

    def _rappel(self, donnees, nb_images, temps, statut):
        bloc = donnees[:, 0]
        with self._verrou:
            self._tampon = np.roll(self._tampon, -len(bloc))
            self._tampon[-len(bloc):] = bloc

    def demarrer(self):
        self._flux.start()

    def arreter(self):
        self._flux.stop()
        self._flux.close()

    def derniere_trame(self):
        with self._verrou:
            return self._tampon.copy()


class SourceSimulee:
    """Source de test : joue une suite de notes avec harmoniques et bruit (sans micro)."""

    def __init__(self, freq_ech=44100, taille_trame=8192, frequences=None, duree_note=3.0):
        self.freq_ech = freq_ech
        self.taille_trame = taille_trame
        self.frequences = frequences or [82.41, 110.0, 146.83, 196.0, 246.94, 329.63, 442.0, 261.63]
        self.duree_note = duree_note
        self._t0 = None

    def demarrer(self):
        self._t0 = time.time()

    def arreter(self):
        pass

    def derniere_trame(self):
        t_fin = time.time() - self._t0
        index = int(t_fin / self.duree_note) % len(self.frequences)
        f0 = self.frequences[index]
        t = t_fin - np.arange(self.taille_trame)[::-1] / self.freq_ech
        signal = sum((1.0 / h) * np.sin(2 * np.pi * h * f0 * t) for h in range(1, 5))
        signal = 0.3 * signal / np.max(np.abs(signal))
        return (signal + 0.005 * np.random.randn(self.taille_trame)).astype(np.float32)


def lire_fichier_wav(chemin):
    """Charge un WAV (1er canal) -> (signal float dans [-1, 1], freq_ech)."""
    import wave
    with wave.open(chemin, "rb") as w:
        nb_canaux, largeur, fe, nb = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        brut = w.readframes(nb)
    if largeur == 1:
        donnees = (np.frombuffer(brut, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif largeur == 2:
        donnees = np.frombuffer(brut, dtype=np.int16).astype(np.float64) / 32768.0
    elif largeur == 4:
        donnees = np.frombuffer(brut, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError("Format WAV non supporté (8, 16 ou 32 bits).")
    return donnees.reshape(-1, nb_canaux)[:, 0], fe
