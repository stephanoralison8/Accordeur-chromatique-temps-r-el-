"""Accordeur chromatique temps réel (TFD) — point d'entrée.

Utilisation :
    python main.py                  # micro, interface graphique
    python main.py --demo           # source simulée (sans micro)
    python main.py --fichier a.wav  # analyse d'un fichier WAV (console)
    python main.py --console        # micro, affichage texte uniquement
    python main.py --capture sortie.png [--demo]   # enregistre une capture de l'interface
"""
import argparse
import sys
import time
import numpy as np

from accordeur.audio import CaptureAudio, SourceSimulee, lire_fichier_wav
from accordeur.detection import calculer_spectre, detecter_frequence_fondamentale
from accordeur.notes import analyser_note

FREQ_ECH = 44100
TAILLE_TRAME = 8192          # ~186 ms : compromis réactivité / résolution
LISSAGE = 5                  # médiane glissante sur les dernières détections


class Lisseur:
    """Médiane glissante : stabilise l'affichage et rejette les valeurs aberrantes."""

    def __init__(self, taille=LISSAGE):
        self.historique = []
        self.taille = taille

    def ajouter(self, valeur):
        if valeur is None:
            self.historique.clear()
            return None
        self.historique.append(valeur)
        self.historique = self.historique[-self.taille:]
        return float(np.median(self.historique))


def analyser_trame(trame, lisseur):
    f = lisseur.ajouter(detecter_frequence_fondamentale(trame, FREQ_ECH))
    if f is None:
        return None
    nom, octave, f_th, cents = analyser_note(f)
    return {"f": f, "nom": nom, "octave": octave, "f_th": f_th, "cents": cents}


def mode_console(source):
    lisseur = Lisseur()
    source.demarrer()
    print("Écoute en cours (Ctrl+C pour arrêter)...")
    try:
        while True:
            time.sleep(0.2)
            r = analyser_trame(source.derniere_trame(), lisseur)
            if r is None:
                print("\r(silence)                                   ", end="")
            else:
                print(f"\r{r['nom']}{r['octave']}  {r['f']:7.2f} Hz  ({r['cents']:+5.1f} cents)      ", end="")
    except KeyboardInterrupt:
        pass
    finally:
        source.arreter()


def mode_fichier(chemin):
    signal, fe = lire_fichier_wav(chemin)
    lisseur = Lisseur()
    pas = TAILLE_TRAME // 2
    print(f"{chemin} : {len(signal) / fe:.2f} s à {fe} Hz")
    for debut in range(0, len(signal) - TAILLE_TRAME, pas):
        trame = signal[debut:debut + TAILLE_TRAME]
        f = detecter_frequence_fondamentale(trame, fe)
        if f is None:
            continue
        nom, octave, _, cents = analyser_note(f)
        print(f"t={debut / fe:6.2f}s  {nom}{octave:<2d} {f:8.2f} Hz  {cents:+6.1f} cents")


def construire_interface(source, capture=None):
    import matplotlib
    if capture:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig = plt.figure(figsize=(11, 6.5))
    fig.canvas.manager.set_window_title("Accordeur chromatique - TFD") if not capture else None
    grille = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.45)
    ax_spectre = fig.add_subplot(grille[0])
    ax_jauge = fig.add_subplot(grille[1])

    # --- Spectre fréquentiel ---
    ax_spectre.set_title("Spectre d'amplitude (TFD de la trame, fenêtre de Hann)")
    ax_spectre.set_xlabel("Fréquence (Hz)")
    ax_spectre.set_ylabel("Amplitude")
    ax_spectre.set_xlim(0, 2000)
    (courbe,) = ax_spectre.plot([], [], lw=1.0, color="tab:blue")
    ligne_f0 = ax_spectre.axvline(0, color="tab:red", ls="--", lw=1.2, visible=False)
    texte_f0 = ax_spectre.text(0.98, 0.92, "", transform=ax_spectre.transAxes,
                               ha="right", va="top", fontsize=11, color="tab:red")
    ax_spectre.grid(alpha=0.3)

    # --- Jauge d'accordage ---
    ax_jauge.set_xlim(-50, 50)
    ax_jauge.set_ylim(0, 1)
    ax_jauge.set_yticks([])
    ax_jauge.set_xlabel("Écart à la note juste (cents)")
    ax_jauge.axvspan(-5, 5, color="tab:green", alpha=0.25)
    for c in range(-50, 51, 10):
        ax_jauge.axvline(c, color="gray", lw=0.6, alpha=0.5)
    aiguille = ax_jauge.axvline(0, color="black", lw=3, visible=False)
    texte_note = ax_jauge.text(0, 0.55, "--", ha="center", va="center", fontsize=40, weight="bold")
    texte_info = ax_jauge.text(0, 0.12, "", ha="center", va="center", fontsize=12)

    lisseur = Lisseur()
    source.demarrer()

    def maj(_):
        trame = source.derniere_trame()
        r = analyser_trame(trame, lisseur)
        freqs, amps = calculer_spectre(trame, FREQ_ECH)
        masque = freqs <= 2000
        courbe.set_data(freqs[masque], amps[masque])
        ax_spectre.set_ylim(0, max(1e-6, amps[masque].max()) * 1.1)
        if r is None:
            ligne_f0.set_visible(False)
            aiguille.set_visible(False)
            texte_f0.set_text("")
            texte_note.set_text("--")
            texte_note.set_color("gray")
            texte_info.set_text("En attente d'un son...")
        else:
            ligne_f0.set_xdata([r["f"], r["f"]])
            ligne_f0.set_visible(True)
            texte_f0.set_text(f"f0 = {r['f']:.2f} Hz")
            cents = max(-50, min(50, r["cents"]))
            aiguille.set_xdata([cents, cents])
            aiguille.set_visible(True)
            juste = abs(r["cents"]) <= 5
            texte_note.set_text(f"{r['nom']}{r['octave']}")
            texte_note.set_color("tab:green" if juste else "tab:orange")
            sens = "juste" if juste else ("trop grave : serrer" if r["cents"] < 0 else "trop aigu : desserrer")
            texte_info.set_text(f"{r['cents']:+.1f} cents  |  cible {r['f_th']:.2f} Hz  |  {sens}")
        return courbe, ligne_f0, aiguille

    if capture:
        # Alimente le lisseur avec quelques trames puis sauvegarde l'image
        for _ in range(LISSAGE):
            maj(None)
        fig.savefig(capture, dpi=130, bbox_inches="tight")
        source.arreter()
        print(f"Capture enregistrée : {capture}")
        return
    anim = FuncAnimation(fig, maj, interval=80, blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        source.arreter()
    return anim


def main():
    p = argparse.ArgumentParser(description="Accordeur chromatique temps réel (TFD)")
    p.add_argument("--demo", action="store_true", help="source simulée (sans micro)")
    p.add_argument("--console", action="store_true", help="affichage texte uniquement")
    p.add_argument("--fichier", help="analyser un fichier WAV")
    p.add_argument("--capture", help="enregistrer une capture PNG de l'interface")
    p.add_argument("--note", type=float, help="avec --demo/--capture : fréquence simulée unique (Hz)")
    p.add_argument("--peripherique", type=int, help="indice du périphérique d'entrée")
    a = p.parse_args()

    if a.fichier:
        return mode_fichier(a.fichier)

    if a.demo or a.capture:
        freqs = [a.note] if a.note else None
        source = SourceSimulee(FREQ_ECH, TAILLE_TRAME, frequences=freqs)
    else:
        try:
            source = CaptureAudio(FREQ_ECH, TAILLE_TRAME, a.peripherique)
        except Exception as e:
            print(f"Impossible d'ouvrir le microphone : {e}\nEssayez : python main.py --demo", file=sys.stderr)
            return 1

    if a.console:
        return mode_console(source)
    construire_interface(source, a.capture)


if __name__ == "__main__":
    sys.exit(main())
