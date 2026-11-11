"""Génère les figures (docs/figures) et le rapport PDF (docs/Rapport_Accordeur_TFD.pdf)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from accordeur.detection import (calculer_spectre, produit_spectral_harmonique,
                                 detecter_frequence_fondamentale)
from accordeur.dft import tfd_directe, tfd_rapide
from accordeur.notes import analyser_note

FE, N = 44100, 8192
DOSSIER = os.path.join(os.path.dirname(__file__), "docs")
FIG = os.path.join(DOSSIER, "figures")
os.makedirs(FIG, exist_ok=True)

NOM = "RALISON Andriniaina Stephano Michel"


def signal_test(f0, poids=(0.35, 1.0, 0.6, 0.4, 0.25), bruit=0.01, graine=0):
    t = np.arange(N) / FE
    s = sum(a * np.sin(2 * np.pi * (h + 1) * f0 * t) for h, a in enumerate(poids))
    s = 0.3 * s / np.max(np.abs(s))
    return s + bruit * np.random.default_rng(graine).standard_normal(N)


def figure_spectre():
    f0 = 220.0
    x = signal_test(f0)
    freqs, amps = calculer_spectre(x, FE)
    f_det = detecter_frequence_fondamentale(x, FE)
    t = np.arange(N) / FE
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 6.2), gridspec_kw={"hspace": 0.45})
    m = t < 0.03
    a1.plot(t[m] * 1000, x[m], lw=1)
    a1.set_title("Signal temporel (La3 = 220 Hz, 30 premières ms)")
    a1.set_xlabel("Temps (ms)"); a1.set_ylabel("Amplitude"); a1.grid(alpha=.3)
    mf = freqs <= 1400
    a2.plot(freqs[mf], amps[mf], lw=1)
    a2.axvline(f_det, color="r", ls="--", lw=1, label=f"fondamentale détectée : {f_det:.2f} Hz")
    for h in range(2, 6):
        a2.axvline(h * f_det, color="gray", ls=":", lw=0.8)
        a2.text(h * f_det, a2.get_ylim()[1] * 0.95, f"{h}·f0", ha="center", fontsize=8, color="gray")
    a2.set_title("Spectre d'amplitude (TFD, fenêtre de Hann, zero-padding x4)")
    a2.set_xlabel("Fréquence (Hz)"); a2.set_ylabel("|X(f)|"); a2.legend(loc="upper right"); a2.grid(alpha=.3)
    fig.savefig(os.path.join(FIG, "spectre_frequentiel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_hps():
    x = signal_test(220.0)
    freqs, amps = calculer_spectre(x, FE)
    hps = produit_spectral_harmonique(amps)
    mf = freqs <= 1000
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5), gridspec_kw={"hspace": 0.5})
    a1.plot(freqs[mf], amps[mf], lw=1); a1.set_title("Spectre brut : le 2e harmonique (440 Hz) domine")
    a1.set_xlabel("Fréquence (Hz)"); a1.grid(alpha=.3)
    a2.plot(freqs[mf], hps[mf] / hps[mf].max(), lw=1, color="tab:green")
    a2.set_title("Produit spectral harmonique (HPS) : le pic à 220 Hz ressort")
    a2.set_xlabel("Fréquence (Hz)"); a2.grid(alpha=.3)
    fig.savefig(os.path.join(FIG, "hps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_fenetre():
    t = np.arange(N) / FE
    x = np.sin(2 * np.pi * 230.0 * t)
    n_fft = 1 << 15
    res = {}
    for nom, w in (("Rectangulaire", np.ones(N)), ("Hann", np.hanning(N))):
        X = np.abs(tfd_rapide(np.concatenate([x * w, np.zeros(n_fft - N)]))[: n_fft // 2 + 1])
        res[nom] = 20 * np.log10(X / X.max() + 1e-12)
    f = np.arange(n_fft // 2 + 1) * FE / n_fft
    m = (f > 100) & (f < 400)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for nom, v in res.items():
        ax.plot(f[m], v[m], lw=1, label=nom)
    ax.set_ylim(-100, 3); ax.set_xlabel("Fréquence (Hz)"); ax.set_ylabel("Amplitude (dB)")
    ax.set_title("Fuite spectrale : fenêtre rectangulaire vs fenêtre de Hann"); ax.grid(alpha=.3); ax.legend()
    fig.savefig(os.path.join(FIG, "fenetrage.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def tableau_precision():
    lignes = [["Note", "f théorique (Hz)", "f détectée (Hz)", "Erreur (cents)"]]
    for f0, nom in [(82.41, "Mi2"), (110.0, "La2"), (146.83, "Ré3"), (196.0, "Sol3"), (246.94, "Si3"),
                    (329.63, "Mi4"), (440.0, "La4"), (523.25, "Do5"), (880.0, "La5")]:
        f = detecter_frequence_fondamentale(signal_test(f0), FE)
        lignes.append([nom, f"{f0:.2f}", f"{f:.2f}", f"{1200 * np.log2(f / f0):+.2f}"])
    return lignes


def verifier_tfd():
    x = np.random.default_rng(1).standard_normal(512)
    return float(np.max(np.abs(tfd_directe(x) - tfd_rapide(x))))


def construire_pdf(chemin):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
                                    Table, TableStyle, Preformatted)

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib
    dossier_polices = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for nom, fichier in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"),
                         ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf")):
        pdfmetrics.registerFont(TTFont(nom, os.path.join(dossier_polices, fichier)))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
    st = getSampleStyleSheet()
    for nom_style in ("Normal", "Heading1", "Heading2", "Title", "Italic"):
        st[nom_style].fontName = "DV-B" if nom_style.startswith(("Heading", "Title")) else ("DV-I" if nom_style == "Italic" else "DV")
    corps = ParagraphStyle("corps", parent=st["Normal"], fontSize=10, leading=14.5, alignment=4)
    h1 = ParagraphStyle("h1", parent=st["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=8, keepWithNext=1)
    h2 = ParagraphStyle("h2", parent=st["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, keepWithNext=1)
    leg = ParagraphStyle("leg", parent=st["Italic"], fontSize=9, alignment=1, textColor=colors.grey)
    code = ParagraphStyle("code", parent=st["Code"], fontSize=8.5, leading=11, backColor=colors.whitesmoke,
                          borderPadding=4)
    titre = ParagraphStyle("titre", parent=st["Title"], fontSize=22, leading=28)
    centre = ParagraphStyle("centre", parent=st["Normal"], alignment=1, fontSize=12, leading=18)

    def img(nom, largeur=15.5, dossier=FIG):
        p = os.path.join(dossier, nom)
        from PIL import Image as PI
        w, h = PI.open(p).size
        return Image(p, width=largeur * cm, height=largeur * cm * h / w)

    E = []
    # --- Page de garde ---
    E += [Spacer(1, 4 * cm), Paragraph("Examen — Codage et traitement du signal", centre), Spacer(1, 1 * cm),
          Paragraph("Détection de notes et accordeur chromatique en temps réel", titre),
          Paragraph("basés sur la Transformée de Fourier Discrète", centre), Spacer(1, 3 * cm),
          Paragraph(f"<b>{NOM}</b>", centre), Paragraph("L2 Informatique", centre), Spacer(1, 1 * cm),
          Paragraph("Date de rendu : 10 novembre 2026", centre), PageBreak()]

    # --- 1. Introduction ---
    E += [Paragraph("1. Objectif et architecture", h1),
          Paragraph("Le projet consiste à capter un son (voix, guitare, etc.) par le microphone, à en extraire la "
                    "<b>fréquence fondamentale</b> grâce à la Transformée de Fourier Discrète (TFD), puis à "
                    "afficher la note la plus proche de la gamme tempérée ainsi que l'écart d'accordage en cents, "
                    "à la manière d'un accordeur chromatique. Le traitement est réalisé en temps réel, trame "
                    "après trame, en Python (NumPy, Matplotlib, sounddevice).", corps),
          Paragraph("Architecture du code", h2),
          Preformatted(
              "accordeur/dft.py        TFD directe O(N^2) et FFT radix-2 (Cooley-Tukey) écrites à la main\n"
              "accordeur/detection.py  fenêtrage, zero-padding, HPS, interpolation, extraction de f0\n"
              "accordeur/notes.py      fréquence -> note (nom, octave, écart en cents)\n"
              "accordeur/audio.py      capture micro (tampon circulaire), source simulée, lecture WAV\n"
              "main.py                 interface temps réel (spectre + jauge), modes console / fichier\n"
              "tests/                  tests unitaires (pytest)\n"
              "generer_rapport.py      figures et rapport PDF", code),
          Spacer(1, 6),
          Paragraph("Chaîne de traitement : <b>micro</b> → trame de 8192 échantillons (Fe = 44100 Hz, soit "
                    "186 ms) → suppression de la composante continue → fenêtre de Hann → zero-padding "
                    "(x4, soit 32768 points) → FFT → HPS → interpolation parabolique → conversion en note → "
                    "lissage par médiane glissante → affichage.", corps)]

    # --- 2. Théorie ---
    E += [Paragraph("2. Transformée de Fourier Discrète", h1),
          Paragraph("Pour un signal échantillonné x[n], n = 0..N-1, la TFD est définie par : "
                    "<b>X[k] = Σ x[n]·e<super>-2iπkn/N</super></b>, k = 0..N-1. "
                    "Le coefficient X[k] mesure la présence de la fréquence f<sub>k</sub> = k·Fe/N dans le signal. "
                    "Pour un signal réel, le spectre est symétrique ; on ne conserve que les fréquences de 0 à "
                    "Fe/2 (théorème de Shannon-Nyquist).", corps),
          Paragraph("La TFD directe coûte O(N²) opérations. L'algorithme de <b>FFT de Cooley-Tukey</b> "
                    "(radix-2) exploite la décomposition récursive en indices pairs et impairs pour atteindre "
                    "O(N log N) : pour N = 32768, cela représente environ 500 000 opérations de papillons "
                    "au lieu d'un milliard, ce qui rend le temps réel possible. Les deux versions sont "
                    f"implémentées dans <font face='Courier'>dft.py</font> ; leur écart maximal sur un signal "
                    f"aléatoire de 512 points est de {verifier_tfd():.1e} (erreurs d'arrondi uniquement).", corps),
          Paragraph("Résolution fréquentielle et compromis", h2),
          Paragraph("La résolution naturelle de la TFD est Fe/N = 44100/8192 ≈ 5,4 Hz. Elle est insuffisante "
                    "pour distinguer deux notes graves (le Mi2 à 82,41 Hz et le Fa2 à 87,31 Hz sont séparés de "
                    "moins de 5 Hz) avec la précision d'un accordeur (quelques cents). Trois techniques "
                    "permettent de l'améliorer sans allonger la trame (donc sans perdre en réactivité) :", corps),
          Paragraph("• <b>Fenêtre de Hann</b> : un signal n'est jamais périodique sur exactement N échantillons, "
                    "ce qui provoque une <i>fuite spectrale</i> (figure 3). La fenêtre de Hann réduit "
                    "fortement les lobes secondaires.<br/>"
                    "• <b>Zero-padding</b> : ajouter des zéros avant la FFT interpole le spectre (pas "
                    "fréquentiel réduit à ≈ 1,35 Hz) sans ajouter d'information, mais facilite la localisation du pic.<br/>"
                    "• <b>Interpolation parabolique</b> : on ajuste une parabole sur le logarithme du spectre "
                    "autour du pic (3 points) ; l'abscisse du sommet donne la fréquence avec une précision "
                    "très inférieure au pas de la TFD.", corps)]

    # --- 3. Extraction f0 ---
    E += [PageBreak(), Paragraph("3. Extraction de la fréquence fondamentale", h1),
          Paragraph("La fréquence fondamentale f0 est la plus basse fréquence du son périodique ; l'oreille "
                    "perçoit la hauteur de la note correspondante. Dans un instrument réel, le spectre contient "
                    "f0 et ses <b>harmoniques</b> (2·f0, 3·f0, …). Le pic le plus haut n'est pas toujours "
                    "la fondamentale (figure 2 : à 220 Hz, le deuxième harmonique domine), ce qui conduirait à "
                    "une <b>erreur d'octave</b> si l'on prenait simplement le maximum du spectre.", corps),
          Spacer(1, 6), img("spectre_frequentiel.png", 14.5),
          Paragraph("Figure 1 — Signal temporel et diagramme du spectre fréquentiel d'un La3 (220 Hz) avec harmoniques.", leg),
          Paragraph("Algorithme utilisé (<font face='Courier'>detection.py</font>)", h2),
          Paragraph("1. <b>Seuil de silence</b> : si la valeur efficace (RMS) de la trame est inférieure à 0,01, "
                    "aucune note n'est affichée.<br/>"
                    "2. <b>Prétraitement</b> : suppression de la moyenne, multiplication par la fenêtre de Hann, "
                    "zero-padding à 4N points.<br/>"
                    "3. <b>TFD (FFT)</b> et calcul du module |X[k]|.<br/>"
                    "4. <b>Produit spectral harmonique (HPS)</b> : HPS[k] = Π<sub>h=1..5</sub> |X[h·k]|. "
                    "Si f0 est la fondamentale, ses harmoniques 2f0, 3f0, … alignent leurs pics sur h·k : le "
                    "produit présente alors un maximum net en k0 (figure 2).<br/>"
                    "5. <b>Recherche du maximum</b> du HPS entre 50 Hz et 2000 Hz, puis vérification de la "
                    "sous-octave pour corriger une éventuelle erreur d'octave.<br/>"
                    "6. <b>Interpolation parabolique</b> du pic sur le spectre en échelle logarithmique : "
                    "k* = k + ½·(a − c)/(a − 2b + c), où a, b, c sont les valeurs en k−1, k, k+1. "
                    "La fréquence est f0 = k*·Fe/N_fft.", corps),
          Spacer(1, 6), img("hps.png", 13.5),
          Paragraph("Figure 2 — Spectre brut et produit spectral harmonique pour un La3.", leg),
          img("fenetrage.png", 13.5),
          Paragraph("Figure 3 — Effet du fenêtrage sur la fuite spectrale (sinusoïde de 230 Hz).", leg)]

    # --- 4. Notes ---
    E += [Paragraph("4. Conversion en note de musique", h1),
          Paragraph("Dans la gamme tempérée, deux demi-tons consécutifs sont dans le rapport 2<super>1/12</super>. "
                    "Avec La4 = 440 Hz, le numéro MIDI d'une fréquence f est "
                    "<b>m = 69 + 12·log<sub>2</sub>(f/440)</b>. La note la plus proche est obtenue en arrondissant m ; "
                    "son nom est donné par m mod 12 (Do, Do#, Ré, …) et son octave par ⌊m/12⌋ − 1. "
                    "L'écart d'accordage en <b>cents</b> (1 demi-ton = 100 cents) vaut "
                    "<b>1200·log<sub>2</sub>(f / f<sub>théorique</sub>)</b>. L'accordeur affiche la note en vert "
                    "lorsque |écart| ≤ 5 cents.", corps),
          Paragraph("5. Interface temps réel", h1),
          Paragraph("Le microphone est lu par un flux <font face='Courier'>sounddevice</font> (blocs de 1024 "
                    "échantillons) qui alimente un tampon circulaire des 8192 derniers échantillons. Toutes les "
                    "80 ms, la dernière trame est analysée et l'interface Matplotlib est rafraîchie : "
                    "spectre en haut, note détectée et jauge en cents en bas. Une médiane glissante sur les "
                    "5 dernières détections stabilise l'affichage. Les captures ci-dessous ont été obtenues "
                    "avec l'application en mode source simulée (<font face='Courier'>python main.py --capture ... --demo</font>) : "
                    "un instrument virtuel avec harmoniques et bruit, ce qui rend les mesures reproductibles.", corps)]

    caps = os.path.join(DOSSIER, "captures")
    for fichier, texte in [("capture_demo_La3.png", "Capture 1 — Détection d'un La3 (220 Hz) accordé."),
                           ("capture_demo_Mi2.png", "Capture 2 — Détection d'un Mi2 (82,41 Hz), corde grave de guitare."),
                           ("capture_demo_La4_desaccorde.png", "Capture 3 — La4 désaccordé (446 Hz, +23,5 cents) : « trop aigu »."),]:
        E += [Spacer(1, 6), img(fichier, 12.0, caps), Paragraph(texte, leg)]

    # --- 6. Résultats ---
    tab = tableau_precision()
    T = Table(tab, hAlign="CENTER", colWidths=[3 * cm, 4 * cm, 4 * cm, 3.5 * cm])
    T.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                           ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.whitesmoke])]))
    E += [PageBreak(), Paragraph("6. Validation", h1),
          Paragraph("Des signaux synthétiques (fondamentale + 4 harmoniques, bruit gaussien) ont été analysés. "
                    "L'erreur reste inférieure à quelques cents sur toute la tessiture d'une guitare et "
                    "d'une voix, même lorsque la fondamentale est plus faible que le 2e harmonique "
                    "(cas du La3 testé ici).", corps), Spacer(1, 8), T,
          Paragraph("Tableau 1 — Précision de la détection (signaux synthétiques bruités).", leg),
          Paragraph("Les 14 tests unitaires (<font face='Courier'>pytest</font>) vérifient : l'égalité entre "
                    "la TFD directe, la FFT maison et <font face='Courier'>numpy.fft</font> ; le rejet des "
                    "longueurs invalides ; l'erreur de moins de 3 cents sur 9 notes ; l'absence d'erreur "
                    "d'octave avec fondamentale atténuée ; la gestion du silence ; la conversion en notes.", corps),
          Paragraph("7. Limites et perspectives", h1),
          Paragraph("• La résolution reste limitée pour les notes très graves (< 60 Hz) avec une trame de 186 ms ; "
                    "une trame plus longue améliore la précision au prix de la réactivité.<br/>"
                    "• Les sons polyphoniques (accords) ne sont pas traités : HPS suppose une seule fondamentale.<br/>"
                    "• Des méthodes temporelles (autocorrélation, YIN) pourraient compléter l'approche spectrale.<br/>"
                    "• Le diapason (La4 = 440 Hz) est un paramètre modifiable dans <font face='Courier'>notes.py</font>.", corps),
          Paragraph("8. Utilisation", h1),
          Preformatted("pip install -r requirements.txt\n"
                       "python main.py                 # micro + interface graphique\n"
                       "python main.py --demo          # source simulée\n"
                       "python main.py --console       # affichage texte\n"
                       "python main.py --fichier x.wav # analyse d'un WAV\n"
                       "python -m pytest               # tests", code)]

    doc = SimpleDocTemplate(chemin, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="Accordeur chromatique temps réel (TFD)", author=NOM)
    doc.build(E)


if __name__ == "__main__":
    figure_spectre(); figure_hps(); figure_fenetre()
    construire_pdf(os.path.join(DOSSIER, "Rapport_Accordeur_TFD.pdf"))
    print("Rapport généré :", os.path.join(DOSSIER, "Rapport_Accordeur_TFD.pdf"))
