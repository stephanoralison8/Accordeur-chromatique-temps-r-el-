# Accordeur chromatique temps réel (Transformée de Fourier Discrète)

Examen « Codage et traitement du signal » — détection de notes et accordeur chromatique en temps réel.

**Auteur :** RALISON Andriniaina Stephano Michel (L2 Informatique)

Le programme capte le son du microphone, calcule la TFD (FFT radix-2 écrite à la main dans
`accordeur/dft.py`), extrait la **fréquence fondamentale** (fenêtre de Hann, zero-padding,
produit spectral harmonique, interpolation parabolique) puis affiche la note la plus proche
et l'écart en cents.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

```bash
python main.py                    # micro + interface graphique (spectre + jauge)
python main.py --demo             # source simulée, sans micro
python main.py --console          # affichage texte
python main.py --fichier x.wav    # analyse d'un fichier WAV
python main.py --peripherique 1   # choisir l'entrée audio (voir sounddevice.query_devices())
python main.py --capture img.png  # enregistre une capture de l'interface
python -m pytest                  # tests unitaires
python generer_rapport.py         # régénère les figures et le rapport PDF
```

## Structure

| Fichier | Rôle |
|---|---|
| `accordeur/dft.py` | TFD directe O(N²) et FFT Cooley-Tukey O(N log N) |
| `accordeur/detection.py` | Extraction de la fondamentale (Hann, zero-padding, HPS, interpolation) |
| `accordeur/notes.py` | Fréquence → note, octave, écart en cents (La4 = 440 Hz) |
| `accordeur/audio.py` | Capture micro, source simulée, lecture WAV |
| `main.py` | Interface temps réel et modes console / fichier |
| `tests/` | Tests pytest (précision < 3 cents sur 9 notes) |
| `docs/Rapport_Accordeur_TFD.pdf` | Rapport : captures, spectre, explication de l'extraction de f0 |

## Principe en bref

1. Trame de 8192 échantillons (Fe = 44100 Hz) → suppression de la moyenne → fenêtre de Hann.
2. FFT sur 32768 points (zero-padding ×4) → spectre d'amplitude.
3. HPS : produit du spectre et de ses versions compressées ×2…×5 → évite les erreurs d'octave.
4. Interpolation parabolique du pic → fréquence fondamentale f0.
5. `m = 69 + 12·log2(f0/440)` → note ; `cents = 1200·log2(f0/f_théorique)`.
