"""Conversion fréquence <-> note de musique (gamme tempérée, La4 = 440 Hz)."""
import math

NOMS_NOTES = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
FREQ_LA4 = 440.0


def frequence_vers_midi(frequence, la4=FREQ_LA4):
    """Numéro MIDI (flottant) correspondant à une fréquence."""
    return 69.0 + 12.0 * math.log2(frequence / la4)


def midi_vers_frequence(midi, la4=FREQ_LA4):
    return la4 * 2.0 ** ((midi - 69.0) / 12.0)


def analyser_note(frequence, la4=FREQ_LA4):
    """Retourne (nom, octave, frequence_theorique, ecart_en_cents) de la note la plus proche."""
    midi = frequence_vers_midi(frequence, la4)
    midi_proche = int(round(midi))
    nom = NOMS_NOTES[midi_proche % 12]
    octave = midi_proche // 12 - 1
    f_theorique = midi_vers_frequence(midi_proche, la4)
    cents = 1200.0 * math.log2(frequence / f_theorique)
    return nom, octave, f_theorique, cents
