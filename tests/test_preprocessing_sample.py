from pathlib import Path
import sys

# Ajoute la racine du projet au chemin Python
RACINE_PROJET = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE_PROJET))

from scripts.preprocess_events import (
    charger_evenements,
    filtrer_par_ville,
    VILLE_CIBLE,
)


CHEMIN_SAMPLE = Path("data/samples/sample_events_raw.json")


def test_sample_est_charge():
    evenements = charger_evenements(CHEMIN_SAMPLE)

    assert len(evenements) == 3


def test_filtrage_geographique_sample():
    evenements = charger_evenements(CHEMIN_SAMPLE)

    evenements_marseille = filtrer_par_ville(
        evenements,
        VILLE_CIBLE,
    )

    assert len(evenements_marseille) == 1

    evenement = evenements_marseille[0]

    assert evenement["location"]["city"] == "Marseille"
    assert evenement["uid"] == 45424319
    assert evenement["title"]["fr"] == "Dormir comme le soleil"