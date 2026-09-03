"""
Test de conformité des données vectorisées.

Vérifie que chaque entrée présente dans la base vectorielle (FAISS)
correspond bien à un événement situé à Marseille et daté de moins
d'un an, conformément au périmètre du projet.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


CHEMIN_METADATA = Path("faiss_index/metadata.json")
CHEMIN_CONFIG_PREPROCESSING = Path(
    "data/processed/preprocessing_metadata.json"
)

VILLE_ATTENDUE = "Marseille"

def charger_metadata_vectorisee() -> list[dict]:
    """Charge les métadonnées réellement indexées dans FAISS."""

    if not CHEMIN_METADATA.exists():
        pytest.skip(
            f"Métadonnées FAISS introuvables ({CHEMIN_METADATA}) -- "
            f"lancez d'abord build_faiss_index.py."
        )

    with CHEMIN_METADATA.open(
        "r",
        encoding="utf-8-sig",
    ) as fichier:
        metadata = json.load(fichier)

    if not isinstance(metadata, list):
        raise ValueError(
            "Le fichier de métadonnées FAISS doit contenir une liste JSON."
        )

    return metadata


def charger_config_preprocessing() -> dict:
    """Charge la configuration réellement utilisée lors du preprocessing."""

    if not CHEMIN_CONFIG_PREPROCESSING.exists():
        pytest.skip(
            f"Configuration de preprocessing introuvable "
            f"({CHEMIN_CONFIG_PREPROCESSING}) -- "
            f"lancez d'abord preprocess_events.py."
        )

    with CHEMIN_CONFIG_PREPROCESSING.open(
        "r",
        encoding="utf-8",
    ) as fichier:
        config = json.load(fichier)

    return config

#===========================================================================
# Test 1 : la base vectorielle n'est pas vide
#===========================================================================

def test_base_vectorielle_non_vide():
    """La base vectorielle doit contenir au moins un événement indexé."""

    metadata = charger_metadata_vectorisee()

    assert len(metadata) > 0, "Aucun événement dans la base vectorielle."


#===========================================================================
# Test 2 : tous les événements sont bien situés à Marseille
#===========================================================================

def test_tous_les_evenements_sont_a_marseille():
    """
    Chaque entrée de la base vectorielle doit correspondre à un événement
    dont la ville est Marseille (comparaison insensible à la casse).
    """

    metadata = charger_metadata_vectorisee()
    config = charger_config_preprocessing()

    evenements_hors_perimetre = [
        chunk for chunk in metadata
        if chunk.get("ville", "").strip().casefold() != VILLE_ATTENDUE.casefold()
    ]

    assert not evenements_hors_perimetre, (
        f"{len(evenements_hors_perimetre)} entrée(s) hors périmètre géographique "
        f"détectée(s) (attendu : {VILLE_ATTENDUE}). "
        f"Exemples : {[e.get('chunk_id') for e in evenements_hors_perimetre[:5]]}"
    )


#===========================================================================
# Test 3 : tous les événements datent de moins d'un an
#===========================================================================

def test_tous_les_evenements_respectent_la_fenetre_temporelle():
    """
    Vérifie que chaque événement indexé chevauche la fenêtre
    réellement utilisée lors du preprocessing.
    """

    metadata = charger_metadata_vectorisee()
    config = charger_config_preprocessing()

    borne_min = datetime.fromisoformat(
        config["date_min"]
    )

    borne_max = datetime.fromisoformat(
        config["date_max"]
    )

    evenements_hors_fenetre = []

    for chunk in metadata:
        debut_str = chunk.get("debut", "")
        fin_str = chunk.get("fin", "")

        try:
            debut = datetime.fromisoformat(debut_str)

            if debut.tzinfo is None:
                debut = debut.replace(tzinfo=timezone.utc)

            debut = debut.astimezone(timezone.utc)

            if fin_str:
                fin = datetime.fromisoformat(fin_str)

                if fin.tzinfo is None:
                    fin = fin.replace(tzinfo=timezone.utc)

                fin = fin.astimezone(timezone.utc)
            else:
                fin = debut

        except (ValueError, TypeError):
            evenements_hors_fenetre.append(chunk)
            continue

        if not (
            fin >= borne_min
            and debut <= borne_max
        ):
            evenements_hors_fenetre.append(chunk)

    assert not evenements_hors_fenetre, (
        f"{len(evenements_hors_fenetre)} entrée(s) "
        f"hors fenêtre temporelle détectée(s). "
        f"Exemples : "
        f"{[e.get('chunk_id') for e in evenements_hors_fenetre[:5]]}"
    )

#===========================================================================
# Test 4 : chaque entrée a les champs minimaux requis
#===========================================================================

def test_champs_minimaux_presents():
    """
    Chaque entrée doit avoir au minimum un uid, un titre non vide,
    et une date de début -- sans quoi elle serait inexploitable
    par la chaîne RAG.
    """

    metadata = charger_metadata_vectorisee()

    entrees_incompletes = [
        chunk for chunk in metadata
        if not chunk.get("uid")
        or not chunk.get("titre", "").strip()
        or not chunk.get("debut", "").strip()
    ]

    assert not entrees_incompletes, (
        f"{len(entrees_incompletes)} entrée(s) incomplète(s) détectée(s)."
    )