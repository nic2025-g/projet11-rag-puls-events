"""
Tests unitaires de la recherche FAISS.

Ces tests vérifient la déduplication des résultats de recherche
sans appel à l'API Mistral et sans utiliser un véritable index FAISS.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "scripts"),
)

from search_faiss import dedupliquer_par_evenement


# ===========================================================================
# Test 1 : même UID -> un seul événement
# ===========================================================================

def test_deduplication_meme_uid():
    """
    Deux chunks appartenant au même événement doivent produire
    un seul résultat : celui ayant le meilleur score.
    """

    metadata = [
        {
            "uid": "evt_1",
            "chunk_id": "evt_1_0",
            "titre": "Concert jazz",
        },
        {
            "uid": "evt_1",
            "chunk_id": "evt_1_1",
            "titre": "Concert jazz",
        },
    ]

    scores = np.array([0.90, 0.80], dtype=np.float32)
    indices = np.array([0, 1])

    resultats = dedupliquer_par_evenement(
        scores,
        indices,
        metadata,
        top_k=5,
    )

    assert len(resultats) == 1
    assert resultats[0]["uid"] == "evt_1"
    assert resultats[0]["chunk_id"] == "evt_1_0"


# ===========================================================================
# Test 2 : UID différents mais même titre -> un seul événement
# ===========================================================================

def test_deduplication_meme_titre():
    """
    Deux événements ayant des UID différents mais le même titre
    ne doivent apparaître qu'une seule fois dans les résultats.
    """

    metadata = [
        {
            "uid": "evt_1",
            "chunk_id": "evt_1_0",
            "titre": "Projet Tae'thir",
        },
        {
            "uid": "evt_2",
            "chunk_id": "evt_2_0",
            "titre": "Projet Tae'thir",
        },
    ]

    scores = np.array([0.90, 0.85], dtype=np.float32)
    indices = np.array([0, 1])

    resultats = dedupliquer_par_evenement(
        scores,
        indices,
        metadata,
        top_k=5,
    )

    assert len(resultats) == 1
    assert resultats[0]["uid"] == "evt_1"
    assert resultats[0]["chunk_id"] == "evt_1_0"


# ===========================================================================
# Test 3 : événements distincts -> ordre conservé
# ===========================================================================

def test_evenements_distincts_ordre_conserve():
    """
    Lorsque tous les événements sont distincts,
    ils doivent être conservés dans l'ordre fourni par FAISS.
    """

    metadata = [
        {
            "uid": "evt_1",
            "chunk_id": "evt_1_0",
            "titre": "Concert jazz",
        },
        {
            "uid": "evt_2",
            "chunk_id": "evt_2_0",
            "titre": "Exposition photo",
        },
        {
            "uid": "evt_3",
            "chunk_id": "evt_3_0",
            "titre": "Festival musique",
        },
    ]

    scores = np.array(
        [0.95, 0.85, 0.75],
        dtype=np.float32,
    )

    indices = np.array([0, 1, 2])

    resultats = dedupliquer_par_evenement(
        scores,
        indices,
        metadata,
        top_k=5,
    )

    assert len(resultats) == 3

    assert [
        resultat["uid"]
        for resultat in resultats
    ] == [
        "evt_1",
        "evt_2",
        "evt_3",
    ]

    assert resultats[0]["score_similarite"] > resultats[1]["score_similarite"]
    assert resultats[1]["score_similarite"] > resultats[2]["score_similarite"]


# ===========================================================================
# Test 4 : index FAISS -1 -> ignoré proprement
# ===========================================================================

def test_index_moins_un_ignore():
    """
    FAISS peut retourner -1 lorsqu'il n'existe pas suffisamment
    de voisins. Cette valeur doit être ignorée sans provoquer d'erreur.
    """

    metadata = [
        {
            "uid": "evt_1",
            "chunk_id": "evt_1_0",
            "titre": "Concert jazz",
        },
        {
            "uid": "evt_2",
            "chunk_id": "evt_2_0",
            "titre": "Festival musique",
        },
    ]

    scores = np.array(
        [0.90, 0.80, -1.0],
        dtype=np.float32,
    )

    indices = np.array([0, 1, -1])

    resultats = dedupliquer_par_evenement(
        scores,
        indices,
        metadata,
        top_k=5,
    )

    assert len(resultats) == 2

    assert [
        resultat["uid"]
        for resultat in resultats
    ] == [
        "evt_1",
        "evt_2",
    ]