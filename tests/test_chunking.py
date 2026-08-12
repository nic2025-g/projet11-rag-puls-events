"""
Tests unitaires du module de chunking.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chunk_events import decouper_en_chunks, creer_chunks_evenements


def test_aucun_chunk_vide():
    """Aucun chunk généré ne doit être une chaîne vide."""

    texte_long = "Phrase un. " * 300  # texte largement au-dessus du seuil
    chunks = decouper_en_chunks(texte_long)

    assert len(chunks) > 0
    assert all(chunk.strip() != "" for chunk in chunks)


def test_chunking_document_court():
    """Un texte sous le seuil doit rester un unique chunk, inchangé."""

    texte_court = "Une petite exposition sympathique au centre-ville."
    chunks = decouper_en_chunks(texte_court)

    assert len(chunks) == 1
    assert chunks[0] == texte_court


def test_chunk_ids_uniques():
    """Chaque chunk généré doit avoir un chunk_id unique, même sur plusieurs événements."""

    evenements = [
        {"uid": 1, "titre": "Événement A", "texte_complet": "Phrase. " * 300},
        {"uid": 2, "titre": "Événement B", "texte_complet": "Autre phrase. " * 300},
    ]

    resultat = creer_chunks_evenements(evenements)
    identifiants = [chunk["chunk_id"] for chunk in resultat]

    assert len(identifiants) == len(set(identifiants))


def test_tous_evenements_representes():
    """Chaque événement source doit apparaître au moins une fois dans les chunks produits."""

    evenements = [
        {"uid": 10, "titre": "Court", "texte_complet": "Un texte court."},
        {"uid": 20, "titre": "Long", "texte_complet": "Une phrase. " * 300},
    ]

    resultat = creer_chunks_evenements(evenements)
    uids_representes = {chunk["uid"] for chunk in resultat}

    assert uids_representes == {10, 20}