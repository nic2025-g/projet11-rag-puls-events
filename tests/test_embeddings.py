"""
Tests unitaires du module de génération d'embeddings.

Utilise un client Mistral simulé (mock) pour ne jamais appeler
la vraie API pendant les tests.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "scripts"),
)

from generate_embeddings import (
    charger_chunks,
    generer_embeddings_lot,
    generer_tous_les_embeddings,
)


# ===========================================================================
# Outils de simulation (mock)
# ===========================================================================

class FausseDonneeEmbedding:
    """Simule un objet 'item' retourné par l'API, avec juste ce qu'on utilise."""

    def __init__(self, embedding: list[float]):
        self.embedding = embedding


class FausseReponse:
    """Simule la réponse complète de client.embeddings.create()."""

    def __init__(self, textes: list[str]):
        # Un faux vecteur de dimension 4 (au lieu de 1024, pour aller vite),
        # dont la valeur dépend de la longueur du texte, pour rester déterministe.
        self.data = [
            FausseDonneeEmbedding([float(len(texte))] * 4)
            for texte in textes
        ]


class FauxClientMistral:
    """Simule le client Mistral, sans aucun appel réseau réel."""

    class embeddings:
        @staticmethod
        def create(model: str, inputs: list[str]) -> FausseReponse:
            return FausseReponse(inputs)


# ===========================================================================
# Test 1 : charger_chunks() sur un JSON valide
# ===========================================================================

def test_charger_chunks_json_valide(tmp_path):
    """Un fichier JSON valide doit être chargé correctement en liste de dicts."""

    chemin = tmp_path / "chunks_test.json"
    contenu = [
        {"chunk_id": "1_0", "texte": "Premier chunk."},
        {"chunk_id": "1_1", "texte": "Deuxième chunk."},
    ]
    chemin.write_text(json.dumps(contenu), encoding="utf-8")

    resultat = charger_chunks(chemin)

    assert resultat == contenu
    assert len(resultat) == 2


# ===========================================================================
# Test 2 : generer_embeddings_lot() avec le client simulé
# ===========================================================================

def test_generer_embeddings_lot_nombre_correct():
    """Le nombre de vecteurs renvoyés doit correspondre au nombre de textes envoyés."""

    client = FauxClientMistral()
    textes = ["Un texte.", "Un autre texte plus long.", "Court."]

    embeddings = generer_embeddings_lot(client, textes)

    assert len(embeddings) == len(textes)

    assert embeddings[0] == [float(len(textes[0]))] * 4
    assert embeddings[1] == [float(len(textes[1]))] * 4
    assert embeddings[2] == [float(len(textes[2]))] * 4


# ===========================================================================
# Test 3 : generer_tous_les_embeddings() conserve l'alignement
# ===========================================================================

def test_generer_tous_les_embeddings_alignement():
    """Chaque embedding généré doit rester associé au bon chunk, même sur plusieurs lots."""

    client = FauxClientMistral()
    chunks = [
        {"chunk_id": f"evt_{i}", "texte": f"Texte numero {i}"}
        for i in range(45)  # volontairement > taille_lot, pour forcer plusieurs lots
    ]

    embeddings, metadonnees = generer_tous_les_embeddings(client, chunks, taille_lot=20)

    assert len(embeddings) == len(chunks)
    assert len(metadonnees) == len(chunks)

    # Vérifie que l'ordre est conservé : le chunk_id de la métadonnée n°i
    # doit correspondre au chunk d'origine n°i
    for i, chunk_original in enumerate(chunks):
        assert metadonnees[i]["chunk_id"] == chunk_original["chunk_id"]
    
    # On vérifies que le vecteur n°i correspond bien au texte du chunk n°i.
    for i, chunk_original in enumerate(chunks):
        assert embeddings[i] == [float(len(chunk_original["texte"]))] * 4


# ===========================================================================
# Test 4 : un lot vide retourne une liste vide
# ===========================================================================

def test_generer_embeddings_lot_vide():
    """Un appel avec une liste de textes vide ne doit pas planter et renvoyer []."""

    client = FauxClientMistral()
    embeddings = generer_embeddings_lot(client, [])

    assert embeddings == []