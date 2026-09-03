"""
Tests unitaires de l'intégration LangChain.

Vérifie le retriever personnalisé et le formatage des documents,
sans appel réseau réel : la recherche FAISS/Mistral est simulée.
"""

import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "scripts"),
)

from langchain_rag import (
    RetrieverEvenementsFAISS,
    formater_documents,
)


# ===========================================================================
# Test 1 : le retriever transforme correctement les résultats en Document
# ===========================================================================

def test_retriever_transforme_en_document():
    """
    Le retriever doit transformer chaque résultat de rechercher_evenements
    en Document LangChain, avec le texte dans page_content et les
    métadonnées associées.
    """

    resultats_simules = [
        {
            "uid": "evt_1",
            "chunk_id": "evt_1_0",
            "titre": "Concert jazz",
            "lieu": "Friche la Belle de Mai",
            "adresse": "41 rue Jobin, 13003 Marseille",
            "debut": "2026-06-13T19:30:00+02:00",
            "fin": "2026-06-13T23:00:00+02:00",
            "texte": (
                "Titre : Concert jazz. "
                "Description : Une soirée jazz."
            ),
            "score_similarite": 0.85,
        }
    ]

    with patch(
        "langchain_rag.rechercher_evenements",
        return_value=resultats_simules,
    ):
        retriever = RetrieverEvenementsFAISS(top_k=1)

        documents = retriever.invoke(
            "concert jazz"
        )

    assert len(documents) == 1
    assert isinstance(documents[0], Document)

    assert documents[0].page_content == (
        "Titre : Concert jazz. "
        "Description : Une soirée jazz."
    )

    assert documents[0].metadata["uid"] == "evt_1"
    assert documents[0].metadata["chunk_id"] == "evt_1_0"
    assert documents[0].metadata["titre"] == "Concert jazz"
    assert documents[0].metadata["lieu"] == "Friche la Belle de Mai"
    assert documents[0].metadata["score_similarite"] == 0.85


# ===========================================================================
# Test 2 : le retriever transmet bien top_k
# ===========================================================================

def test_retriever_transmet_top_k():
    """
    Le paramètre top_k du retriever doit être transmis
    à rechercher_evenements().
    """

    with patch(
        "langchain_rag.rechercher_evenements",
        return_value=[],
    ) as mock_recherche:

        retriever = RetrieverEvenementsFAISS(
            top_k=3
        )

        retriever.invoke(
            "une requête"
        )

        mock_recherche.assert_called_once_with(
            "une requête",
            top_k=3,
        )


# ===========================================================================
# Test 3 : formater_documents produit un contexte structuré
# ===========================================================================

def test_formater_documents_contient_les_champs_cles():
    """
    formater_documents doit transformer les Documents LangChain
    en contexte structuré utilisable par le prompt.
    """

    documents = [
        Document(
            page_content=(
                "Titre : Concert jazz. "
                "Description : Une soirée jazz."
            ),
            metadata={
                "uid": "evt_1",
                "chunk_id": "evt_1_0",
                "titre": "Concert jazz",
                "lieu": "Friche la Belle de Mai",
                "adresse": "41 rue Jobin, 13003 Marseille",
                "debut": "2026-06-13T19:30:00+02:00",
                "fin": "2026-06-13T23:00:00+02:00",
                "score_similarite": 0.85,
            },
        )
    ]

    contexte = formater_documents(
        documents
    )

    assert "ÉVÉNEMENT 1" in contexte
    assert "Concert jazz" in contexte
    assert "Friche la Belle de Mai" in contexte
    assert "41 rue Jobin" in contexte
    assert "13/06/2026" in contexte
    assert "Une soirée jazz." in contexte


# ===========================================================================
# Test 4 : formater_documents gère une liste vide
# ===========================================================================

def test_formater_documents_liste_vide():
    """
    Une liste de Documents vide ne doit pas provoquer d'erreur.
    """

    contexte = formater_documents([])

    assert "Aucun événement" in contexte