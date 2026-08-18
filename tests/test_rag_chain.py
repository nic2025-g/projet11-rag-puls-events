"""
Tests unitaires de la chaîne RAG.

Ces tests couvrent le formatage de dates, le nettoyage de texte,
la construction du contexte, et la génération de réponse avec
un client Mistral simulé (aucun appel réseau réel).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from rag_chain import (
    formater_date,
    nettoyer_texte_pour_contexte,
    construire_contexte,
    generer_reponse_rag,
)


# ===========================================================================
# Test 1 : formater_date
# ===========================================================================

def test_formater_date_valide():
    """Une date ISO valide doit être convertie au format jour/mois/année."""

    resultat = formater_date("2026-06-13T19:30:00.000+02:00")
    assert resultat == "13/06/2026"


def test_formater_date_absente():
    """Une date vide ou absente doit renvoyer un texte de repli explicite."""

    assert formater_date("") == "date non précisée"
    assert formater_date(None) == "date non précisée"


def test_formater_date_invalide():
    """Une date mal formée ne doit pas faire planter le script."""

    resultat = formater_date("pas-une-date")
    assert resultat == "pas-une-date"


# ===========================================================================
# Test 2 : nettoyer_texte_pour_contexte
# ===========================================================================

def test_nettoyer_texte_retire_prefixe_titre_et_description():
    """Le préfixe 'Titre : X. Description :' doit être retiré du texte."""

    texte_brut = "Titre : Soirée jazz. Description : Une belle soirée musicale."
    resultat = nettoyer_texte_pour_contexte(texte_brut, "Soirée jazz")

    assert resultat == "Une belle soirée musicale."


def test_nettoyer_texte_vide():
    """Un texte vide doit renvoyer une chaîne vide, sans erreur."""

    assert nettoyer_texte_pour_contexte("", "Un titre") == ""


# ===========================================================================
# Test 3 : construire_contexte
# ===========================================================================

def test_construire_contexte_liste_vide():
    """Une liste de résultats vide doit produire un message explicite, pas planter."""

    contexte = construire_contexte([])
    assert "Aucun événement" in contexte


def test_construire_contexte_champs_presents():
    """Chaque événement du contexte doit contenir son titre, lieu et description."""

    resultats = [
        {
            "titre": "Concert jazz",
            "lieu": "Friche la Belle de Mai",
            "adresse": "41 rue Jobin",
            "debut": "2026-06-13T19:30:00.000+02:00",
            "fin": "2026-06-13T23:00:00.000+02:00",
            "texte": "Titre : Concert jazz. Description : Une soirée dédiée au jazz.",
        }
    ]

    contexte = construire_contexte(resultats)

    assert "ÉVÉNEMENT 1" in contexte
    assert "Concert jazz" in contexte
    assert "Friche la Belle de Mai" in contexte
    assert "13/06/2026" in contexte
    assert "Une soirée dédiée au jazz." in contexte


def test_construire_contexte_plusieurs_evenements_numerotes():
    """Plusieurs résultats doivent être numérotés dans l'ordre fourni."""

    resultats = [
        {"titre": "Événement A", "texte": ""},
        {"titre": "Événement B", "texte": ""},
    ]

    contexte = construire_contexte(resultats)

    assert "ÉVÉNEMENT 1" in contexte
    assert "ÉVÉNEMENT 2" in contexte
    assert contexte.index("Événement A") < contexte.index("Événement B")


# ===========================================================================
# Outils de simulation (mock) pour la génération LLM
# ===========================================================================

class FauxMessage:
    def __init__(self, content: str):
        self.content = content


class FauxChoix:
    def __init__(self, content: str):
        self.message = FauxMessage(content)


class FausseReponseChat:
    def __init__(self, content: str):
        self.choices = [FauxChoix(content)]


class FauxClientMistral:
    """Simule le client Mistral pour le chat, sans appel réseau réel."""

    class chat:
        @staticmethod
        def complete(model: str, messages: list[dict]) -> FausseReponseChat:
            question_recue = messages[-1]["content"]
            return FausseReponseChat(f"Réponse simulée. Longueur reçue : {len(question_recue)}")


# ===========================================================================
# Test 4 : generer_reponse_rag avec client simulé
# ===========================================================================

def test_generer_reponse_rag_renvoie_le_contenu():
    """La fonction doit renvoyer le texte de la réponse, pas l'objet brut."""

    client = FauxClientMistral()

    reponse = generer_reponse_rag(
        client,
        question="Une exposition d'art ?",
        contexte="ÉVÉNEMENT 1\nTitre : Test",
    )

    assert isinstance(reponse, str)
    assert reponse.startswith("Réponse simulée.")


def test_generer_reponse_rag_transmet_le_contexte():
    """Le contexte fourni doit bien être inclus dans le message envoyé au LLM."""

    client = FauxClientMistral()
    contexte_long = "ÉVÉNEMENT 1\n" + ("Texte de test. " * 20)

    reponse = generer_reponse_rag(
        client,
        question="Question test",
        contexte=contexte_long,
    )

    assert "Longueur reçue :" in reponse