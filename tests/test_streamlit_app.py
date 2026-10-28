
"""Tests des fonctions de présentation de Puls-Events."""

from datetime import datetime, timedelta, timezone

from app import (
    formater_date_affichage,
    evenement_a_venir,
    nettoyer_description_affichage,
)


def test_formatage_date():
    date = "2026-11-23T13:00:00+01:00"

    resultat = formater_date_affichage(date)

    assert resultat == "23/11/2026 à 13h00"


def test_date_absente():
    assert formater_date_affichage(None) == "Date non précisée"


def test_evenement_futur():
    demain = datetime.now(timezone.utc) + timedelta(days=1)

    evenement = {"fin": demain.isoformat()}

    assert evenement_a_venir(evenement) is True


def test_evenement_passe():
    hier = datetime.now(timezone.utc) - timedelta(days=1)

    evenement = {"fin": hier.isoformat()}

    assert evenement_a_venir(evenement) is False


def test_evenement_sans_date():
    assert evenement_a_venir({"fin": None}) is False


def test_nettoyage_description():
    texte = (
        "Titre : Exposition Marseille. "
        "Description : Une exposition culturelle. "
        "Lieu : Mucem. Adresse : Marseille."
    )

    resultat = nettoyer_description_affichage(
        texte,
        "Exposition Marseille",
    )

    assert resultat == "Une exposition culturelle"
