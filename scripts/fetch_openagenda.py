"""
Feature 2 : récupération des événements OpenAgenda.

Ce script interroge plusieurs agendas OpenAgenda (multi-sources), récupère
tous leurs événements avec pagination, fusionne et déduplique le résultat,
puis le sauvegarde dans data/raw.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# ===========================================================================
# 0. Configuration
# ===========================================================================

AGENDAS = [
    {"uid": 21769447, "name": "Aix-Marseille-Provence Métropole"},
    {"uid": 2119473, "name": "Musées de Marseille"},
    {"uid": 65135660, "name": "gmem-CNCM-marseille"},
]

TAILLE_PAGE = 100
MAX_TENTATIVES = 5
DELAI_ATTENTE = 10
TIMEOUT_REQUETE = 30

CHEMIN_SORTIE = Path("data/raw/events_multisources_raw.json")


# ===========================================================================
# 1. Chargement de la clé API
# ===========================================================================

def charger_cle_api() -> str:
    """Charge et vérifie la présence de la clé API OpenAgenda."""

    load_dotenv()
    api_key = os.getenv("OPENAGENDA_API_KEY")

    if not api_key:
        print("❌ Variable OPENAGENDA_API_KEY introuvable.")
        print("Vérifie la présence de la clé dans le fichier .env.")
        sys.exit(1)

    print("✅ Clé API chargée avec succès.")
    return api_key


# ===========================================================================
# 2. Récupération d'une page
# ===========================================================================

def recuperer_page(
    session: requests.Session,
    url_api: str,
    headers: dict[str, str],
    params: dict[str, Any],
    numero_page: int,
) -> dict[str, Any]:
    """Interroge l'API pour une page donnée, avec plusieurs tentatives en cas d'erreur."""

    for tentative in range(1, MAX_TENTATIVES + 1):
        try:
            response = session.get(
                url_api,
                headers=headers,
                params=params,
                timeout=TIMEOUT_REQUETE,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as erreur:
            print(f"⚠️ Page {numero_page}, tentative {tentative}/{MAX_TENTATIVES} : {erreur}")
            if tentative < MAX_TENTATIVES:
                print(f"   Nouvelle tentative dans {DELAI_ATTENTE} secondes...")
                time.sleep(DELAI_ATTENTE)

    raise RuntimeError(f"Impossible de récupérer la page {numero_page} après {MAX_TENTATIVES} tentatives.")


# ===========================================================================
# 3. Pagination complète pour UN agenda
# ===========================================================================

def recuperer_tous_les_evenements(
    api_key: str,
    agenda_uid: int,
) -> tuple[list[dict[str, Any]], int | None]:
    """Récupère tous les événements d'un agenda donné, avec pagination par curseur."""

    url_api = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"

    tous_les_evenements: list[dict[str, Any]] = []
    headers = {"key": api_key}
    after_cursor = None
    numero_page = 1
    total_api = None

    with requests.Session() as session:
        while True:
            params: dict[str, Any] = {"size": TAILLE_PAGE, "detailed": 1}
            if after_cursor:
                params["after[]"] = after_cursor

            data = recuperer_page(
                session=session,
                url_api=url_api,
                headers=headers,
                params=params,
                numero_page=numero_page,
            )

            evenements_page = data.get("events", [])

            if total_api is None:
                total_api = data.get("total")

            if not evenements_page:
                break

            tous_les_evenements.extend(evenements_page)

            print(
                f"  Page {numero_page:>2} : {len(evenements_page):>3} événements reçus "
                f"(total cumulé : {len(tous_les_evenements)})"
            )

            after_cursor = data.get("after")
            if not after_cursor:
                break
            if total_api and len(tous_les_evenements) >= total_api:
                break

            numero_page += 1

    return tous_les_evenements, total_api


# ===========================================================================
# 4. Collecte multi-sources : fusion + déduplication
# ===========================================================================

def collecter_toutes_les_sources(api_key: str) -> list[dict[str, Any]]:
    """Interroge chaque agenda de la liste AGENDAS, fusionne et déduplique par uid."""

    evenements_par_source: dict[str, list[dict[str, Any]]] = {}

    for agenda in AGENDAS:
        print(f"\n--- Agenda : {agenda['name']} (UID {agenda['uid']}) ---")

        evenements, total_api = recuperer_tous_les_evenements(api_key, agenda["uid"])

        for evenement in evenements:
            evenement["_source_agenda"] = agenda["name"]

        evenements_par_source[agenda["name"]] = evenements

        print(
            f"✅ {agenda['name']} : {len(evenements)} événements récupérés"
            + (f" (total API : {total_api})" if total_api is not None else "")
        )

    tous_les_evenements = [
        evenement
        for liste in evenements_par_source.values()
        for evenement in liste
    ]

    print(f"\nTotal avant déduplication : {len(tous_les_evenements)}")

    evenements_dedupliques: dict[Any, dict[str, Any]] = {}
    for evenement in tous_les_evenements:
        uid = evenement.get("uid")
        if uid not in evenements_dedupliques:
            evenements_dedupliques[uid] = evenement

    resultat = list(evenements_dedupliques.values())

    print(
        f"Total après déduplication : {len(resultat)} "
        f"({len(tous_les_evenements) - len(resultat)} doublons supprimés)"
    )

    return resultat


# ===========================================================================
# 5. Sauvegarde
# ===========================================================================

def sauvegarder_evenements(evenements: list[dict[str, Any]], chemin_sortie: Path) -> None:
    """Sauvegarde les événements au format JSON, via une écriture atomique."""

    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    chemin_temporaire = chemin_sortie.with_suffix(".tmp")

    with chemin_temporaire.open("w", encoding="utf-8") as fichier:
        json.dump(evenements, fichier, ensure_ascii=False, indent=2)

    chemin_temporaire.replace(chemin_sortie)


# ===========================================================================
# 6. Programme principal
# ===========================================================================

def main() -> None:
    """Exécute la collecte multi-sources complète."""

    print("=" * 60)
    print("COLLECTE OPENAGENDA MULTI-SOURCES")
    print("=" * 60)
    for agenda in AGENDAS:
        print(f"- {agenda['name']} (UID {agenda['uid']})")

    api_key = charger_cle_api()

    try:
        evenements = collecter_toutes_les_sources(api_key)
    except RuntimeError as erreur:
        print(f"\n❌ Échec de la collecte : {erreur}")
        sys.exit(1)

    if not evenements:
        print("\n❌ Aucun événement récupéré. Le fichier ne sera pas écrasé.")
        sys.exit(1)

    sauvegarder_evenements(evenements, CHEMIN_SORTIE)

    print(f"\n✅ Fichier sauvegardé : {CHEMIN_SORTIE}")
    print(f"✅ Nombre de documents (après fusion et déduplication) : {len(evenements)}")


if __name__ == "__main__":
    main()