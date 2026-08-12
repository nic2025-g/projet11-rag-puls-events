"""
Feature 3 : pré-processing des événements OpenAgenda.

Ce script :
- charge les événements bruts ;
- filtre les événements situés à Marseille ;
- nettoie les champs textuels ;
- contrôle la qualité des données ;
- filtre la période retenue ;
- supprime les doublons ;
- prépare les documents destinés aux embeddings et à FAISS.
"""

import html
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ===========================================================================
# 0. Configuration
# ===========================================================================

AGENDA_NAME = "Aix-Marseille-Provence Métropole"
AGENDA_UID = 21769447

VILLE_CIBLE = "Marseille"

#CHEMIN_SOURCE = Path("data/raw/events_metropole_raw.json")
CHEMIN_SOURCE = Path("data/raw/events_multisources_raw.json")
CHEMIN_SORTIE = Path("data/processed/events_clean.json")

NOMBRE_JOURS_PASSES = 60    # événements récents / encore en cours
NOMBRE_JOURS_FUTURS = 305   # reste de la fenêtre, orienté vers l'avenir

AGENDAS_EXCLUS = {
    "Mes événements France Travail",
    "Challenges Geovelo",
    "Mai à vélo",
}


AGENDAS = [
    {"uid": 21769447, "name": "Aix-Marseille-Provence Métropole"},
    {"uid": 2119473, "name": "Musées de Marseille"},
    {"uid": 65135660, "name": "gmem-CNCM-marseille"},
]

# ===========================================================================
# 1. Fonctions utilitaires
# ===========================================================================

def nettoyer_texte(valeur: Any) -> str:
    if valeur is None:
        return ""

    texte = str(valeur)

    texte = html.unescape(texte)

    # Supprime les balises HTML
    texte = re.sub(r"<[^>]+>", " ", texte)

    # Nettoie quelques marqueurs Markdown
    texte = texte.replace(r"\[", "[").replace(r"\]", "]")
    texte = re.sub(r"\*\*(.*?)\*\*", r"\1", texte)
    texte = re.sub(r"_(.*?)_", r"\1", texte)

    # Réduit les espaces multiples
    texte = re.sub(r"\s+", " ", texte)
    
    # Supprimer les antislashs Markdown répétés
    texte = re.sub(r"\\+", " ", texte)

    # Supprimer les titres Markdown
    texte = re.sub(r"#{1,6}\s*", "", texte)

    # Supprimer les puces Markdown
    texte = re.sub(r"\*\s*", "", texte)

    # Transformer les liens Markdown [texte](url) en texte
    texte = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texte)

    return texte.strip()


def obtenir_traduction_francaise(
    champ: Any,
    valeur_defaut: str = "",
) -> str:
    """
    Extrait la version française d'un champ multilingue OpenAgenda.

    Si le français est absent, prend la première valeur textuelle disponible.
    """

    if isinstance(champ, str):
        return nettoyer_texte(champ)

    if not isinstance(champ, dict):
        return valeur_defaut

    valeur_fr = champ.get("fr")

    if valeur_fr:
        return nettoyer_texte(valeur_fr)

    for valeur in champ.values():
        if valeur:
            return nettoyer_texte(valeur)

    return valeur_defaut


def construire_texte_complet(ligne: pd.Series) -> str:
    """Construit le document textuel envoyé au modèle d'embeddings."""

    parties = []

    if ligne["titre"]:
        parties.append(f"Titre : {ligne['titre']}.")

    if ligne["description"]:
        parties.append(f"Description : {ligne['description']}.")

    if ligne["lieu"]:
        parties.append(f"Lieu : {ligne['lieu']}.")

    if ligne["adresse"]:
        parties.append(f"Adresse : {ligne['adresse']}.")

    if ligne["debut"]:
        parties.append(f"Début : {ligne['debut']}.")

    if ligne["fin"]:
        parties.append(f"Fin : {ligne['fin']}.")

    return nettoyer_texte(" ".join(parties))


# ===========================================================================
# 2. Chargement
# ===========================================================================

def charger_evenements(chemin_source: Path) -> list[dict[str, Any]]:
    """Charge et vérifie le fichier JSON brut."""

    if not chemin_source.exists():
        print(f"❌ Fichier introuvable : {chemin_source}")
        sys.exit(1)

    try:
        with chemin_source.open("r", encoding="utf-8-sig") as fichier:
            donnees = json.load(fichier)
    except json.JSONDecodeError as erreur:
        print(f"❌ JSON invalide : {erreur}")
        sys.exit(1)

    if not isinstance(donnees, list):
        print("❌ Le fichier source doit contenir une liste d'événements.")
        sys.exit(1)

    if not donnees:
        print("❌ Le fichier source ne contient aucun événement.")
        sys.exit(1)

    return donnees


# ===========================================================================
# 3. Filtrage Marseille
# ===========================================================================

def filtrer_par_ville(
    evenements: list[dict[str, Any]],
    ville_cible: str,
) -> list[dict[str, Any]]:
    """Filtre les événements selon location.city."""

    ville_normalisee = ville_cible.strip().casefold()

    return [
        evenement
        for evenement in evenements
        if nettoyer_texte(
            evenement.get("location", {}).get("city", "")
        ).casefold()
        == ville_normalisee
    ]
    
    
def obtenir_titre_agenda(evenement: dict[str, Any]) -> str:
    """Retourne le titre de l'agenda d'origine."""

    agenda = evenement.get("originAgenda") or {}
    titre = agenda.get("title") or ""

    return nettoyer_texte(titre)


def est_agenda_exclu(evenement: dict[str, Any]) -> bool:
    """Indique si l'agenda d'origine doit être exclu."""

    titre_agenda = obtenir_titre_agenda(evenement)

    return titre_agenda in AGENDAS_EXCLUS

# ===========================================================================
# 4. Structuration
# ===========================================================================

def structurer_evenements(
    evenements: list[dict[str, Any]],
) -> pd.DataFrame:
    """Transforme les événements JSON en DataFrame."""

    lignes = []

    for evenement in evenements:
        location = evenement.get("location") or {}
        first_timing = evenement.get("firstTiming") or {}
        last_timing = evenement.get("lastTiming") or {}

        lignes.append(
            {
                "uid": evenement.get("uid"),
                "titre": obtenir_traduction_francaise(evenement.get("title")),
                "description": (
                    obtenir_traduction_francaise(evenement.get("longDescription"))
                    or obtenir_traduction_francaise(evenement.get("description"))
                ),
                "lieu": nettoyer_texte(location.get("name")),
                "adresse": nettoyer_texte(location.get("address")),
                "code_postal": nettoyer_texte(location.get("postalCode")),
                "ville": nettoyer_texte(location.get("city")),
                "debut": nettoyer_texte(first_timing.get("begin")),
                "fin": nettoyer_texte(last_timing.get("end")),
                "source_agenda": nettoyer_texte(evenement.get("_source_agenda")),
            }
        )

    return pd.DataFrame(lignes)


# ===========================================================================
# 5. Programme principal
# ===========================================================================

def main() -> None:
    """Exécute le pré-traitement complet."""

    events_bruts = charger_evenements(CHEMIN_SOURCE)

    print("=" * 60)
    print("PRÉ-TRAITEMENT OPENAGENDA")
    print("=" * 60)
    print(f"Événements chargés : {len(events_bruts)}")
    
    
    events_marseille = filtrer_par_ville(
    events_bruts,
    VILLE_CIBLE,
    )

    print(
        f"Événements à {VILLE_CIBLE} : "
        f"{len(events_marseille)} sur {len(events_bruts)}"
    )

    if not events_marseille:
        print("❌ Aucun événement ne correspond à la ville cible.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Filtrage métier selon l'agenda d'origine
    # -----------------------------------------------------------------------

    nombre_avant_filtre_metier = len(events_marseille)

    events_culturels = [
        evenement
        for evenement in events_marseille
        if not est_agenda_exclu(evenement)
    ]

    nombre_exclus_agendas = (
        nombre_avant_filtre_metier - len(events_culturels)
    )

    if not events_culturels:
        print("❌ Aucun événement après le filtre métier.")
        sys.exit(1)

    df = structurer_evenements(events_culturels)
    
    # -----------------------------------------------------------------------
    # Diagnostic avant nettoyage
    # -----------------------------------------------------------------------

    nb_sans_uid = df["uid"].isna().sum()
    nb_sans_titre = df["titre"].eq("").sum()
    nb_sans_description = df["description"].eq("").sum()
    nb_sans_lieu = df["lieu"].eq("").sum()
    nb_sans_date = df["debut"].eq("").sum()
    nb_doublons_uid = df["uid"].duplicated(keep="first").sum()

    # -----------------------------------------------------------------------
    # Conversion des dates
    # -----------------------------------------------------------------------

    df["debut_dt"] = pd.to_datetime(
        df["debut"],
        utc=True,
        errors="coerce",
    )

    df["fin_dt"] = pd.to_datetime(
        df["fin"],
        utc=True,
        errors="coerce",
    )

    nb_dates_invalides = df["debut_dt"].isna().sum()

    # -----------------------------------------------------------------------
    # Suppression des doublons
    # -----------------------------------------------------------------------

    nombre_avant_dedoublonnage = len(df)

    # Les UID absents ne doivent pas tous être considérés comme identiques.
    df_avec_uid = df[df["uid"].notna()].drop_duplicates(
        subset=["uid"],
        keep="first",
    )

    df_sans_uid = df[df["uid"].isna()].drop_duplicates(
        subset=["titre", "lieu", "debut"],
        keep="first",
    )

    df = pd.concat(
        [df_avec_uid, df_sans_uid],
        ignore_index=True,
    )

    nb_supprimes_dedoublonnage = nombre_avant_dedoublonnage - len(df)

    # -----------------------------------------------------------------------
    # Filtrage temporel : événements à venir ou encore en cours
    # -----------------------------------------------------------------------
    
    maintenant = datetime.now(timezone.utc)
    date_debut_periode = maintenant - timedelta(days=NOMBRE_JOURS_PASSES)
    date_fin_periode = maintenant + timedelta(days=NOMBRE_JOURS_FUTURS)

    date_reference_fin = df["fin_dt"].fillna(df["debut_dt"])

    masque_periode = (
        df["debut_dt"].notna()
        & (date_reference_fin >= date_debut_periode)
        & (df["debut_dt"] <= date_fin_periode)
    )

    df_periode = df.loc[masque_periode].copy()

    # -----------------------------------------------------------------------
    # Exclusion des documents inutilisables
    # -----------------------------------------------------------------------

    # Un événement doit au minimum avoir un titre.
    df_periode = df_periode[
        df_periode["titre"].str.len() > 0
    ].copy()

    # Construction du contenu destiné aux embeddings.
    df_periode["texte_complet"] = df_periode.apply(
        construire_texte_complet,
        axis=1,
    )

    # Protection contre les documents quasiment vides.
    df_final = df_periode[
        df_periode["texte_complet"].str.len() >= 20
    ].copy()

    # Tri chronologique pour rendre le fichier plus facile à inspecter.
    df_final = df_final.sort_values(
        by="debut_dt",
        ascending=True,
    )

    colonnes_finales = [
        "uid",
        "titre",
        "texte_complet",
        "lieu",
        "adresse",
        "code_postal",
        "ville",
        "debut",
        "fin",
        "source_agenda",
    ]

    df_final = df_final[colonnes_finales]
    
    # -----------------------------------------------------------------------
    # Sauvegarde
    # -----------------------------------------------------------------------

    CHEMIN_SORTIE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_final.to_json(
        CHEMIN_SORTIE,
        orient="records",
        force_ascii=False,
        indent=2,
    )

    # -----------------------------------------------------------------------
    # Rapport
    # -----------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("RAPPORT DE PRÉ-TRAITEMENT")
    print("=" * 60)

    #print(f"Agenda                         : {AGENDA_NAME}")
    #print(f"UID agenda                     : {AGENDA_UID}")
    #print(f"Événements collectés           : {len(events_bruts)}")
    #print(f"Événements à Marseille         : {len(events_marseille)}")
    #print(f"Sans UID                       : {nb_sans_uid}")
    
    print(f"Agenda(s) source(s)            : {', '.join(a['name'] for a in AGENDAS)}")
    print(f"Événements collectés           : {len(events_bruts)}")
    print(f"Événements à Marseille         : {len(events_marseille)}")
    print(f"Exclus par filtre métier       : {nombre_exclus_agendas}")
    print(f"Après filtre métier            : {len(events_culturels)}")
    print(f"Sans UID                       : {nb_sans_uid}")
    ...
    
    print(f"Sans titre                     : {nb_sans_titre}")
    print(f"Sans description               : {nb_sans_description}")
    print(f"Sans lieu                      : {nb_sans_lieu}")
    print(f"Sans date de début             : {nb_sans_date}")
    print(f"Dates de début invalides       : {nb_dates_invalides}")
    print(f"Doublons UID détectés          : {nb_doublons_uid}")
    print(
        f"Doublons réellement supprimés  : "
        f"{nb_supprimes_dedoublonnage}"
    )
    print(
    f"Période retenue                : "
    f"{date_debut_periode.date()} au {date_fin_periode.date()}"
    )
    print(f"Après filtre temporel          : {len(df_periode)}")
    print(f"Événements finaux              : {len(df_final)}")
    print(f"Fichier produit                : {CHEMIN_SORTIE}")

    print("=" * 60)

    print("\nAperçu de trois documents :")

    for texte in df_final["texte_complet"].head(3):
        print("\n---")
        print(texte)
    

if __name__ == "__main__":
    main()