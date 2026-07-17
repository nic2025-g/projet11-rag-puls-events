"""
Feature 3 : Pre-processing des événements OpenAgenda.

Ce script charge les événements bruts, filtre sur Marseille,
et produit un jeu de données propre pour l'indexation FAISS.
"""

import json

#===========================================================================
# 1.  Chargement des données brutes
#===========================================================================

with open("data/samples/sample_events_raw.json", "r", encoding="utf-8") as f:
    events_bruts = json.load(f)

print(f"Nombre d'événements chargés : {len(events_bruts)}")

print("\nPremier événement (aperçu) :")
print(f"- Titre : {events_bruts[0]['title']['fr']}")
print(f"- Ville : {events_bruts[0]['location']['city']}")

#===========================================================================
# 2.  Filtrage sur la ville de Marseille
#===========================================================================

VILLE_CIBLE = "Marseille"

events_marseille = []

for event in events_bruts:
    ville = event["location"]["city"]
    if ville == VILLE_CIBLE:
        events_marseille.append(event)

print(f"\nÉvénements à {VILLE_CIBLE} : {len(events_marseille)} sur {len(events_bruts)}")

for event in events_marseille:
    print(f"- {event['title']['fr']}")

#===========================================================================
# 3.  Structuration avec pandas
#===========================================================================

import pandas as pd

lignes = []

for event in events_marseille:
    lignes.append({
        "uid": event["uid"],
        "titre": event["title"]["fr"],
        "description": event["description"]["fr"],
        "lieu": event["location"]["name"],
        "adresse": event["location"]["address"],
        "debut": event["firstTiming"]["begin"],
        "fin": event["lastTiming"]["end"],
    })

df = pd.DataFrame(lignes)

print("\nAperçu du tableau structuré :")
print(df)

#===========================================================================
# 4.  Filtrage sur la période (moins de 12 mois)
#===========================================================================

from datetime import datetime, timedelta, timezone

DATE_LIMITE = datetime.now(timezone.utc) - timedelta(days=365)

print(f"\nDate limite (il y a 12 mois) : {DATE_LIMITE.date()}")

df["debut_dt"] = pd.to_datetime(df["debut"])

df_recent = df[df["debut_dt"] >= DATE_LIMITE]

print(f"Événements des 12 derniers mois : {len(df_recent)} sur {len(df)}")
print(df_recent[["titre", "debut"]])

#===========================================================================
# 5.  Préparation du texte et sauvegarde finale
#===========================================================================

import os

# On combine titre + description en un seul texte, celui qui sera vectorisé plus tard
df_recent["texte_complet"] = df_recent["titre"] + ". " + df_recent["description"]

# On ne garde que les colonnes utiles pour la suite du projet
colonnes_finales = ["uid", "titre", "texte_complet", "lieu", "adresse", "debut", "fin"]
df_final = df_recent[colonnes_finales]

print("\nAperçu final avant sauvegarde :")
print(df_final)

os.makedirs("data/processed", exist_ok=True)

df_final.to_json("data/processed/events_clean.json", orient="records", force_ascii=False, indent=2)

print(f"\n✅ {len(df_final)} événements sauvegardés dans data/processed/events_clean.json")