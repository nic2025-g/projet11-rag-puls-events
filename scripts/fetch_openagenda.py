"""
Feature 2 : Récupération des événements OpenAgenda.

Ce script interroge l'API OpenAgenda et sauvegarde les événements
dans le dossier data/raw.
"""

import os
import json
import requests
import time

from dotenv import load_dotenv

#===========================================================================
# 1.  Reconnaissance de la clé API
#===========================================================================

# Charger les variables d'environnement
load_dotenv()

# Lire la clé API
API_KEY = os.getenv("OPENAGENDA_API_KEY")

# Vérification
if API_KEY:
    print("✅ Clé API chargée avec succès.")
else:
    print("❌ Clé API introuvable.")

#===========================================================================
# 2.  Communication avec l'API
#===========================================================================

# UID de l'agenda ciblé (Aix-Marseille-Provence Métropole)
AGENDA_UID = 21769447

# URL de l'API OpenAgenda -- on cible maintenant les évènements d'UN agenda précis
url = f"https://api.openagenda.com/v2/agendas/{AGENDA_UID}/events"

# Paramètres de la requête
params = {
    "size": 5
}

# En-tête HTTP contenant la clé API
headers = {
    "key": API_KEY
}

print("\nConnexion à l'API OpenAgenda...")

# Envoi de la requête

# Nombre de tentatives avant d'abandonner
MAX_TENTATIVES = 5
DELAI_ATTENTE = 10  # secondes entre deux tentatives

response = None
for tentative in range(1, MAX_TENTATIVES + 1):
    print(f"\nConnexion à l'API OpenAgenda... (tentative {tentative}/{MAX_TENTATIVES})")
    response = requests.get(
    url,
    headers=headers,
    params=params,
    timeout=30,
)
    print(f"Code HTTP : {response.status_code}")

    if response.status_code == 200:
        print("✅ Connexion à l'API réussie.")
        break
    else:
        print(f"❌ Erreur HTTP : {response.status_code}")
        if tentative < MAX_TENTATIVES:
            print(f"Nouvelle tentative dans {DELAI_ATTENTE} secondes...")
            time.sleep(DELAI_ATTENTE)

response.raise_for_status()

print(f"Code HTTP : {response.status_code}")

if response.status_code == 200:
    print("✅ Connexion à l'API réussie.")
else:
    print(f"❌ Erreur HTTP : {response.status_code}")

if response.status_code != 200:
    print("\n❌ Impossible de contacter OpenAgenda après plusieurs tentatives.")
    exit()

#===========================================================================
# 3.  Récupration des agendas
#===========================================================================

data = response.json()

print(f"Nombre total d'événements dans cet agenda : {data['total']}")

for event in data["events"]:
    titre = event.get("title", {}).get("fr", "Titre inconnu")
    ville = event.get("location", {}).get("city", "Ville inconnue")
    debut = event.get("firstTiming", {}).get("begin", "Date inconnue")
    print(f"- {titre} | {ville} | à partir du {debut}")

#===========================================================================
# 4.  Sauvegarde des événements bruts
#===========================================================================

os.makedirs("data/raw", exist_ok=True)

with open("data/raw/events_marseille_raw.json", "w", encoding="utf-8") as f:
    json.dump(data["events"], f, ensure_ascii=False, indent=2)

print(f"\n✅ {len(data['events'])} événements sauvegardés dans data/raw/events_marseille_raw.json")