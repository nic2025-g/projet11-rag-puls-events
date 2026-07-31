"""Script d'exploration : analyse la répartition des événements par agenda source et par tag de lieu, 
pour calibrer les filtres métier."""

import json
from collections import Counter
from pathlib import Path

FICHIER_SOURCE = Path("data/raw/events_metropole_raw.json")

def obtenir_tag_francais(tag):
    """Retourne le libellé français d'un tag OpenAgenda."""

    if isinstance(tag, str):
        return tag.strip()

    if not isinstance(tag, dict):
        return None

    label = tag.get("label")

    if isinstance(label, str):
        return label.strip()

    if isinstance(label, dict):
        valeur = (
            label.get("fr")
            or label.get("en")
            or next(iter(label.values()), None)
        )

        if isinstance(valeur, str):
            return valeur.strip()

    return None


with FICHIER_SOURCE.open("r", encoding="utf-8") as fichier:
    evenements = json.load(fichier)


evenements_marseille = [
    evenement
    for evenement in evenements
    if (
        ((evenement.get("location") or {}).get("city") or "")
        .strip()
        .lower()
        == "marseille"
    )
]


compteur_agendas = Counter(
    (evenement.get("originAgenda") or {}).get(
        "title",
        "Sans agenda"
    )
    for evenement in evenements_marseille
)


compteur_tags = Counter()

for evenement in evenements_marseille:
    location = evenement.get("location") or {}
    tags = location.get("tags") or []

    for tag in tags:
        libelle = obtenir_tag_francais(tag)

        if libelle:
            compteur_tags[libelle] += 1


print("=" * 60)
print("ANALYSE DES ÉVÉNEMENTS OPENAGENDA")
print("=" * 60)

print(f"Événements à Marseille : {len(evenements_marseille)}")

print("\n20 agendas les plus fréquents :")

for agenda, nombre in compteur_agendas.most_common(20):
    print(f"{nombre:4} | {agenda}")


print("\n30 tags de lieux les plus fréquents :")

for tag, nombre in compteur_tags.most_common(30):
    print(f"{nombre:4} | {tag}")