import json
import sys

sys.path.insert(0, "scripts")

from chunk_events import creer_chunks_evenements


with open(
    "data/processed/events_clean.json",
    encoding="utf-8",
) as fichier:
    evenements = json.load(fichier)


# Cherche l'événement ayant le texte le plus long
plus_long = max(
    evenements,
    key=lambda evenement: len(
        evenement["texte_complet"]
    ),
)

# On appelle maintenant la fonction COMPLETE,
# celle qui ajoute le contexte et les métadonnées.
chunks_evenement = creer_chunks_evenements(
    [plus_long]
)


print("=" * 60)
print("TEST DU DOCUMENT LE PLUS LONG")
print("=" * 60)

print(f"Titre              : {plus_long['titre']}")
print(
    f"Longueur originale : "
    f"{len(plus_long['texte_complet'])}"
)
print(
    f"Nombre de chunks   : "
    f"{len(chunks_evenement)}"
)


for chunk in chunks_evenement:

    print()
    print(
        f"--- Chunk "
        f"{chunk['chunk_index'] + 1} "
        f"({len(chunk['texte'])} caractères) ---"
    )

    print(chunk["texte"][:300])