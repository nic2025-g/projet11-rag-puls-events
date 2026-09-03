"""
Test manuel : vérifie le comportement de generer_embeddings_lot
sur 3 vrais chunks du corpus, avant généralisation à l'ensemble.
"""

import numpy as np
import sys

sys.path.insert(0, "scripts")

from mistralai.client import Mistral

from generate_embeddings import (
    charger_cle_api,
    charger_chunks,
    generer_embeddings_lot,
    CHEMIN_SOURCE,
)


# ===========================================================================
# 1. Préparation
# ===========================================================================

api_key = charger_cle_api()

client = Mistral(
    api_key=api_key
)

chunks = charger_chunks(
    CHEMIN_SOURCE
)

echantillon = chunks[:3]

print("\nChunks sélectionnés pour le test :")

for chunk in echantillon:
    print(
        f"- {chunk['chunk_id']} | "
        f"{chunk['titre'][:50]}"
    )


# ===========================================================================
# 2. Génération des embeddings
# ===========================================================================

textes = [
    chunk["texte"]
    for chunk in echantillon
]

embeddings = generer_embeddings_lot(
    client,
    textes,
)


# ===========================================================================
# 3. Vérifications
# ===========================================================================

print("\n" + "=" * 60)
print("VÉRIFICATIONS")
print("=" * 60)


# ---------------------------------------------------------------------------
# Critère 1 :
# nombre de textes envoyés = nombre d'embeddings reçus
# ---------------------------------------------------------------------------

nombre_ok = (
    len(embeddings)
    == len(textes)
)

print(
    f"1. Embeddings reçus : {len(embeddings)} "
    f"pour {len(textes)} textes : "
    f"{'✅' if nombre_ok else '❌'}"
)


# ---------------------------------------------------------------------------
# Critère 2 :
# chaque embedding doit faire 1024 dimensions
# ---------------------------------------------------------------------------

dimensions = [
    len(vecteur)
    for vecteur in embeddings
]

dimensions_ok = all(
    dimension == 1024
    for dimension in dimensions
)

print(
    f"2. Dimensions des vecteurs : {dimensions} : "
    f"{'✅' if dimensions_ok else '❌'}"
)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

assert nombre_ok, (
    "Le nombre d'embeddings ne correspond pas "
    "au nombre de textes envoyés."
)

assert dimensions_ok, (
    "Au moins un embedding n'a pas 1024 dimensions."
)


print("\n✅ Test manuel Mistral réussi.")

# ===========================================================================
# Critère 3 : vérification empirique de l'ordre
# ===========================================================================

print(
    "\n3. Vérification de l'ordre "
    "(similarité cosinus avec le chunk n°2 réutilisé) :"
)

# On reprend exactement le texte du deuxième chunk
texte_reference = textes[1]

# On génère à nouveau son embedding
embedding_reference = generer_embeddings_lot(
    client,
    [texte_reference],
)[0]


def similarite_cosinus(
    vecteur_a: list[float],
    vecteur_b: list[float],
) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """

    a = np.array(vecteur_a, dtype=np.float32)
    b = np.array(vecteur_b, dtype=np.float32)

    return float(
        np.dot(a, b)
        / (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )
    )


similarites = [
    similarite_cosinus(
        embedding_reference,
        embedding,
    )
    for embedding in embeddings
]


for index, score in enumerate(similarites):

    indication = ""

    if index == 1:
        indication = " <-- attendu le plus élevé"

    print(
        f"   Chunk {index} : "
        f"{score:.6f}"
        f"{indication}"
    )


index_plus_proche = int(
    np.argmax(similarites)
)

ordre_ok = index_plus_proche == 1

print(
    f"\nIndex le plus proche : "
    f"{index_plus_proche} "
    f"(attendu : 1) : "
    f"{'✅' if ordre_ok else '❌'}"
)

assert ordre_ok, (
    "L'embedding du chunk n°2 "
    "n'est pas celui qui présente "
    "la plus forte similarité."
)

print("✅ Ordre des embeddings confirmé.")