"""
Feature 5 : génération des embeddings Mistral.

Ce script charge les chunks produits lors de l'étape de chunking
et génère leurs représentations vectorielles avec Mistral.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral


# ===========================================================================
# 0. Configuration
# ===========================================================================

CHEMIN_SOURCE = Path("data/processed/events_chunks.json")

MODELE_EMBEDDING = "mistral-embed"

MAX_TENTATIVES = 5
DELAI_ATTENTE = 5


# ===========================================================================
# 1. Chargement de la clé API
# ===========================================================================

def charger_cle_api() -> str:
    """
    Charge la clé API Mistral depuis le fichier .env.
    """

    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY introuvable dans le fichier .env."
        )

    return api_key


# ===========================================================================
# 2. Chargement des chunks
# ===========================================================================

def charger_chunks(chemin: Path) -> list[dict]:
    """
    Charge les chunks depuis le fichier JSON produit
    par chunk_events.py.
    """

    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier de chunks introuvable : {chemin}"
        )

    with open(chemin, "r", encoding="utf-8") as fichier:
        chunks = json.load(fichier)

    if not isinstance(chunks, list):
        raise ValueError(
            "Le fichier de chunks doit contenir une liste JSON."
        )

    return chunks


# ===========================================================================
# 3. Génération des embeddings par lot
# ===========================================================================

def generer_embeddings_lot(
    client: Mistral,
    textes: list[str],
) -> list[list[float]]:
    """
    Envoie une liste de textes à l'API Mistral et retourne
    les vecteurs d'embedding correspondants, dans le même ordre.
    """

    if not textes:
        return []

    for tentative in range(1, MAX_TENTATIVES + 1):

        try:
            reponse = client.embeddings.create(
                model=MODELE_EMBEDDING,
                inputs=textes,
            )

            embeddings = [
                item.embedding
                for item in reponse.data
            ]

            if len(embeddings) != len(textes):
                raise RuntimeError(
                    f"Nombre d'embeddings inattendu : "
                    f"{len(embeddings)} reçus "
                    f"pour {len(textes)} textes."
                )

            return embeddings

        except Exception as erreur:

            print(
                f"⚠️ Tentative {tentative}/{MAX_TENTATIVES} : "
                f"{erreur}"
            )

            if tentative < MAX_TENTATIVES:
                print(
                    f"Nouvelle tentative dans "
                    f"{DELAI_ATTENTE} secondes..."
                )
                time.sleep(DELAI_ATTENTE)

    raise RuntimeError(
        "Impossible de générer les embeddings "
        f"après {MAX_TENTATIVES} tentatives."
    )
    
# ===========================================================================
# 4. Génération des embeddings pour tout le corpus, par lots
# ===========================================================================

def generer_tous_les_embeddings(
    client: Mistral,
    chunks: list[dict],
    taille_lot: int = 20,
) -> tuple[list[list[float]], list[dict]]:
    """
    Génère les embeddings de tous les chunks par lots successifs.

    Retourne deux listes alignées :
    - les vecteurs ;
    - les métadonnées des chunks correspondants.
    """

    tous_les_embeddings: list[list[float]] = []
    toutes_les_metadonnees: list[dict] = []

    nombre_lots = (
        len(chunks) + taille_lot - 1
    ) // taille_lot

    for numero_lot in range(nombre_lots):
        debut = numero_lot * taille_lot
        fin = debut + taille_lot

        lot = chunks[debut:fin]

        textes_lot = [
            chunk["texte"]
            for chunk in lot
        ]

        print(
            f"Lot {numero_lot + 1}/{nombre_lots} : "
            f"{len(textes_lot)} textes..."
        )

        embeddings_lot = generer_embeddings_lot(
            client,
            textes_lot,
        )

        if len(embeddings_lot) != len(lot):
            raise RuntimeError(
                f"Incohérence dans le lot {numero_lot + 1} : "
                f"{len(lot)} chunks mais "
                f"{len(embeddings_lot)} embeddings."
            )

        tous_les_embeddings.extend(
            embeddings_lot
        )

        toutes_les_metadonnees.extend(
            lot
        )

    return (
        tous_les_embeddings,
        toutes_les_metadonnees,
    )

    
# ===========================================================================
# 5. Sauvegarde
# ===========================================================================

CHEMIN_EMBEDDINGS = Path("data/processed/embeddings.npy")
CHEMIN_METADATA = Path("data/processed/embeddings_metadata.json")


def sauvegarder_embeddings(
    embeddings: list[list[float]],
    metadonnees: list[dict],
) -> None:
    """Sauvegarde les vecteurs (.npy) et les métadonnées (.json) séparément."""

    import numpy as np

    CHEMIN_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)

    tableau = np.array(embeddings, dtype=np.float32)
    np.save(CHEMIN_EMBEDDINGS, tableau)

    with CHEMIN_METADATA.open("w", encoding="utf-8") as fichier:
        json.dump(metadonnees, fichier, ensure_ascii=False, indent=2)

    print(f"\n✅ Vecteurs sauvegardés : {CHEMIN_EMBEDDINGS} (shape {tableau.shape})")
    print(f"✅ Métadonnées sauvegardées : {CHEMIN_METADATA}")


# ===========================================================================
# 6. Programme principal
# ===========================================================================

def main() -> None:
    api_key = charger_cle_api()

    client = Mistral(
        api_key=api_key
    )

    chunks = charger_chunks(
        CHEMIN_SOURCE
    )

    print(
        f"✅ {len(chunks)} chunks chargés "
        f"depuis {CHEMIN_SOURCE}"
    )

    embeddings, metadonnees = (
        generer_tous_les_embeddings(
            client,
            chunks,
            taille_lot=20,
        )
    )

    correspondance_ok = (
        len(chunks)
        == len(embeddings)
        == len(metadonnees)
    )

    if not correspondance_ok:
        raise RuntimeError(
            f"Incohérence détectée : "
            f"{len(chunks)} chunks, "
            f"{len(embeddings)} embeddings, "
            f"{len(metadonnees)} métadonnées."
        )

    dimension = (
        len(embeddings[0])
        if embeddings
        else 0
    )

    dimensions_ok = all(
        len(vecteur) == dimension
        for vecteur in embeddings
    )

    if not dimensions_ok:
        raise RuntimeError(
            "Les embeddings n'ont pas tous "
            "la même dimension."
        )

    sauvegarder_embeddings(
        embeddings,
        metadonnees,
    )

    taille_lot = 20

    nombre_lots = (
        len(chunks)
        + taille_lot
        - 1
    ) // taille_lot

    print("\n" + "=" * 60)
    print("RAPPORT DE GÉNÉRATION DES EMBEDDINGS")
    print("=" * 60)

    print(
        f"Chunks chargés              : "
        f"{len(chunks)}"
    )

    print(
        f"Lots traités                : "
        f"{nombre_lots}"
    )

    print(
        f"Embeddings générés          : "
        f"{len(embeddings)}"
    )

    print(
        f"Dimension                   : "
        f"{dimension}"
    )

    print(
        f"Dimensions homogènes        : "
        f"{'OK' if dimensions_ok else 'ERREUR'}"
    )

    print(
        f"Correspondance chunks/vect. : "
        f"{'OK' if correspondance_ok else 'ERREUR'}"
    )

    print("=" * 60)
    
if __name__ == "__main__":
    main()