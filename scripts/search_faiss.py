"""
Feature 7 : recherche par similarité dans l'index FAISS.

Charge l'index et les métadonnées, vectorise une requête avec Mistral,
puis affiche les top_k événements les plus proches (dédupliqués par uid).

Usage :
    python scripts/search_faiss.py "exposition d'art contemporain à Marseille"
"""

import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from mistralai.client import Mistral


# ===========================================================================
# Configuration
# ===========================================================================

CHEMIN_INDEX = Path("faiss_index/index.faiss")
CHEMIN_METADATA = Path("faiss_index/metadata.json")

MODELE_EMBEDDING = "mistral-embed"
TOP_K = 5
FACTEUR_SURECHANTILLONNAGE = 4


# ===========================================================================
# Déduplication par événement (uid)
# ===========================================================================

def dedupliquer_par_evenement(
    scores: np.ndarray,
    indices: np.ndarray,
    metadata: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Garde le meilleur chunk de chaque événement, en dédupliquant à la fois
    par UID (pour éviter deux chunks du même événement) et par titre
    normalisé (pour éviter deux événements distincts qui semblent
    identiques du point de vue de l'utilisateur, ex. même titre décliné
    sur deux lieux). Cette déduplication n'affecte que les résultats
    affichés -- l'index FAISS et le corpus source restent intacts.
    """

    resultats = []
    uids_deja_vus = set()
    titres_deja_vus = set()

    for score, idx in zip(scores, indices):
        if idx == -1:
            continue

        chunk = metadata[idx]
        uid = chunk.get("uid")

        if uid is None:
            uid = chunk.get("chunk_id")

        titre_normalise = chunk.get("titre", "").strip().casefold()

        if uid in uids_deja_vus:
            continue

        if titre_normalise and titre_normalise in titres_deja_vus:
            continue

        uids_deja_vus.add(uid)
        titres_deja_vus.add(titre_normalise)

        resultat = chunk.copy()
        resultat["score_similarite"] = float(score)
        resultats.append(resultat)

        if len(resultats) >= top_k:
            break

    return resultats


def main() -> None:
    # -----------------------------------------------------------------------
    # Requête utilisateur
    # -----------------------------------------------------------------------

    requete = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "exposition d'art contemporain à Marseille"
    )

    # -----------------------------------------------------------------------
    # Clé API Mistral
    # -----------------------------------------------------------------------

    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY introuvable dans le fichier .env."
        )

    client = Mistral(
        api_key=api_key
    )

    # -----------------------------------------------------------------------
    # Vérification des fichiers
    # -----------------------------------------------------------------------

    if not CHEMIN_INDEX.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {CHEMIN_INDEX}"
        )

    if not CHEMIN_METADATA.exists():
        raise FileNotFoundError(
            f"Métadonnées introuvables : {CHEMIN_METADATA}"
        )

    # -----------------------------------------------------------------------
    # Chargement FAISS + métadonnées
    # -----------------------------------------------------------------------

    index = faiss.read_index(
        str(CHEMIN_INDEX)
    )

    with CHEMIN_METADATA.open(
        "r",
        encoding="utf-8-sig",
    ) as fichier:
        metadata = json.load(fichier)

    if index.ntotal != len(metadata):
        raise RuntimeError(
            f"Incohérence : {index.ntotal} vecteurs FAISS "
            f"pour {len(metadata)} métadonnées."
        )

    # -----------------------------------------------------------------------
    # Embedding de la requête
    # -----------------------------------------------------------------------

    reponse = client.embeddings.create(
        model=MODELE_EMBEDDING,
        inputs=[requete],
    )

    vecteur = np.array(
        [reponse.data[0].embedding],
        dtype=np.float32,
    )

    faiss.normalize_L2(vecteur)

    # -----------------------------------------------------------------------
    # Recherche FAISS (sur-échantillonnée pour permettre la déduplication)
    # -----------------------------------------------------------------------

    nombre_a_recuperer = min(
        TOP_K * FACTEUR_SURECHANTILLONNAGE,
        index.ntotal,
    )

    scores, indices = index.search(
        vecteur,
        nombre_a_recuperer,
    )

    # -----------------------------------------------------------------------
    # Déduplication par événement
    # -----------------------------------------------------------------------

    resultats = dedupliquer_par_evenement(
        scores[0],
        indices[0],
        metadata,
        TOP_K,
    )

    # -----------------------------------------------------------------------
    # Affichage
    # -----------------------------------------------------------------------

    print(
        f"\nRequête : « {requete} »\n"
        + "=" * 60
    )

    for rang, resultat in enumerate(resultats, start=1):
        print(
            f"\n{rang}. {resultat['titre']} "
            f"(score : {resultat['score_similarite']:.4f})"
        )
        print(
            f"   Chunk ID : {resultat['chunk_id']}"
        )
        print(
            f"   Lieu     : {resultat.get('lieu', '')}"
        )
        print(
            f"   {resultat['texte'][:200]}..."
        )

    if len(resultats) < TOP_K:
        print(
            f"\n⚠️ Seulement {len(resultats)} événements distincts trouvés "
            f"sur {TOP_K} demandés (corpus limité ou requête peu couverte)."
        )


if __name__ == "__main__":
    main()