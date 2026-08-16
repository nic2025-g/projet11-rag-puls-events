"""
Feature 6 : construction de l'index FAISS.

Ce script charge les embeddings générés par Mistral et construit
un index FAISS permettant la recherche par similarité.
"""

import json
from pathlib import Path

import faiss
import numpy as np

#===========================================================================
# 0. Configuration
#===========================================================================

CHEMIN_EMBEDDINGS = Path("data/processed/embeddings.npy")
CHEMIN_METADATA = Path("data/processed/embeddings_metadata.json")
CHEMIN_INDEX = Path("faiss_index/index.faiss")
CHEMIN_INDEX_METADATA = Path("faiss_index/metadata.json")

#===========================================================================
# 1. Chargement des embeddings et métadonnées
#===========================================================================

# On vérifie que le tableau est en float32, à 2 dimensions, et sans valeurs non finies.

def charger_embeddings(chemin: Path) -> np.ndarray:
    """Charge et valide le tableau numpy des embeddings."""

    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier d'embeddings introuvable : {chemin}"
        )

    vecteurs = np.load(chemin)

    if vecteurs.ndim != 2:
        raise ValueError(
            f"Embeddings invalides : shape {vecteurs.shape}"
        )

    vecteurs = vecteurs.astype(np.float32)

    if not np.isfinite(vecteurs).all():
        raise ValueError(
            "Les embeddings contiennent des valeurs NaN ou infinies."
        )

    print(
        f"✅ Embeddings chargés : "
        f"shape {vecteurs.shape}, dtype {vecteurs.dtype}"
    )

    return vecteurs

# On vérifie si le JSON contient bien une liste

def charger_metadata(chemin: Path) -> list[dict]:
    """Charge les métadonnées associées aux embeddings."""

    if not chemin.exists():
        raise FileNotFoundError(
            f"Fichier de métadonnées introuvable : {chemin}"
        )

    with chemin.open("r", encoding="utf-8-sig") as fichier:
        metadata = json.load(fichier)

    if not isinstance(metadata, list):
        raise ValueError(
            "Le fichier de métadonnées doit contenir une liste JSON."
        )

    print(
        f"✅ Métadonnées chargées : "
        f"{len(metadata)} entrées"
    )

    return metadata
# Controle 

#if len(vecteurs) != len(metadata):
#    raise RuntimeError(
#        f"Incohérence : {len(vecteurs)} vecteurs "
#        f"pour {len(metadata)} métadonnées."
 #   )
    
# ===========================================================================
# 2. Validation des données
# ===========================================================================

# ===========================================================================
# 2. Validation des données
# ===========================================================================

def valider_donnees(
    vecteurs: np.ndarray,
    metadata: list[dict],
) -> None:
    """Vérifie la cohérence entre les embeddings et leurs métadonnées."""

    if len(vecteurs) != len(metadata):
        raise ValueError(
            f"Incohérence : {len(vecteurs)} vecteurs "
            f"pour {len(metadata)} métadonnées."
        )

    if len(vecteurs) == 0:
        raise ValueError(
            "Aucun embedding à indexer."
        )

    print(
        f"✅ Correspondance vecteurs/métadonnées : "
        f"{len(vecteurs)} / {len(metadata)}"
    )

# ===========================================================================
# 3. Construction de l'index FAISS
# ===========================================================================

def construire_index(vecteurs: np.ndarray) -> faiss.Index:
    """
    Construit un index FAISS utilisant le produit scalaire
    sur des vecteurs normalisés.

    Avec des vecteurs normalisés L2, le produit scalaire
    correspond à la similarité cosinus.
    """

    vecteurs_normalises = np.ascontiguousarray(
        vecteurs.copy(),
        dtype=np.float32,
    )

    faiss.normalize_L2(vecteurs_normalises)

    dimension = vecteurs_normalises.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vecteurs_normalises)

    print(f"✅ Index FAISS construit")
    print(f"   Dimension       : {dimension}")
    print(f"   Vecteurs indexés : {index.ntotal}")

    return index


#===========================================================================
# 4. Sauvegarde de l'index et des métadonnées
#===========================================================================

def sauvegarder_index(
    index: faiss.Index,
    metadata: list[dict],
) -> None:
    """Sauvegarde l'index FAISS et les métadonnées associées."""

    CHEMIN_INDEX.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(CHEMIN_INDEX),
    )

    with CHEMIN_INDEX_METADATA.open(
        "w",
        encoding="utf-8",
    ) as fichier:
        json.dump(
            metadata,
            fichier,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"✅ Index FAISS sauvegardé : "
        f"{CHEMIN_INDEX}"
    )

    print(
        f"✅ Métadonnées sauvegardées : "
        f"{CHEMIN_INDEX_METADATA}"
    )

# ===========================================================================
# 5. Programme principal
# ===========================================================================

def main() -> None:
    """Charge les données, les valide et construit l'index FAISS."""

    vecteurs = charger_embeddings(CHEMIN_EMBEDDINGS)
    metadata = charger_metadata(CHEMIN_METADATA)

    valider_donnees(
        vecteurs,
        metadata,
    )

    index = construire_index(vecteurs)
    
    sauvegarder_index(
        index,
        metadata,
    )

    print("\n" + "=" * 60)
    print("RAPPORT DE CONSTRUCTION DE L'INDEX FAISS")
    print("=" * 60)
    print(f"Embeddings chargés : {len(vecteurs)}")
    print(f"Métadonnées        : {len(metadata)}")
    print(f"Dimension          : {vecteurs.shape[1]}")
    print(f"Vecteurs indexés   : {index.ntotal}")
    print("=" * 60)


if __name__ == "__main__":
    main()