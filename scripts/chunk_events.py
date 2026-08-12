"""
Feature 4 : chunking des documents OpenAgenda.

Ce script :
- charge les événements pré-traités ;
- découpe les textes trop longs en plusieurs chunks ;
- conserve les limites de phrases autant que possible ;
- ajoute le titre à chaque chunk pour préserver le contexte ;
- conserve les métadonnées de l'événement ;
- sauvegarde les chunks dans data/processed/events_chunks.json.
"""

import json
import re
from pathlib import Path


# ===========================================================================
# 0. Configuration
# ===========================================================================

CHEMIN_SOURCE = Path("data/processed/events_clean.json")
CHEMIN_SORTIE = Path("data/processed/events_chunks.json")

# Si le texte est inférieur ou égal à ce seuil,
# il reste en un seul chunk.
SEUIL_CHUNKING = 1500

# Taille approximative recherchée pour les chunks.
TAILLE_CHUNK_CIBLE = 800


# ===========================================================================
# 1. Découpage du texte
# ===========================================================================

def decouper_en_phrases(texte: str) -> list[str]:
    """
    Découpe un texte en phrases.

    La coupure est effectuée après :
    - un point ;
    - un point d'exclamation ;
    - un point d'interrogation.
    """

    phrases = re.split(r"(?<=[.!?])\s+", texte)

    return [
        phrase.strip()
        for phrase in phrases
        if phrase.strip()
    ]

def decouper_phrase_longue(
    phrase: str,
    taille_max: int,
) -> list[str]:
    """
    Découpe une phrase trop longue sans couper les mots.
    """

    mots = phrase.split()

    morceaux = []
    morceau_courant = ""

    for mot in mots:
        candidat = (
            f"{morceau_courant} {mot}".strip()
        )

        if len(candidat) <= taille_max:
            morceau_courant = candidat
        else:
            if morceau_courant:
                morceaux.append(morceau_courant)

            morceau_courant = mot

    if morceau_courant:
        morceaux.append(morceau_courant)

    return morceaux


def decouper_en_chunks(texte: str) -> list[str]:
    """
    Découpe un document en chunks d'environ
    TAILLE_CHUNK_CIBLE caractères.

    Pour préserver le contexte entre deux chunks,
    la dernière phrase du chunk précédent est reprise
    dans le chunk suivant.
    """

    texte = texte.strip()

    # Aucun texte
    if not texte:
        return []

    # Document suffisamment court :
    # aucun découpage nécessaire.
    if len(texte) <= SEUIL_CHUNKING:
        return [texte]

    phrases = decouper_en_phrases(texte)

    chunks = []

    # On conserve les phrases du chunk courant dans une liste.
    chunk_courant = []

    longueur_courante = 0

    for phrase in phrases:

        # ---------------------------------------------------------------
        # Cas particulier :
        # une phrase seule est déjà plus longue que la taille cible.
        # ---------------------------------------------------------------

        if len(phrase) > TAILLE_CHUNK_CIBLE:

            # On sauvegarde d'abord le chunk en cours.
            if chunk_courant:
                chunks.append(
                    " ".join(chunk_courant)
                )

                chunk_courant = []
                longueur_courante = 0

            # Puis on découpe cette très longue phrase.
            morceaux = decouper_phrase_longue(
                phrase,
                TAILLE_CHUNK_CIBLE,
            )

            chunks.extend(morceaux)

            continue

        # ---------------------------------------------------------------
        # Vérification de la taille si on ajoute la phrase
        # ---------------------------------------------------------------

        longueur_avec_phrase = (
            longueur_courante
            + len(phrase)
            + (1 if chunk_courant else 0)
        )

        if longueur_avec_phrase <= TAILLE_CHUNK_CIBLE:

            chunk_courant.append(phrase)
            longueur_courante = longueur_avec_phrase

        else:
            # Le chunk courant est terminé.
            chunks.append(
                " ".join(chunk_courant)
            )

            # -----------------------------------------------------------
            # Chevauchement :
            # on reprend la dernière phrase du chunk précédent.
            # -----------------------------------------------------------

            CHEVAUCHEMENT_MAX = 150
            
            derniere_phrase = (
                chunk_courant[-1]
                if chunk_courant
                else ""
            )

            if len(derniere_phrase) <= CHEVAUCHEMENT_MAX:
                reprise = derniere_phrase
            else:
                reprise = ""
            
            chunk_courant = []

            if reprise:
                chunk_courant.append(reprise)

            chunk_courant.append(phrase)

        #    chunk_courant = []

        #    if derniere_phrase:
        #        chunk_courant.append(
        #            derniere_phrase
        #        )

        #    chunk_courant.append(phrase)

            longueur_courante = len(
                " ".join(chunk_courant)
            )

    # Ajouter le dernier chunk restant.
    if chunk_courant:
        chunks.append(
            " ".join(chunk_courant)
        )

    return chunks


# ===========================================================================
# 2. Transformation des événements en chunks
# ===========================================================================

"""
def creer_chunks_evenements(
    evenements: list[dict],
) -> list[dict]:

#    Transforme chaque événement en un ou plusieurs chunks.

#    Chaque chunk conserve :
#    - l'UID de l'événement ;
#    - son titre ;
#    - ses métadonnées ;
#    - son numéro de chunk ;
#    - son texte enrichi avec le titre.
#    

    resultat = []

    for evenement in evenements:

        texte = evenement.get(
            "texte_complet",
            "",
        ).strip()

        titre = evenement.get(
            "titre",
            "",
        ).strip()

        chunks = decouper_en_chunks(
            texte
        )

        for index, chunk in enumerate(chunks):

            # -----------------------------------------------------------
            # Conserver le contexte de l'événement dans chaque chunk.
            #
            # Le premier chunk contient déjà le titre car texte_complet
            # commence par "Titre : ...".
            #
            # Pour les chunks suivants, on ajoute explicitement le titre.
            # -----------------------------------------------------------

            if index == 0:
                texte_avec_contexte = chunk

            elif titre:
                texte_avec_contexte = (
                    f"Titre : {titre}. {chunk}"
                )

            else:
                texte_avec_contexte = chunk

            resultat.append(
                {
                    # Exemple :
                    # 45424319_0
                    # 45424319_1
                    "chunk_id": (
                        f"{evenement['uid']}_{index}"
                    ),

                    # UID original OpenAgenda
                    "uid": evenement["uid"],

                    # Position du chunk dans l'événement
                    "chunk_index": index,

                    # Nombre total de chunks de cet événement
                    "nombre_chunks": len(chunks),

                    # Métadonnées
                    "titre": titre,

                    "lieu": evenement.get(
                        "lieu",
                        "",
                    ),

                    "adresse": evenement.get(
                        "adresse",
                        "",
                    ),

                    "code_postal": evenement.get(
                        "code_postal",
                        "",
                    ),

                    "ville": evenement.get(
                        "ville",
                        "",
                    ),

                    "debut": evenement.get(
                        "debut",
                        "",
                    ),

                    "fin": evenement.get(
                        "fin",
                        "",
                    ),

                    "source_agenda": evenement.get(
                        "source_agenda",
                        "",
                    ),

                    # Texte qui sera ensuite vectorisé
                    "texte": texte_avec_contexte,
                }
            )

    return resultat
"""

def creer_chunks_evenements(evenements: list[dict]) -> list[dict]:
    """Transforme les événements en chunks avec métadonnées."""

    resultat = []

    for evenement in evenements:
        texte = evenement.get("texte_complet", "").strip()
        titre = evenement.get("titre", "").strip()

        chunks = decouper_en_chunks(texte)

        for index, chunk in enumerate(chunks):

            # Le premier chunk contient déjà le titre dans texte_complet.
            # Pour les suivants, on réinjecte le titre afin que chaque
            # chunk reste compréhensible lorsqu'il est retrouvé seul.
            if index == 0:
                texte_avec_contexte = chunk

            elif titre:
                texte_avec_contexte = f"Titre : {titre}. {chunk}"

            else:
                texte_avec_contexte = chunk

            resultat.append(
                {
                    "chunk_id": f"{evenement['uid']}_{index}",
                    "uid": evenement["uid"],
                    "chunk_index": index,
                    "nombre_chunks": len(chunks),
                    "titre": titre,
                    "texte": texte_avec_contexte,
                    "lieu": evenement.get("lieu", ""),
                    "adresse": evenement.get("adresse", ""),
                    "ville": evenement.get("ville", ""),
                    "debut": evenement.get("debut", ""),
                    "fin": evenement.get("fin", ""),
                    "source_agenda": evenement.get("source_agenda", ""),
                }
            )

    return resultat


# ===========================================================================
# 3. Programme principal
# ===========================================================================

def main() -> None:
    """Exécute le chunking complet du corpus."""

    # Vérification du fichier source
    if not CHEMIN_SOURCE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CHEMIN_SOURCE}"
        )

    # Chargement des événements
    with CHEMIN_SOURCE.open(
        "r",
        encoding="utf-8",
    ) as fichier:
        evenements = json.load(fichier)

    # Création des chunks
    chunks = creer_chunks_evenements(
        evenements
    )

    # Création du dossier si nécessaire
    CHEMIN_SORTIE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Sauvegarde
    with CHEMIN_SORTIE.open(
        "w",
        encoding="utf-8",
    ) as fichier:

        json.dump(
            chunks,
            fichier,
            ensure_ascii=False,
            indent=2,
        )

    # -----------------------------------------------------------------------
    # Rapport
    # -----------------------------------------------------------------------

    nb_evenements_decoupes = sum(
        1
        for evenement in evenements
        if len(
            evenement.get(
                "texte_complet",
                "",
            )
        ) > SEUIL_CHUNKING
    )

    print("=" * 60)
    print("CHUNKING DES ÉVÉNEMENTS")
    print("=" * 60)

    print(
        f"Événements source              : "
        f"{len(evenements)}"
    )

    print(
        f"Événements réellement découpés : "
        f"{nb_evenements_decoupes}"
    )

    print(
        f"Chunks générés                 : "
        f"{len(chunks)}"
    )

    print(
        f"Seuil de chunking              : "
        f"{SEUIL_CHUNKING} caractères"
    )

    print(
        f"Taille cible                   : "
        f"{TAILLE_CHUNK_CIBLE} caractères"
    )

    print(
        f"Fichier produit                : "
        f"{CHEMIN_SORTIE}"
    )

    print("=" * 60)

    # -----------------------------------------------------------------------
    # Aperçu
    # -----------------------------------------------------------------------

    print("\nAperçu des trois premiers chunks :")

    for chunk in chunks[:3]:

        print("\n---")
        print(
            f"Chunk ID : "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Titre    : "
            f"{chunk['titre']}"
        )

        print(
            f"Chunk    : "
            f"{chunk['chunk_index'] + 1}/"
            f"{chunk['nombre_chunks']}"
        )

        print(
            chunk["texte"][:400]
            + (
                "..."
                if len(chunk["texte"]) > 400
                else ""
            )
        )


if __name__ == "__main__":
    main()