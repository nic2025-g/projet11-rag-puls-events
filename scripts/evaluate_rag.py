"""
Évaluation du système RAG sur un jeu de questions annotées.

Le script :
- charge les questions de référence ;
- exécute la chaîne RAG LangChain ;
- enregistre les réponses générées ;
- prépare un rapport JSON exploitable pour une évaluation manuelle.
"""

import json
from datetime import datetime
from pathlib import Path

from langchain_rag import construire_chaine_rag


# ===========================================================================
# Configuration
# ===========================================================================

CHEMIN_JEU_TEST = Path(
    "data/evaluation/questions_reponses.json"
)

CHEMIN_RESULTATS = Path(
    "data/evaluation/resultats_evaluation.json"
)


# ===========================================================================
# Chargement du jeu de test
# ===========================================================================

def charger_jeu_test(
    chemin: Path,
) -> list[dict]:
    """Charge les questions annotées depuis le fichier JSON."""

    if not chemin.exists():
        raise FileNotFoundError(
            f"Jeu de test introuvable : {chemin}"
        )

    with chemin.open(
        "r",
        encoding="utf-8",
    ) as fichier:
        donnees = json.load(fichier)

    if not isinstance(donnees, list):
        raise ValueError(
            "Le jeu de test doit contenir une liste JSON."
        )

    return donnees


# ===========================================================================
# Exécution d'un cas d'évaluation
# ===========================================================================

def evaluer_cas(
    chaine,
    cas: dict,
) -> dict:
    """
    Exécute une question sur la chaîne RAG
    et retourne le résultat enrichi.
    """

    question = cas.get("question", "").strip()

    if not question:
        raise ValueError(
            f"Question vide pour le cas {cas.get('id')}"
        )
    
    date_reference = cas.get("date_reference")
    
    print(
        f"\nCas {cas.get('id')} "
        f"| {cas.get('categorie', 'sans catégorie')}"
    )

    print(
        f"Question : {question}"
    )

# ===========================================================================
# Transformation de la question 
# ===========================================================================

#    date_reference = cas.get("date_reference")

    entree_rag = {
        "question": question,
        "date_reference": date_reference,
    }

    reponse = chaine.invoke(entree_rag)


    resultat = {
        "id": cas.get("id"),
        "categorie": cas.get("categorie"),
        "question": question,
        "date_reference": date_reference,
        "reponse_reference": cas.get(
            "reponse_reference"
        ),
        "criteres_attendus": cas.get(
            "criteres_attendus",
            [],
        ),
        "reponse_generee": reponse,

        # À remplir lors de la revue humaine
        "criteres_valides": [],
        "criteres_non_valides": [],
        "score_manuel": None,
        "commentaire_evaluateur": "",
    }

    return resultat

# ===========================================================================
# Évaluation complète
# ===========================================================================

def executer_evaluation(
    jeu_test: list[dict],
) -> list[dict]:
    """Exécute tous les cas du jeu de test."""

    chaine = construire_chaine_rag()

    resultats = []

    total = len(jeu_test)

    for position, cas in enumerate(
        jeu_test,
        start=1,
    ):
        print(
            "\n" + "=" * 60
        )

        print(
            f"ÉVALUATION {position}/{total}"
        )

        print(
            "=" * 60
        )

        resultat = evaluer_cas(
            chaine,
            cas,
        )

        resultats.append(
            resultat
        )

        print(
            "\nRéponse générée :"
        )

        print(
            resultat["reponse_generee"]
        )

    return resultats


# ===========================================================================
# Sauvegarde
# ===========================================================================

def sauvegarder_resultats(
    resultats: list[dict],
    chemin: Path,
) -> None:
    """Sauvegarde les résultats au format JSON."""

    chemin.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    contenu = {
        "date_evaluation": (
            datetime.now().isoformat()
        ),
        "nombre_cas": len(resultats),
        "resultats": resultats,
    }

    with chemin.open(
        "w",
        encoding="utf-8",
    ) as fichier:
        json.dump(
            contenu,
            fichier,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\n✅ Résultats sauvegardés : "
        f"{chemin}"
    )


# ===========================================================================
# Programme principal
# ===========================================================================

def main() -> None:
    jeu_test = charger_jeu_test(
        CHEMIN_JEU_TEST
    )

    print(
        f"✅ {len(jeu_test)} cas "
        f"d'évaluation chargés."
    )

    resultats = executer_evaluation(
        jeu_test
    )

    sauvegarder_resultats(
        resultats,
        CHEMIN_RESULTATS,
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "ÉVALUATION TERMINÉE"
    )

    print(
        "=" * 60
    )

    print(
        f"Cas exécutés : {len(resultats)}"
    )

    print(
        "Les scores manuels restent à renseigner "
        "dans le fichier de résultats."
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
    