import csv
import json
from collections import defaultdict
from pathlib import Path


FICHIER_RESULTATS = Path(
    "data/evaluation/resultats_evaluation.json"
)

FICHIER_RESUME = Path(
    "data/evaluation/resume_evaluation.csv"
)


def main() -> None:

    # ------------------------------------------------------------------
    # Chargement des résultats
    # ------------------------------------------------------------------

    with open(FICHIER_RESULTATS, "r", encoding="utf-8") as f:
        donnees = json.load(f)

    # Le fichier peut contenir directement une liste ou un dictionnaire
    # contenant cette liste.
    if isinstance(donnees, list):
        resultats = donnees

    elif isinstance(donnees, dict):

        if "resultats" in donnees:
            resultats = donnees["resultats"]

        elif "cas" in donnees:
            resultats = donnees["cas"]

        else:
            raise ValueError(
                "Impossible de trouver la liste des cas d'évaluation "
                f"dans {FICHIER_RESULTATS}.\n"
                f"Clés trouvées : {list(donnees.keys())}"
            )

    else:
        raise TypeError(
            "Structure JSON inattendue dans le fichier de résultats."
        )

    # ------------------------------------------------------------------
    # Calcul des scores
    # ------------------------------------------------------------------

    scores = []
    scores_par_categorie = defaultdict(list)

    print("\n" + "=" * 70)
    print("RÉSUMÉ DE L'ÉVALUATION RAG")
    print("=" * 70)

    print(
        f"{'ID':<4} "
        f"{'Catégorie':<25} "
        f"{'Score':>8}"
    )

    print("-" * 70)

    for cas in resultats:

        if not isinstance(cas, dict):
            continue

        score = cas.get("score_manuel")

        if score is None:
            continue

        categorie = cas.get(
            "categorie",
            "inconnue"
        )

        scores.append(float(score))
        scores_par_categorie[categorie].append(float(score))

        print(
            f"{str(cas.get('id', '-')):<4} "
            f"{categorie:<25} "
            f"{float(score):>8.2f}"
        )

    print("-" * 70)

    # ------------------------------------------------------------------
    # Score global
    # ------------------------------------------------------------------

    if not scores:
        print(
            "\n❌ Aucun score manuel trouvé."
        )
        return

    moyenne = sum(scores) / len(scores)

    print(
        f"\nScore moyen global : {moyenne:.3f}"
    )

    print(
        f"Pourcentage global : {moyenne * 100:.1f} %"
    )

    print(
        f"Nombre de cas évalués : {len(scores)}"
    )

    # ------------------------------------------------------------------
    # Scores par catégorie
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("RÉSULTATS PAR CATÉGORIE")
    print("=" * 70)

    resume_categories = []

    for categorie, valeurs in sorted(
        scores_par_categorie.items()
    ):

        moyenne_cat = sum(valeurs) / len(valeurs)

        print(
            f"{categorie:<25} "
            f"{moyenne_cat:.2f} "
            f"({moyenne_cat * 100:.1f} %) "
            f"- {len(valeurs)} cas"
        )

        resume_categories.append(
            {
                "categorie": categorie,
                "nombre_cas": len(valeurs),
                "score_moyen": round(
                    moyenne_cat,
                    3
                ),
                "pourcentage": round(
                    moyenne_cat * 100,
                    1
                ),
            }
        )

    # ------------------------------------------------------------------
    # Export CSV
    # ------------------------------------------------------------------

    FICHIER_RESUME.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        FICHIER_RESUME,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "categorie",
                "nombre_cas",
                "score_moyen",
                "pourcentage",
            ],
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(resume_categories)

    print("\n" + "=" * 70)

    print(
        f"✅ Résumé CSV sauvegardé : {FICHIER_RESUME}"
    )

    print(
        f"✅ Score global : "
        f"{moyenne * 10:.2f} / 10 "
        f"soit {moyenne * 100:.1f} %"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()