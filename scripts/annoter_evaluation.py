"""
Insère l'annotation manuelle (critères validés/non validés, score, commentaire)
dans le fichier de résultats d'évaluation existant, sans toucher aux autres champs.
"""

import json
from pathlib import Path

CHEMIN_RESULTATS = Path("data/evaluation/resultats_evaluation.json")

ANNOTATIONS = {
    1: {
        "criteres_valides": [
            "La réponse mentionne au moins un événement du type exposition",
            "La gratuité est explicitement confirmée dans la réponse, pas supposée",
            "Le lieu et la date de l'exposition sont indiqués",
        ],
        "criteres_non_valides": [],
        "score_manuel": 1.0,
        "commentaire_evaluateur": "La réponse propose plusieurs expositions, confirme explicitement leur gratuité et fournit le lieu ainsi que les dates.",
    },
    2: {
        "criteres_valides": [
            "La réponse distingue les événements confirmant explicitement 'danse contemporaine' de ceux qui n'évoquent que des performances proches",
            "Aucun événement n'est présenté comme certain s'il ne l'est pas dans le contexte",
        ],
        "criteres_non_valides": [],
        "score_manuel": 1.0,
        "commentaire_evaluateur": "Nuance exemplaire : 'Pleine Lune' proposé avec la réserve explicite 'pas de la danse contemporaine au sens strict'. Plus aucune invention de source externe après renforcement du prompt.",
    },
    3: {
        "criteres_valides": [
            "La réponse liste plusieurs activités si le corpus en contient plusieurs",
            "Le nom du lieu correspond bien au Mucem, pas à un autre musée",
        ],
        "criteres_non_valides": [],
        "score_manuel": 1.0,
        "commentaire_evaluateur": "Trois activités listées, toutes bien rattachées au Mucem, avec dates et détails cohérents.",
    },
    4: {
        "criteres_valides": [
            "Seuls les événements dont le public jeune/familial est explicitement mentionné sont proposés",
            "La réponse ne propose pas d'événement pour adultes non qualifié comme familial",
        ],
        "criteres_non_valides": [],
        "score_manuel": 1.0,
        "commentaire_evaluateur": "Un seul événement proposé, explicitement qualifié pour les 7 ans et plus. Réponse courte mais fondée.",
    },
    5: {
        "criteres_valides": [
            "La réponse raisonne correctement sur la date relative 'demain' par rapport à la date du jour",
            "Si aucun concert n'a lieu à cette date précise, la réponse le dit clairement plutôt que de proposer des concerts à d'autres dates comme réponse valide",
        ],
        "criteres_non_valides": [],
        "score_manuel": 1.0,
        "commentaire_evaluateur": "La date du 21/08/2026 est correctement déduite et l'absence de concert est affirmée sans ambiguïté.",
    },
    6: {
        "criteres_valides": [
            "La réponse signale que le critère 'plein air' n'est pas confirmé si le contexte ne le précise pas",
            "Les événements proposés restent thématiquement liés à la musique",
        ],
        "criteres_non_valides": [],
        "score_manuel": 1.0,
        "commentaire_evaluateur": "Aucun événement ne satisfaisant simultanément les critères musique et plein air n'est proposé. Le système préfère signaler l'absence de résultat plutôt que recommander un événement approximatif.",
    },
    7: {
        "criteres_valides": [
            "Tous les événements cités ont bien une date en septembre",
            "Aucun événement d'un autre mois n'est mentionné comme correspondant",
        ],
        "criteres_non_valides": [],
        "score_manuel": 1.0,
        "commentaire_evaluateur": "Deux événements corrects. Point mineur : Kouss.Kouss débute en août, mais chevauche bien septembre, donc l'inclusion reste défendable.",
    },
    8: {
        "criteres_valides": [
            "La réponse ne doit pas inventer de restaurant absent du contexte",
            "La réponse reste dans les informations explicitement présentes dans le contexte",
        ],
        "criteres_non_valides": [
            "Si un événement gastronomique organisé dans un restaurant est présent, la réponse doit préciser qu'il s'agit d'un événement du corpus et non d'une recommandation générale de restaurant",
        ],
        "score_manuel": 0.67,
        "commentaire_evaluateur": "La réponse précise bien 'pas un restaurant traditionnel', mais présente tout de même Kouss·Agneau comme une suggestion positive plutôt que de refuser franchement de répondre à une demande hors périmètre du corpus.",
    },
    9: {
        "criteres_valides": [
            "Tous les événements cités ont bien pour lieu la Friche la Belle de Mai",
            "Le type d'événement (exposition) est respecté",
        ],
        "criteres_non_valides": [],
        "score_manuel": 1.0,
        "commentaire_evaluateur": "Cinq expositions correctement listées, toutes situées au bon lieu. Aucun événement déjà terminé n'apparaît.",
    },
    10: {
        "criteres_valides": [
            "La réponse propose une sélection variée, pas un seul type d'événement",
        ],
        "criteres_non_valides": [
            "Les dates proposées correspondent bien à un week-end proche de la date d'exécution",
        ],
        "score_manuel": 0.5,
        "commentaire_evaluateur": "Sélection variée et bien détaillée, mais le calcul du 'prochain week-end' est erroné (annonce 27-28 septembre puis liste des événements du 19-20 septembre) -- limite connue du raisonnement temporel relatif complexe des LLM, à corriger en v2 par un calcul programmatique plutôt que délégué au modèle.",
    },
}


def main() -> None:
    with CHEMIN_RESULTATS.open("r", encoding="utf-8") as fichier:
        contenu = json.load(fichier)

    for resultat in contenu["resultats"]:
        annotation = ANNOTATIONS.get(resultat["id"])
        if annotation:
            resultat.update(annotation)

    with CHEMIN_RESULTATS.open("w", encoding="utf-8") as fichier:
        json.dump(contenu, fichier, ensure_ascii=False, indent=2)

    scores = [a["score_manuel"] for a in ANNOTATIONS.values()]
    moyenne = sum(scores) / len(scores)

    print(f"✅ {len(ANNOTATIONS)} cas annotés dans {CHEMIN_RESULTATS}")
    print(f"Score global : {sum(scores):.2f} / {len(scores)} soit {moyenne * 100:.1f} %")


if __name__ == "__main__":
    main()