"""
Feature 8 : chaîne RAG complète.

Construit le contexte à partir des résultats FAISS, puis interroge
le LLM Mistral pour produire une réponse fondée sur ce contexte.
"""

import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from mistralai.client import Mistral

from search_faiss import rechercher_evenements

date_reference = datetime.now().strftime("%d/%m/%Y")


# ===========================================================================
# Configuration
# ===========================================================================

MODELE_LLM = "mistral-small-latest"
TOP_K = 5


# ===========================================================================
# Formatage des dates
# ===========================================================================

def formater_date(date_iso: str) -> str:
    """Convertit une date ISO en format lisible."""

    if not date_iso:
        return "date non précisée"

    try:
        date_convertie = datetime.fromisoformat(date_iso)
        return date_convertie.strftime("%d/%m/%Y")

    except (ValueError, TypeError):
        return str(date_iso)


# ===========================================================================
# Nettoyage du texte pour le contexte
# ===========================================================================

def nettoyer_texte_pour_contexte(
    texte: str,
    titre: str,
) -> str:
    """
    Retire les préfixes déjà présentés séparément dans le contexte.
    """

    if not texte:
        return ""

    texte = texte.strip()

    prefixe_titre = f"Titre : {titre}."

    if texte.startswith(prefixe_titre):
        texte = texte[len(prefixe_titre):].strip()

    if texte.startswith("Description :"):
        texte = texte[len("Description :"):].strip()

    return texte


# ===========================================================================
# Construction du contexte
# ===========================================================================

def construire_contexte(resultats: list[dict]) -> str:
    """Transforme les résultats FAISS en contexte structuré pour le LLM."""

    if not resultats:
        return "Aucun événement pertinent n'a été trouvé."

    blocs = []

    for rang, resultat in enumerate(resultats, start=1):

        titre = resultat.get("titre") or "Titre inconnu"

        date_debut = formater_date(
            resultat.get("debut", "")
        )

        date_fin = formater_date(
            resultat.get("fin", "")
        )

        texte = nettoyer_texte_pour_contexte(
            resultat.get("texte", ""),
            titre,
        )

        bloc = (
            f"ÉVÉNEMENT {rang}\n"
            f"Titre : {titre}\n"
            f"Lieu : {resultat.get('lieu') or 'Lieu non précisé'}\n"
            f"Adresse : {resultat.get('adresse') or 'Adresse non précisée'}\n"
            f"Début : {date_debut}\n"
            f"Fin : {date_fin}\n"
            f"Informations : {texte or 'Description non disponible'}"
        )

        blocs.append(bloc)

    return "\n\n".join(blocs)


# ===========================================================================
# Génération de la réponse avec Mistral
# ===========================================================================

def generer_reponse_rag(
    client: Mistral,
    question: str,
    contexte: str,
) -> str:
    
    """
    Génère une réponse à partir uniquement du contexte récupéré.
    """

    message_systeme = (
        "Tu es un assistant spécialisé dans les événements culturels à Marseille. "
        f"La date de référence est le {date_reference}. "
        "Interprète 'aujourd'hui', 'demain', 'ce soir', etc. à partir de cette date. "
        "Réponds uniquement à partir des informations explicitement présentes "
        "dans le contexte fourni. "
        "N'invente aucune information et ne déduis pas qu'un événement respecte "
        "une contrainte si le contexte ne le confirme pas clairement."
    )
    message_utilisateur = (
        f"QUESTION UTILISATEUR :\n{question}\n\n"
        f"CONTEXTE :\n{contexte}"
    )

    reponse = client.chat.complete(
        model=MODELE_LLM,
        messages=[
            {
                "role": "system",
                "content": message_systeme,
            },
            {
                "role": "user",
                "content": message_utilisateur,
            },
        ],
    )

    return reponse.choices[0].message.content


# ===========================================================================
# Chaîne RAG complète
# ===========================================================================

def executer_rag(
    question: str,
    top_k: int = TOP_K,
) -> str:
    """Exécute retrieval + construction du contexte + génération."""

    question = question.strip()

    if not question:
        raise ValueError("La question ne peut pas être vide.")

    # 1. Retrieval FAISS
    resultats = rechercher_evenements(
        question,
        top_k=top_k,
    )

    # 2. Construction du contexte
    contexte = construire_contexte(
        resultats
    )

    # 3. Client Mistral
    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY introuvable dans le fichier .env."
        )

    client = Mistral(
        api_key=api_key
    )

    # 4. Génération
    return generer_reponse_rag(
        client,
        question,
        contexte,
    )


# ===========================================================================
# Exécution en ligne de commande
# ===========================================================================

def main() -> None:

    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Je cherche une exposition d'art contemporain à Marseille."
    )

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    reponse = executer_rag(
        question
    )

    print("\n" + "=" * 60)
    print("RÉPONSE RAG")
    print("=" * 60)
    print(reponse)
    print("=" * 60)


if __name__ == "__main__":
    main()