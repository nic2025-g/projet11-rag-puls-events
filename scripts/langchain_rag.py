"""
Feature 9 : intégration LangChain de la chaîne RAG.

Enveloppe le pipeline de recherche FAISS existant dans un retriever
LangChain, et assemble la chaîne complète avec ChatMistralAI via LCEL.
"""
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_mistralai import ChatMistralAI
from pydantic import Field

from langchain_core.runnables import RunnableLambda

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from search_faiss import rechercher_evenements
from rag_chain import construire_contexte

# ===========================================================================
# Configuration
# ===========================================================================

MODELE_LLM = "mistral-small-latest"
TOP_K = 5

# Section 1 : le retriever personnalisé

# ===========================================================================
# Retriever LangChain personnalisé
# ===========================================================================

class RetrieverEvenementsFAISS(BaseRetriever):
    """
    Retriever LangChain qui délègue la recherche au pipeline FAISS
    existant.

    Il réutilise :
    - les embeddings Mistral ;
    - l'index FAISS ;
    - la déduplication par uid ;
    - la déduplication par titre.

    Il convertit ensuite les résultats en objets Document LangChain.
    """

    top_k: int = Field(
        default=TOP_K,
        ge=1,
    )

    def _get_relevant_documents(
        self,
        query: str,
    ) -> List[Document]:
        """
        Transforme une requête utilisateur en Documents LangChain.
        """

        query = query.strip()

        if not query:
            return []

        resultats = rechercher_evenements(
            query,
            top_k=self.top_k,
        )

        documents = []

        for resultat in resultats:

            document = Document(
                page_content=resultat.get(
                    "texte",
                    "",
                ),
                metadata={
                    "uid": resultat.get("uid"),
                    "chunk_id": resultat.get("chunk_id"),
                    "titre": resultat.get("titre"),
                    "lieu": resultat.get("lieu"),
                    "adresse": resultat.get("adresse"),
                    "debut": resultat.get("debut"),
                    "fin": resultat.get("fin"),
                    "score_similarite": resultat.get("score_similarite"),
                },
            )

            documents.append(document)

        return documents

# Section 2 : le prompt LangChain et le formatage de documents reccupere

# ===========================================================================
# Prompt LangChain
# ===========================================================================

PROMPT_SYSTEME = """
Tu es un assistant spécialisé dans les événements culturels à Marseille.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir des événements fournis dans le contexte ci-dessous.
2. N'invente jamais d'événement, de lieu, de date ou de détail absent du contexte.
3. Pour toute recommandation, utilise comme titre principal uniquement
   le champ "Titre" d'un événement du contexte.
4. Pour toute expression temporelle relative comme "demain" ou "ce week-end",
    calcule la période à partir de la date de référence fournie et ne remplace
    jamais cette période par une autre date trouvée dans le contexte.
5. Un événement ne doit être présenté comme correspondant à un critère
   demandé (gratuit, enfant, plein air, danse contemporaine, etc.) que si
   ce critère est explicitement confirmé dans le contexte.
6. Si un événement ne satisfait pas un critère demandé, ne le recommande
   pas comme résultat valide.
7. Pour une contrainte temporelle précise (demain, ce soir, ce week-end,
   un mois donné), ne présente comme résultats valides que les événements
   dont les dates correspondent réellement à cette période.
8. Si aucun événement ne satisfait tous les critères de la question,
   dis-le clairement. Tu peux éventuellement signaler une alternative,
   mais indique explicitement qu'elle ne correspond pas exactement
   à la demande.
9. Cite le lieu et la date lorsqu'ils sont disponibles.
10. Utilise la date de référence uniquement pour interpréter une expression
    temporelle relative présente dans la question.
11. Réponds en français, de façon naturelle et concise.
12. Ne transforme jamais un événement gastronomique organisé dans un restaurant
    en recommandation générale de restaurant. Présente-le comme un événement
    gastronomique du corpus.
13. Cite le nom des événements recommandés, avec leur lieu et leur date.
14. Réponds en français, de façon naturelle et concise.
15. Ne recommande jamais de lieu, d'événement, d'artiste ou de source externe qui n'est pas explicitement présent dans le contexte fourni, même s'il te semble pertinent ou si tu le connais par ailleurs.
16. Si le contexte ne permet pas de répondre à la question, dis-le clairement, sans chercher à combler le vide par une suggestion générale.
17. Si aucun événement ne correspond exactement à la période demandée (par exemple "ce week-end" ou "demain"), dis-le clairement d'abord. Tu peux ensuite proposer des événements à une date différente, mais tu dois alors préciser explicitement l'écart temporel avec la date demandée (par exemple : "aucun événement ce week-end, mais voici ce qui est prévu le week-end suivant, le X").
18. Pour toute question impliquant une date relative (aujourd'hui, demain, ce week-end...), calcule-la à partir de la date du jour indiquée ci-dessus, sans jamais redéfinir silencieusement la période demandée pour qu'elle corresponde aux événements disponibles.
"""

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        PROMPT_SYSTEME,
    ),
    (
        "human",
        "{instruction_date}\n\n"
        "IMPORTANT : si la question contient une expression temporelle relative, "
        "la date de référence ci-dessus est prioritaire sur les dates trouvées "
        "dans le contexte.\n\n"
        "CONTEXTE :\n{contexte}\n\n"
        "QUESTION :\n{question}",
    ),
])

# ===========================================================================
# Formatage des documents récupérés en contexte texte
# ===========================================================================

def formater_documents(
    documents: List[Document],
) -> str:
    """
    Transforme les Documents LangChain en texte structuré.

    On reconvertit les Documents vers le format attendu par
    construire_contexte(), afin de réutiliser la logique déjà testée.
    """

    resultats = []

    for doc in documents:
        resultats.append(
            {
                "uid": doc.metadata.get("uid"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "titre": doc.metadata.get("titre"),
                "lieu": doc.metadata.get("lieu"),
                "adresse": doc.metadata.get("adresse"),
                "debut": doc.metadata.get("debut"),
                "fin": doc.metadata.get("fin"),
                "texte": doc.page_content,
                "score_similarite": doc.metadata.get(
                    "score_similarite"
                ),
            }
        )

    return construire_contexte(
        resultats
    )


# ===========================================================================
# Gestion de la date de référence
# ===========================================================================

EXPRESSIONS_TEMPORELLES_RELATIVES = (
    "aujourd'hui",
    "demain",
    "ce soir",
    "ce week-end",
    "ce weekend",
    "cette semaine",
    "la semaine prochaine",
    "ce mois-ci",
)


def construire_instruction_date(entree: dict) -> str:
    """
    Fournit la date de référence au LLM uniquement lorsque la question
    contient une expression temporelle relative.
    """

    question = entree.get("question", "").casefold()
    date_reference = entree.get("date_reference")

    if not date_reference:
        return ""

    contient_expression_relative = any(
        expression in question
        for expression in EXPRESSIONS_TEMPORELLES_RELATIVES
    )

    if not contient_expression_relative:
        return ""

    if date_reference:
        return (
            f"La date de référence est le {date_reference}. "
            "Utilise cette date uniquement pour interpréter les expressions "
            "temporelles relatives présentes dans la question comme 'demain', 'ce week-enk', 'la semaine prochaine'."
        )

    return (
        "Aucune date de référence explicite n'a été fournie. "
        "N'invente pas de date."
    )

# Section 3 : assemblage LCEL et point d'entrée

# ===========================================================================
# Chaîne LCEL complète
# ===========================================================================

def construire_chaine_rag() -> object:
    """
    Assemble la chaîne RAG LangChain.

    La question originale est envoyée au retriever FAISS.
    La date de référence est utilisée uniquement dans le prompt final
    lorsqu'une expression temporelle relative doit être interprétée.
    """

    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY introuvable dans le fichier .env."
        )

    retriever = RetrieverEvenementsFAISS(
        top_k=TOP_K
    )

    llm = ChatMistralAI(
        model=MODELE_LLM,
        api_key=api_key,
    )

    # Extrait seulement la question pour la recherche FAISS
    recuperer_question = RunnableLambda(
        lambda entree: entree["question"]
    )

    # Construction du contexte :
    # dict -> question seule -> retriever -> formatage
    recuperer_contexte = (
        recuperer_question
        | retriever
        | RunnableLambda(formater_documents)
    )

    chaine = (
        {
            "contexte": recuperer_contexte,
            "question": RunnableLambda(
                lambda entree: entree["question"]
            ),
            "instruction_date": RunnableLambda(
                construire_instruction_date
            ),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chaine

# ===========================================================================
# Exécution en ligne de commande
# ===========================================================================

def main() -> None:
    question = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Je cherche une exposition d'art contemporain à Marseille."
    )

    chaine = construire_chaine_rag()

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(question)

    reponse = chaine.invoke(
        {
            "question": question,
            "date_reference": None,
        }
    )

    print("\n" + "=" * 60)
    print("RÉPONSE (via LangChain)")
    print("=" * 60)
    print(reponse)
    print("=" * 60)


if __name__ == "__main__":
    main()
    