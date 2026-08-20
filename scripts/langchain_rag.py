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
                    "score_similarite": resultat.get(
                        "score_similarite"
                    ),
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
1. Réponds uniquement à partir des événements fournis dans le contexte.
2. N'invente aucune information.
3. Pour toute recommandation, utilise comme titre principal uniquement
   le champ "Titre" d'un ÉVÉNEMENT du contexte.
4. Si la description d'un événement mentionne d'autres œuvres,
   expositions ou activités, tu peux les citer comme détails,
   mais ne les présente pas comme des événements indépendants.
5. Si une contrainte n'est pas explicitement confirmée dans le contexte,
   indique qu'elle n'est pas confirmée.
6. Cite le lieu et la date lorsqu'ils sont disponibles.
7. Réponds en français, de façon concise.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            PROMPT_SYSTEME,
        ),
        (
            "human",
            "CONTEXTE :\n{contexte}\n\n"
            "QUESTION :\n{question}",
        ),
    ]
)

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

# Section 3 : assemblage LCEL et point d'entrée

# ===========================================================================
# Chaîne LCEL complète
# ===========================================================================

def construire_chaine_rag() -> object:
    """
    Assemble la chaîne RAG complète avec LCEL :

    question
        -> retriever FAISS
        -> formatage du contexte
        -> prompt LangChain
        -> LLM Mistral
        -> texte final
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

    chaine = (
        {
            "contexte": retriever | formater_documents,
            "question": RunnablePassthrough(),
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

    reponse = chaine.invoke(question)

    print("\n" + "=" * 60)
    print("RÉPONSE (via LangChain)")
    print("=" * 60)
    print(reponse)
    print("=" * 60)


if __name__ == "__main__":
    main()