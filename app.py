'''
Ce code ajoute trois fonctionnalités essentielles :
- Connexion réelle au RAG : Streamlit transmet la question à langchain_rag.py.
- Chargement optimisé : @st.cache_resource évite de reconstruire la chaîne à chaque interaction.
- Gestion des erreurs : si Mistral retourne un 429, l'utilisateur voit un message compréhensible plutôt qu'un traceback.
'''

import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

# Permet d'importer les modules existants du dossier scripts
DOSSIER_SCRIPTS = Path(__file__).resolve().parent / "scripts"
sys.path.insert(0, str(DOSSIER_SCRIPTS))

from langchain_rag import construire_chaine_rag
from search_faiss import rechercher_evenements
from rag_chain import nettoyer_texte_pour_contexte


st.set_page_config(
    page_title="Puls-Events",
    page_icon="🎭",
    layout="centered",
)

st.title("🎭 Puls-Events")
st.subheader("Votre assistant culturel à Marseille")

st.write(
    "Découvrez les événements culturels à Marseille "
    "en posant simplement votre question."
)


@st.cache_resource
def charger_rag():
    return construire_chaine_rag()

def formater_date_affichage(date_iso):
    if not date_iso:
        return "Date non précisée"

    try:
        date = datetime.fromisoformat(str(date_iso))
        return date.strftime("%d/%m/%Y à %Hh%M")
    except (ValueError, TypeError):
        return str(date_iso)

def evenement_a_venir(evenement):
    """Vérifie qu'un événement n'est pas terminé."""

    date_fin = evenement.get("fin")

    if not date_fin:
        return False

    try:
        fin = datetime.fromisoformat(str(date_fin))
        maintenant = datetime.now(fin.tzinfo) if fin.tzinfo else datetime.now()
        return fin >= maintenant

    except (ValueError, TypeError):
        return False

def nettoyer_description_affichage(texte, titre):
    """Prépare la description d'un événement pour Streamlit."""

    description = nettoyer_texte_pour_contexte(
        texte or "",
        titre,
    )

    # Les champs structurés sont déjà affichés séparément.
    # On retire uniquement le suffixe de métadonnées.
    marqueur = ". Lieu : "

    if marqueur in description:
        description = description.rsplit(marqueur, 1)[0]

    return description.strip()
    
question = st.text_input(
    "Que recherchez-vous ?",
    placeholder="Exemple : Une exposition gratuite à Marseille",
)

if st.button("🔎 Rechercher", type="primary"):

    if not question.strip():
        st.warning("Veuillez saisir une question.")

    else:
        with st.spinner("Recherche des événements en cours..."):
            try:
                chaine = charger_rag()

                reponse = chaine.invoke(
                    {
                        "question": question.strip(),
                        "date_reference": datetime.now().strftime(
                            "%Y-%m-%d"
                        ),
                    }
                )

                st.markdown("### 🤖 Réponse de Puls-Events")
                st.write(reponse)

            except Exception as erreur:
                message = str(erreur)

                if "429" in message or "rate_limited" in message:
                    st.warning(
                        "Le service de génération Mistral est "
                        "temporairement limité. "
                        "Voici les événements retrouvés par "
                        "notre moteur de recherche FAISS."
                    )

                    try:
                        candidats = rechercher_evenements(
                            question.strip(),
                            top_k=30,
                        )

                        resultats = [
                            evenement
                            for evenement in candidats
                            if evenement_a_venir(evenement)
                        ][:5]
                        if not resultats:
                            st.info(
                                "Aucun événement n'a été trouvé."
                            )

                        for rang, evenement in enumerate(
                            resultats, start=1
                        ):
                            titre = evenement.get(
                                "titre", "Événement sans titre"
                            )

                            with st.expander(
                                f"{rang}. {titre}",
                                expanded=(rang == 1),
                            ):
                                st.write(
                                    "**Lieu :**",
                                    evenement.get("lieu")
                                    or "Non précisé",
                                )

                                st.write(
                                    "**Début :**",
                                    formater_date_affichage(evenement.get("debut"))
                                )

                                st.write(
                                    "**Fin :**",
                                    formater_date_affichage(evenement.get("fin"))
                                )

                                description = nettoyer_description_affichage(
                                    evenement.get("texte"),
                                    titre,
                                )

                                st.write(
                                        "**Description :**",
                                        description or "Non disponible",
                                )

                    except Exception:
                        st.error(
                            "La recherche FAISS est également "
                            "indisponible pour le moment."
                        )

                else:
                    st.error(
                        "Une erreur est survenue pendant "
                        "la recherche des événements."
                    )
                    st.exception(erreur)