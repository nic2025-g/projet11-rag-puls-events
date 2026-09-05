# Puls-Events — POC RAG pour la recommandation d'événements culturels à Marseille

Assistant conversationnel qui répond en langage naturel à des questions sur les événements
culturels de Marseille, à partir d'une base de données réelle et à jour (OpenAgenda),
vectorisée et interrogée via une architecture RAG (Retrieval-Augmented Generation).

> Preuve de concept — Projet 11, formation Data Engineering, OpenClassrooms.

---

## Sommaire

- [Aperçu](#aperçu)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration des clés API](#configuration-des-clés-api)
- [Reconstruire la base vectorielle](#reconstruire-la-base-vectorielle)
- [Utilisation du chatbot](#utilisation-du-chatbot)
- [Lancement des tests](#lancement-des-tests)
- [Intégration continue](#intégration-continue)
- [Évaluation du système](#évaluation-du-système)
- [Structuration du dépôt](#structuration-du-dépôt)
- [Résultats du POC](#résultats-du-poc)
- [Documentation complémentaire](#documentation-complémentaire)

---

## Aperçu

```
Question utilisateur
    → Embedding Mistral de la question
    → Recherche par similarité dans FAISS
    → Déduplication (uid, titre)
    → Construction du contexte
    → Génération de la réponse (LangChain + ChatMistralAI)
    → Réponse fondée sur le contexte, ou refus explicite
```

Le système s'appuie sur trois briques technologiques principales :

| Brique | Rôle |
|---|---|
| **[LangChain](https://python.langchain.com/)** | Orchestration du retriever, du prompt et du modèle de génération |
| **[Mistral AI](https://mistral.ai/)** | Génération des embeddings (`mistral-embed`) et de la réponse (`mistral-small-latest`) |
| **[FAISS](https://github.com/facebookresearch/faiss)** | Indexation vectorielle et recherche par similarité cosinus |

Source des données : [OpenAgenda](https://openagenda.com/), via son API officielle
(`api.openagenda.com`), interrogée sur trois agendas complémentaires couvrant Marseille.

---

## Architecture

```
OpenAgenda (3 agendas)
    │
    ▼
fetch_openagenda.py          → collecte brute, pagination, retry
    │
    ▼
preprocess_events.py         → filtrage ville / métier / temporel, nettoyage
    │
    ▼
chunk_events.py              → découpage conditionnel des textes longs
    │
    ▼
generate_embeddings.py       → vectorisation (Mistral, mistral-embed)
    │
    ▼
build_faiss_index.py         → index FAISS (IndexFlatIP, similarité cosinus)
    │
    ▼
langchain_rag.py             → retriever LangChain + prompt + ChatMistralAI
    │
    ▼
Réponse conversationnelle fondée sur le contexte
```

Chaque étape est un script Python indépendant et rejouable. **La base vectorielle
entière peut être reconstruite à la demande**, sans dépendance à un état intermédiaire
non versionné (voir [Reconstruire la base vectorielle](#reconstruire-la-base-vectorielle)).

---

## Installation

### Prérequis

- Python 3.13 ou supérieur
- Un compte [Mistral AI](https://console.mistral.ai) (plan gratuit *Experiment* suffisant)
- Un compte [OpenAgenda](https://openagenda.com) avec une clé API publique

### Mise en place de l'environnement

```powershell
# Cloner le dépôt
git clone git@github.com:nic2025-g/projet11-rag-puls-events.git
cd projet11-rag-puls-events

# Créer et activer l'environnement virtuel
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux

# Installer les dépendances
pip install -r requirements.txt
```

---

## Configuration des clés API

Copier `.env.example` vers `.env`, puis renseigner vos propres clés :

```powershell
Copy-Item .env.example .env
```

```dotenv
MISTRAL_API_KEY=votre_cle_mistral
OPENAGENDA_API_KEY=votre_cle_publique_openagenda
```

- **Clé Mistral** : Console Mistral → *Clés API* → *Créer une clé*.
- **Clé OpenAgenda** : compte OpenAgenda → paramètres du compte → clé publique.

Le fichier `.env` n'est jamais versionné (voir `.gitignore`).

---

## Reconstruire la base vectorielle

L'ensemble du pipeline est rejouable en une suite de commandes. Chaque script lit la
sortie du précédent ; aucune étape manuelle n'est nécessaire.

```powershell
# 1. Collecte des événements depuis OpenAgenda (3 agendas)
python scripts\fetch_openagenda.py

# 2. Nettoyage, filtrage géographique / métier / temporel
python scripts\preprocess_events.py

# 3. Découpage conditionnel des textes longs
python scripts\chunk_events.py

# 4. Génération des embeddings Mistral
python scripts\generate_embeddings.py

# 5. Construction de l'index FAISS
python scripts\build_faiss_index.py
```

À l'issue de ces cinq étapes, `faiss_index/` contient l'index et ses métadonnées,
prêts à être interrogés. Le volume final varie d'une exécution à l'autre : OpenAgenda
est une source vivante (nouveaux événements publiés, anciens expirés).

---

## Utiliser le chatbot

En ligne de commande, avec la chaîne LangChain complète :

```powershell
python scripts\langchain_rag.py "Je cherche une exposition gratuite à Marseille"
```

Sans argument, une question par défaut est utilisée :

```powershell
python scripts\langchain_rag.py
```

Une version plus simple, sans passer par LangChain (utile pour déboguer le seul
retrieval FAISS), est disponible via :

```powershell
python scripts\search_faiss.py "concert de musique en plein air"
```

---

## Lancer les tests

```powershell
# Lancer l'ensemble de la suite de tests
pytest -v
```

La suite compte 33 tests, répartis sur :

| Fichier | Portée |
|---|---|
| `tests/test_chunking.py` | Découpage conditionnel des textes |
| `tests/test_embeddings.py` | Génération d'embeddings par lots (client Mistral simulé) |
| `tests/test_search_faiss.py` | Recherche et déduplication FAISS |
| `tests/test_rag_chain.py` | Construction du contexte, génération (client simulé) |
| `tests/test_langchain_rag.py` | Retriever et chaîne LangChain (mocks) |
| `tests/test_donnees_vectorisees.py` | **Conformité des données réellement indexées** : ville = Marseille, événements de moins d'un an, champs minimaux présents |
| `tests/test_faiss.py` | Test historique de l'environnement |
| `tests/test_preprocessing_sample.py` | Contrôle du chargement et du filtrage géographique du pré-processing sur un échantillon OpenAgenda versionné |

Aucun de ces tests n'effectue d'appel réseau réel, à l'exception implicite de
`test_donnees_vectorisees.py` qui nécessite qu'un index FAISS existe déjà
(sinon il est automatiquement ignoré — `pytest.skip`).

---

## Intégration continue

Le projet intègre un pipeline CI avec **GitHub Actions**, défini dans
`.github/workflows/tests.yml`.

Il est déclenché automatiquement :

- lors des `push` sur `main`, `develop` et les branches `feature/**` ;
- lors des `pull_request` vers `main` ou `develop`.

La CI s'exécute sous Ubuntu avec Python 3.13, installe les dépendances depuis
`requirements.txt`, puis lance automatiquement :

```powershell
pytest -v
```

Le resultat contient 33 tests, tous validés en environnement local lorsque
les artefacts vectoriels ont été reconstruits.

Dans GitHub Actions, le dernier pipeline valide :

- 29 tests réussis ;
- 4 tests ignorés (skipped).

Les quatre tests ignorés correspondent aux contrôles d'intégration de
tests/test_donnees_vectorisees.py. Ils vérifient les données réellement associées
à l'index FAISS et nécessitent donc les artefacts locaux faiss_index/ et
data/processed/, qui sont volontairement non versionnés car entièrement
régénérables.

Afin de conserver un contrôle du pré-processing dans la CI sans dépendre d'API
externes, tests/test_preprocessing_sample.py utilise l'échantillon versionné
data/samples/sample_events_raw.json pour vérifier automatiquement le chargement
des données et le filtrage géographique sur Marseille.

## Évaluer le système

Un jeu de 10 questions annotées manuellement permet une évaluation quantitative,
au-delà des tests automatisés :

```powershell
# Exécute les 10 scénarios sur la chaîne RAG et sauvegarde les réponses générées
python scripts\evaluate_rag.py

# Complète l'annotation manuelle (critères validés, score) dans le fichier de résultats
python scripts\annoter_evaluation.py

# Génère un résumé par catégorie (CSV + affichage console)
python scripts\resume_evaluation.py
```

Fichiers concernés :
- `data/evaluation/questions_reponses.json` — le jeu de questions et critères de référence
- `data/evaluation/resultats_evaluation.json` — les réponses générées et leur annotation
- `data/evaluation/resume_evaluation.csv` — le résumé des scores par catégorie

---

## Structure du dépôt

```
projet11-rag-puls-events/
├── scripts/                 Scripts du pipeline (collecte → RAG)
├── tests/                   Suite de tests pytest (33 tests)
├── data/
│   ├── raw/                 Données brutes OpenAgenda (non versionné, régénérable)
│   ├── processed/           Données nettoyées, chunks, embeddings (non versionné)
│   ├── samples/              Échantillon de test versionné
│   └── evaluation/          Jeu de questions annoté et résultats (versionné)
├── faiss_index/              Index vectoriel et métadonnées (non versionné, régénérable)
├── docs/                     Documentation complémentaire
│
├── .github/
│   └── workflows/
│       └── tests.yml             Pipeline CI GitHub Actions
├── .env.example               Modèle de configuration des clés API
├── requirements.txt           Dépendances Python
└── pytest.ini                 Configuration de la suite de tests

```

Les fichiers volumineux et entièrement régénérables (données brutes, embeddings,
index FAISS) sont exclus du suivi Git — voir `.gitignore`. Seuls le code, la
configuration et les artefacts d'évaluation (rédigés à la main) sont versionnés.

---

## Résultats du POC

| Indicateur | Valeur |
|---|---|
| Agendas OpenAgenda interrogés | 3 |
| Événements retenus après filtrage | ~216 (variable selon la date de collecte) |
| Chunks vectorisés | ~403 |
| Dimension des embeddings | 1024 (`mistral-embed`) |
| Tests automatisés en local| 33 / 33 réussis |
| CI GitHub Actions | 29 réussis / 4 ignorés |
| Score d'évaluation (10 scénarios annotés) | 91,7 % |

Détail complet, choix techniques, limites identifiées et recommandations pour une
version de production : voir le rapport technique.

---

## Documentation complémentaire

- **Rapport technique** — architecture détaillée, choix techniques, résultats,
  limites et recommandations (`rapport_technique.docx`).
- **Présentation** — support de soutenance (`puls_events.pptx`).
- **Journaux de bord** — trace détaillée de chaque étape de développement, incidents
  rencontrés et corrigés (dossier `docs/`, non versionné — conservés en local).

---

## Licence et cadre

Projet réalisé dans le cadre de la formation Data Engineering d'OpenClassrooms.
Les données utilisées proviennent d'OpenAgenda et restent soumises à leurs conditions
d'utilisation respectives.
