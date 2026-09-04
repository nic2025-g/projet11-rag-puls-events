# Projet 11 — Chatbot RAG de recommandation d'événements culturels à Marseille

## Présentation

Ce projet a été réalisé dans le cadre du parcours **Expert en ingénierie des données** d'OpenClassrooms.

L'objectif est de développer un **Proof of Concept (POC) de chatbot RAG** (*Retrieval-Augmented Generation*) capable de recommander des événements culturels à Marseille à partir de données issues d'**OpenAgenda**.

Le système combine :

- la collecte de données depuis OpenAgenda ;
- le nettoyage et le filtrage des événements ;
- le découpage conditionnel des textes en chunks ;
- la génération d'embeddings avec Mistral AI ;
- l'indexation vectorielle avec FAISS ;
- la recherche sémantique ;
- la génération de réponses avec Mistral ;
- l'orchestration du pipeline RAG avec LangChain.

---

## Architecture du projet

Le pipeline principal est le suivant :

```text
OpenAgenda
    │
    ▼
Collecte multi-sources
    │
    ▼
Prétraitement et filtrage
    │
    ▼
216 événements
    │
    ▼
Chunking conditionnel
    │
    ▼
403 chunks
    │
    ▼
Embeddings Mistral
    │
    ▼
403 vecteurs × 1024 dimensions
    │
    ▼
Index FAISS
    │
    ▼
Retriever LangChain
    │
    ▼
Prompt + Mistral
    │
    ▼
Réponse utilisateur
```

Le système sépare volontairement la recherche sémantique de la génération de la réponse.

La question originale de l'utilisateur est envoyée au retriever FAISS. Les informations nécessaires à l'interprétation des expressions temporelles sont ajoutées au niveau du prompt et non à la requête vectorielle afin de ne pas perturber la recherche sémantique.

---

## Sources de données

Les événements sont collectés depuis plusieurs agendas OpenAgenda :

- Aix-Marseille-Provence Métropole ;
- Musées de Marseille ;
- gmem-CNCM-marseille.

La collecte multi-sources permet d'élargir la couverture du corpus culturel tout en conservant la traçabilité de l'agenda d'origine.

Lors du dernier traitement utilisé pour le POC :

```text
2 596 événements collectés
        ↓
1 261 événements situés à Marseille
        ↓
370 événements après filtre métier
        ↓
216 événements après filtre temporel
```

Le corpus final contient donc **216 événements culturels** utilisés pour construire la base vectorielle.

---

## Prétraitement des données

Le prétraitement applique notamment :

- un filtrage géographique sur Marseille ;
- un filtrage métier des agendas non pertinents ;
- une validation des champs essentiels ;
- une déduplication des événements ;
- un filtrage temporel ;
- une normalisation du contenu textuel ;
- la conservation des métadonnées utiles au RAG.

La fenêtre temporelle utilisée couvre une période totale de **365 jours** :

- 60 jours dans le passé ;
- 305 jours dans le futur.

Le filtrage tient compte du chevauchement de la période d'un événement avec cette fenêtre. Un événement ayant commencé avant la date minimale peut donc être conservé s'il est encore en cours pendant la période étudiée.

Les paramètres temporels du prétraitement sont enregistrés afin de rendre les contrôles reproductibles.

---

## Chunking

Le découpage des documents est conditionnel.

Les événements courts sont conservés dans un seul document tandis que les textes longs sont découpés en plusieurs chunks.

Paramètres principaux :

```text
Seuil de découpage : 1 500 caractères
Taille cible       : environ 800 caractères
Événements sources : 216
Événements découpés: 55
Chunks produits    : 403
```

Le titre de l'événement est conservé dans les chunks afin de maintenir le contexte lors de la recherche vectorielle.

---

## Embeddings

Les représentations vectorielles sont générées avec le modèle :

```text
mistral-embed
```

Les chunks sont envoyés à l'API Mistral par lots.

Résultat obtenu :

```text
Chunks              : 403
Embeddings           : 403
Dimension            : 1024
Nombre de lots       : 21
```

Les vecteurs sont enregistrés localement avant la construction de l'index FAISS.

---

## Base vectorielle FAISS

Les embeddings sont indexés avec **FAISS** afin d'effectuer une recherche par similarité.

La base contient :

```text
403 vecteurs
1024 dimensions par vecteur
```

Les métadonnées associées aux chunks sont sauvegardées séparément.

Lors d'une recherche, la question utilisateur est elle-même transformée en embedding avec Mistral puis comparée aux vecteurs présents dans l'index.

Une déduplication par événement permet d'éviter de retourner plusieurs chunks correspondant au même événement parmi les recommandations finales.

---

## Intégration LangChain

LangChain est utilisé pour orchestrer la chaîne RAG.

Un retriever personnalisé basé sur `BaseRetriever` encapsule la recherche FAISS existante.

Les résultats sont transformés en objets `Document`, puis injectés dans une chaîne LCEL comprenant :

```text
Question utilisateur
        ↓
Retriever FAISS
        ↓
Documents LangChain
        ↓
Construction du contexte
        ↓
Prompt
        ↓
ChatMistralAI
        ↓
StrOutputParser
        ↓
Réponse
```

Cette architecture permet de conserver la logique métier développée dans les scripts FAISS tout en bénéficiant de l'orchestration proposée par LangChain.

---

## Structure du dépôt

```text
Projet11-RAG/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── evaluation/
│
├── docs/
│
├── notebooks/
│
├── scripts/
│   ├── fetch_openagenda.py
│   ├── preprocess_events.py
│   ├── chunk_events.py
│   ├── generate_embeddings.py
│   ├── build_faiss_index.py
│   ├── search_faiss.py
│   ├── rag_chain.py
│   ├── langchain_rag.py
│   ├── evaluate_rag.py
│   ├── annoter_evaluation.py
│   └── resume_evaluation.py
│
├── tests/
│
├── .env.example
├── .gitignore
├── main.py
├── pytest.ini
├── README.md
└── requirements.txt
```

---

## Installation

### 1. Cloner le dépôt

```powershell
git clone <URL_DU_DEPOT>
cd Projet11-RAG
```

### 2. Créer l'environnement virtuel

```powershell
python -m venv .venv
```

### 3. Activer l'environnement sous PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

Le terminal doit alors afficher :

```text
(.venv) PS C:\...\Projet11-RAG>
```

### 4. Installer les dépendances

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration des variables d'environnement

Le projet utilise l'API Mistral pour générer les embeddings et les réponses du chatbot.

Créer un fichier `.env` à partir du modèle fourni :

```powershell
Copy-Item .env.example .env
```

Puis renseigner les clés nécessaires dans le fichier `.env`.

Exemple :

```env
MISTRAL_API_KEY=votre_cle_mistral
```

Ne jamais versionner le fichier `.env` contenant les clés réelles.

---

## Reconstruction du pipeline

Les différentes étapes peuvent être exécutées successivement.

### 1. Collecter les événements OpenAgenda

```powershell
python scripts\fetch_openagenda.py
```

### 2. Prétraiter les événements

```powershell
python scripts\preprocess_events.py
```

### 3. Générer les chunks

```powershell
python scripts\chunk_events.py
```

### 4. Générer les embeddings

```powershell
python scripts\generate_embeddings.py
```

Cette étape nécessite une connexion Internet et une clé Mistral valide.

### 5. Construire l'index FAISS

```powershell
python scripts\build_faiss_index.py
```

Après ces étapes, la base vectorielle est prête pour les recherches.

---

## Tester la recherche FAISS

Une recherche sémantique peut être lancée directement depuis PowerShell :

```powershell
python scripts\search_faiss.py "exposition d'art contemporain à Marseille"
```

Le script génère l'embedding de la requête, interroge FAISS puis affiche les événements les plus similaires.

---

## Lancer la chaîne RAG

La version RAG sans orchestration LangChain peut être testée avec :

```powershell
python scripts\rag_chain.py "Je cherche une exposition d'art contemporain à Marseille"
```

Le système :

1. recherche les événements pertinents ;
2. construit le contexte ;
3. transmet le contexte et la question à Mistral ;
4. affiche la réponse générée.

---

## Lancer la version LangChain

La chaîne RAG intégrée à LangChain peut être exécutée avec :

```powershell
python scripts\langchain_rag.py "Je cherche une exposition d'art contemporain à Marseille"
```

Cette version utilise notamment :

- `BaseRetriever` ;
- `Document` ;
- LCEL ;
- `ChatMistralAI` ;
- `StrOutputParser`.

---

## Tests automatisés

Les tests sont exécutés avec `pytest` :

```powershell
pytest
```

Lors de la dernière validation du POC :

```text
31 tests réussis
31 / 31
```

Une partie des tests porte spécifiquement sur les données présentes dans la base vectorielle.

Ils vérifient notamment :

- que la base n'est pas vide ;
- que les événements concernent Marseille ;
- que les événements respectent la fenêtre temporelle du prétraitement ;
- que les champs minimaux nécessaires sont présents.

La fenêtre temporelle utilisée par ces tests est chargée depuis les métadonnées produites lors du prétraitement afin d'éviter qu'un index valide ne devienne artificiellement invalide avec le passage du temps.

---

## Évaluation fonctionnelle du RAG

En complément des tests techniques, le chatbot a été évalué sur **10 scénarios fonctionnels annotés**.

Les scénarios couvrent notamment :

- le tarif ;
- la thématique ;
- le lieu ;
- le public ;
- les contraintes temporelles ;
- le type de lieu ;
- les demandes hors périmètre ;
- les questions ouvertes.

Une date de référence fixe est utilisée pour rendre les scénarios temporels reproductibles.

### Résultat

```text
Score moyen d'évaluation manuelle : 91,7 %
Nombre de scénarios : 10
```

Ce résultat correspond à une **évaluation manuelle sur le jeu de scénarios du POC**. Il ne doit pas être interprété comme une mesure statistique générale de précision du modèle.

Les principaux cas d'échec ou de réponse partielle concernent :

- certaines expressions temporelles relatives, comme « ce week-end » ;
- certaines demandes situées hors du périmètre culturel du corpus.

---

## Limites actuelles

Le POC repose principalement sur une recherche par similarité sémantique.

FAISS permet d'identifier des événements proches du sens de la question, mais ne garantit pas à lui seul le respect exact de contraintes structurées comme :

```text
date précise
gratuité
public enfant
lieu
type d'événement
plein air
```

L'interprétation des expressions temporelles relatives par le LLM peut également produire des erreurs.

Enfin, le corpus dépend de la disponibilité et de la qualité des événements publiés dans les agendas OpenAgenda sélectionnés.

---

## Perspectives d'amélioration

Une évolution importante consisterait à mettre en place une **recherche hybride** :

```text
Question utilisateur
        ↓
Analyse des contraintes
        ↓
Filtres sur les métadonnées
        +
Recherche sémantique FAISS
        ↓
Événements candidats
        ↓
Génération de la réponse
```

Les principales améliorations envisagées sont :

- filtrage déterministe des dates ;
- résolution en Python des expressions comme « demain » ou « ce week-end » ;
- filtrage par lieu, public, tarif ou type d'événement ;
- détection des demandes hors périmètre ;
- automatisation du rafraîchissement du corpus ;
- augmentation du nombre de sources OpenAgenda ;
- enrichissement du jeu d'évaluation.

---

## Résultats principaux

Le POC final permet de construire une chaîne RAG complète et reproductible à partir de données culturelles réelles.

| Indicateur | Résultat |
|---|---:|
| Événements collectés | 2 596 |
| Événements finaux | 216 |
| Chunks | 403 |
| Dimension des embeddings | 1024 |
| Vecteurs FAISS | 403 |
| Tests automatisés | 31 / 31 |
| Scénarios d'évaluation | 10 |
| Score moyen d'évaluation manuelle | 91,7 % |

---

## Technologies utilisées

- Python 3
- OpenAgenda API
- Pandas
- NumPy
- Mistral AI
- FAISS
- LangChain
- Pytest
- python-dotenv
- Git / GitHub

---

## Auteur

**Nicolas Bamania**

Projet réalisé dans le cadre de la formation **Expert en ingénierie des données — OpenClassrooms**.