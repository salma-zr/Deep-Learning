# Analyse Documentaire Complète du Projet

## Vue d'ensemble

Ce dépôt est un **projet de Deep Learning pour un cours de Master 2 "Probabilités et Finance"** (vraisemblablement à Sorbonne Université / UPMC, d'après les liens vers `schwander.isir.upmc.fr`). Il porte sur le **Question Answering (QA) médical**, c'est-à-dire la capacité de systèmes d'intelligence artificielle à répondre automatiquement à des questions médicales.

Le projet est structuré en **deux parties distinctes** :

1. **Le projet principal (`src/`, `configs/`, `report/`, `scripts/`)** : Un pipeline complet et reproductible de QA médical comparant 4 grandes approches (closed-book, prompt engineering, fine-tuning, RAG).
2. **Les TPs du cours (`tp-*.ipynb`)** : Trois notebooks Jupyter correspondant aux travaux pratiques du cours de Deep Learning.

---

## PARTIE 1 : Le Projet Principal — Medical Question Answering

### 1.1 Objectif

Le but est de comparer différentes méthodes pour répondre à des questions médicales de type "flashcard" (question courte → réponse en une phrase). Le projet évalue rigoureusement chaque approche avec des métriques automatiques et un "juge LLM", puis génère un rapport LaTeX complet.

### 1.2 Le Dataset

- **Nom** : `medalpaca/medical_meadow_medical_flashcards` (hébergé sur HuggingFace)
- **Contenu** : ~33 955 paires question/réponse médicales au format flashcard
- **Domaine** : Connaissances médicales générales (anatomie, physiologie, pathologie, pharmacologie)
- **Format** : Chaque entrée contient un champ `input` (question), `output` (réponse) et `instruction`
- **Exemple** : Q: "What is the function of insulin?" → R: "Insulin regulates blood glucose levels."

Le dataset est découpé en :
| Split | Taille | Usage |
|-------|--------|-------|
| Train | ~33 600 (80%) | Entraînement (fine-tuning) |
| Dev | ~4 200 (10%) | Validation |
| Test | ~4 200 (10%) | Évaluation finale |
| Tiny Test | 200 | Tests rapides et débogage |

### 1.3 Les 4 Stratégies Comparées

#### A) Closed-Book Generation (Expériences 01, 02, 03, 11)

Le LLM répond uniquement à partir de ses connaissances internes (acquises pendant le pré-entraînement). Aucun document externe n'est fourni.

**Modèles testés** :
- **GPT-4o-mini** (OpenAI, API cloud) — baseline principal
- **Llama 3.1 8B** via Ollama (inférence locale, gratuit)
- **Mistral 7B** via OpenRouter (API gratuite)
- **Flan-T5-base** via HuggingFace (CPU, fallback)

**Avantages** : Simplicité, rapidité.  
**Inconvénients** : Risque d'hallucinations, connaissances figées.

#### B) Prompt Engineering (Expériences 04, 05)

On modifie la formulation de la question (le "prompt") pour améliorer la qualité des réponses. Le fichier `src/prompts/answerer_prompts.yaml` définit 8 styles de prompts différents :

| Style | Description |
|-------|------------|
| `one_sentence_strict` | Force une réponse en une seule phrase |
| `flashcard_style` | Imite le format flashcard du dataset |
| `uncertainty_allowed` | Autorise le modèle à dire "Insufficient information" |
| `detailed_medical` | Demande une terminologie médicale précise |
| `simple_direct` | Question brute, sans instruction |
| `zero_shot_cot` | Chain-of-thought (raisonnement étape par étape) |
| `few_shot_medical` | Exemples de démonstration inclus |
| `strict_format` | Contraintes de format très strictes |

**Objectif** : Mesurer l'impact de la formulation sur la qualité et le format des réponses.

#### C) Fine-tuning avec LoRA/QLoRA (Expériences 08, 09, 10)

On adapte un modèle pré-entraîné au domaine médical en entraînant seulement un petit nombre de paramètres supplémentaires.

**Technique** : QLoRA (Quantized Low-Rank Adaptation) :
- Le modèle de base est chargé en 4 bits (quantification NF4)
- Des matrices de rang faible sont ajoutées aux couches d'attention (rang r=16, alpha=32)
- Seulement ~1-2% des paramètres sont entraînés
- Mémoire GPU requise : ~8 Go VRAM

**Modèle cible** : TinyLlama-1.1B-Chat

**Ablation sur la taille des données** :
- 1 000 exemples (exp_08)
- 5 000 exemples (exp_09)
- 20 000 exemples (exp_10)

L'objectif est de mesurer l'effet de la quantité de données d'entraînement sur la performance.

#### D) RAG — Retrieval-Augmented Generation (Expériences 06, 07)

Au lieu de se fier uniquement à la mémoire du modèle, on récupère d'abord des documents pertinents depuis Wikipedia, puis on les injecte dans le prompt.

**Pipeline** :
```
Question → Retriever (Wikipedia) → [Documents pertinents] → Prompt + Contexte → LLM → Réponse
```

**Paramètres testés** :
- `top_k=3` (exp_06) vs `top_k=5` (exp_07) — nombre de documents récupérés
- `max_context_chars=2000` — longueur max du contexte injecté

**Retrievers disponibles** : Wikipedia API, DuckDuckGo (web search), mode hybride.

### 1.4 Système d'Évaluation

Le projet utilise **trois niveaux d'évaluation** complémentaires :

#### Métriques automatiques

| Métrique | Ce qu'elle mesure | Limites |
|----------|-------------------|---------|
| **ROUGE-L** | Plus longue sous-séquence commune (F1) | Pénalise les paraphrases valides |
| **BLEU** | Précision des n-grammes (1-4) | Très strict pour les réponses courtes |
| **Exact Match** | Correspondance exacte (après normalisation) | Trop strict pour du QA ouvert |

#### LLM-as-a-Judge

Un modèle GPT-4o-mini évalue la **qualité sémantique** de chaque réponse :
- **Score 2 (Correct)** : Sémantiquement équivalent à la référence
- **Score 1 (Partial)** : Information partiellement correcte
- **Score 0 (Wrong)** : Incorrect, irrelevant ou vide

Le prompt du juge est défini dans `src/prompts/judge_prompt.txt`. Un système de vérification de cohérence (`verify_judge_consistency`) re-juge un échantillon pour mesurer la fiabilité du juge.

#### Statistiques de format
- Longueur moyenne des réponses (caractères, mots)
- Pourcentage de réponses multi-phrases
- Taux de réponses vides
- Taux de réponses "Insufficient information"
- Statistiques de latence (moyenne, médiane, P95)

### 1.5 Architecture du Code Source

```
src/
├── data/           # Téléchargement et découpage du dataset
│   ├── download.py    → Télécharge depuis HuggingFace
│   ├── split.py       → Crée train/dev/test avec seed=42
│   └── cli.py         → Interface CLI (download, split, info, sample)
│
├── generation/     # Génération closed-book
│   ├── base.py        → Classes abstraites (BaseGenerator, GenerationResult)
│   ├── openai_gen.py  → Générateur OpenAI (GPT-4o-mini, GPT-4o)
│   ├── ollama_gen.py  → Générateur Ollama (Llama, Mistral local)
│   ├── openrouter_gen.py → Générateur OpenRouter (modèles gratuits)
│   ├── hf_gen.py      → Générateur HuggingFace (Flan-T5, CPU)
│   ├── prompts.py     → Gestion des templates de prompts
│   └── cli.py         → Interface CLI (run, test-generator)
│
├── rag/            # Retrieval-Augmented Generation
│   ├── retrievers.py  → Retrievers Wikipedia et Web (DuckDuckGo)
│   ├── rag_generator.py → Pipeline RAG complet (retrieve → format → generate)
│   └── cli.py         → Interface CLI (run, test-retrieval, test-rag)
│
├── finetune/       # Fine-tuning LoRA/QLoRA
│   ├── data_prep.py   → Préparation des données d'entraînement (format Alpaca)
│   ├── trainer.py     → FineTuner avec support LoRA/QLoRA + mode symbolique
│   └── cli.py         → Interface CLI (prepare, train, check-gpu)
│
├── eval/           # Évaluation
│   ├── metrics.py     → ROUGE, BLEU, exact match, statistiques
│   ├── judge.py       → LLM-as-a-Judge (évaluation sémantique)
│   ├── qualitative.py → Génération de rapports qualitatifs (HTML)
│   └── cli.py         → Interface CLI (metrics, judge, full, qualitative)
│
├── report/         # Génération de rapport
│   ├── tables.py      → Tables LaTeX auto-générées
│   ├── figures.py     → Graphiques matplotlib/seaborn
│   ├── latex.py       → Gestion LaTeX
│   └── cli.py         → Interface CLI (tables, figures, build, compile)
│
├── prompts/        # Templates de prompts
│   ├── answerer_prompts.yaml → 8 styles de prompts pour le QA
│   ├── judge_prompt.txt      → Prompt pour l'évaluation LLM
│   └── rag_prompt.txt        → Prompt pour le RAG
│
└── utils/          # Utilitaires
    ├── cache.py       → Cache disque pour les appels API (évite les doublons)
    ├── io_utils.py    → Lecture/écriture JSONL, YAML, CSV
    ├── logger.py      → Logging centralisé
    ├── normalize.py   → Normalisation de texte + statistiques
    ├── retry.py       → Retry automatique avec backoff exponentiel
    └── timer.py       → Mesure de temps d'exécution
```

### 1.6 Configurations d'Expériences (11 au total)

Les fichiers YAML dans `configs/` définissent chaque expérience de manière déclarative :

| # | Fichier | Stratégie | Modèle | Objectif |
|---|---------|-----------|--------|----------|
| 01 | `exp_01_openai_baseline.yaml` | Closed-book | GPT-4o-mini | Baseline API cloud |
| 02 | `exp_02_ollama_llama.yaml` | Closed-book | Llama 3.1 8B | Comparaison locale |
| 03 | `exp_03_openrouter_free.yaml` | Closed-book | Mistral 7B (free) | Option gratuite |
| 04 | `exp_04_prompt_flashcard.yaml` | Prompt Eng. | GPT-4o-mini | Prompt style flashcard |
| 05 | `exp_05_prompt_uncertainty.yaml` | Prompt Eng. | GPT-4o-mini | Réduction des hallucinations |
| 06 | `exp_06_rag_wikipedia.yaml` | RAG | GPT-4o-mini + Wikipedia | Grounding par retrieval |
| 07 | `exp_07_rag_topk_ablation.yaml` | RAG | GPT-4o-mini + Wikipedia | Ablation top_k=5 |
| 08 | `exp_08_finetune_1k.yaml` | Fine-tuning | TinyLlama 1.1B | Baseline 1k exemples |
| 09 | `exp_09_finetune_5k.yaml` | Fine-tuning | TinyLlama 1.1B | Données moyennes |
| 10 | `exp_10_finetune_20k.yaml` | Fine-tuning | TinyLlama 1.1B | Grande quantité de données |
| 11 | `exp_11_hf_flan_t5.yaml` | Closed-book | Flan-T5-base | Fallback CPU |

### 1.7 Scripts d'Automatisation

- **`scripts/run_all.sh`** (Linux/Mac) et **`scripts/run_all.ps1`** (Windows) : Exécutent le pipeline complet en 5 étapes :
  1. Préparation des données (download + split)
  2. Expériences de génération (closed-book, prompting, RAG)
  3. Fine-tuning (optionnel, nécessite GPU)
  4. Évaluation (métriques + juge LLM)
  5. Génération du rapport (tables, figures, PDF LaTeX)

Options : `--quick` (test rapide avec 50 exemples), `--skip-api` (ignore les expériences nécessitant des clés API), `--with-finetune` (active le fine-tuning).

### 1.8 Le Rapport LaTeX (`report/report.tex`)

Un document LaTeX complet (~500 lignes) structuré en sections académiques :
- Introduction, Dataset, Méthodes, Méthodologie d'évaluation
- Setup expérimental, Résultats (avec tables et figures auto-générées)
- Analyse qualitative, Discussion, Conclusion
- Annexes (templates de prompts, configurations)
- Bibliographie (6 références : MedAlpaca, LoRA, QLoRA, RAG, ROUGE, BLEU)

Les tables et figures sont générées automatiquement par le code et insérées dans le rapport via `\IfFileExists`.

### 1.9 Principes de Conception

- **Reproductibilité** : Seed 42 partout, cache des appels API, prompts versionnés, température 0.0
- **Conscience des coûts** : Estimation des coûts API, alternatives gratuites (Ollama, OpenRouter free tier), limites configurables
- **Honnêteté** : Avertissement médical omniprésent, discussion des limitations, pas de résultats inventés
- **Modularité** : Chaque composant est un module CLI indépendant (`python -m src.<module>.cli`)

---

## PARTIE 2 : Les Notebooks de TP

### 2.1 `tp-llm.ipynb` — Fine-tuning de LLM

**Sujet** : Fine-tuning d'un petit LLM (SmolLM-135M-Instruct, 135M paramètres) sur le dataset MedAlpaca.

**Contenu** :
- Chargement du dataset `medalpaca/medical_meadow_medical_flashcards`
- Split train/test (80/20)
- Sous-échantillonnage (80 train, 20 eval) pour fonctionner sur CPU
- Chargement du modèle `HuggingFaceTB/SmolLM-135M-Instruct`
- Configuration LoRA via la bibliothèque `peft`
- Entraînement avec `SFTTrainer` (de la bibliothèque `trl`)
- Évaluation avec `evaluate` (métriques ROUGE)
- Suivi avec MLflow

**Motivation pédagogique** : Montrer qu'un petit modèle (135M) peut être spécialisé sur un domaine médical grâce au fine-tuning, approchant potentiellement les performances d'un gros LLM (>1B) pour des tâches spécifiques.

**Bibliothèques utilisées** : `transformers`, `peft`, `trl`, `bitsandbytes`, `evaluate`, `mlflow`, `datasets`

### 2.2 `tp-image.ipynb` — Fine-tuning pour l'Image

**Sujet** : Classification d'images avec des CNN pré-entraînés (transfer learning).

**Contenu** :
- Chargement du dataset **Fashion-MNIST** (60 000 images train, 10 000 test)
- Définition d'un **petit CNN** (SmallCNN) avec PyTorch Lightning :
  - 2 couches convolutionnelles (32 et 64 filtres)
  - Max pooling, dropout, 2 couches fully connected
  - ~421K paramètres entraînables
- Entraînement de bout en bout (10 époques)
- **Fine-tuning de modèles pré-entraînés** :
  - **VGG-16** : CNN relativement simple, chargé avec poids ImageNet
  - **ResNet-50** : CNN plus complexe (skip connections), état de l'art pré-ViT
- Technique : Geler tous les paramètres du réseau pré-entraîné, remplacer la tête de classification

**Bibliothèques utilisées** : `torch`, `pytorch-lightning`, `torchvision`, `datasets`

### 2.3 `tp-embeddings.ipynb` — Embeddings pour le Texte et l'Image

**Sujet** : Utilisation d'embeddings multimodaux (texte + image) avec OpenCLIP.

**Contenu** :
- Chargement du modèle **CLIP** (ViT-B-32) pré-entraîné sur LAION-2B
- Chargement du dataset **CIFAR-10** (1000 exemples)
- Extraction d'embeddings visuels et textuels
- Classification zero-shot (classer des images en utilisant des descriptions textuelles)
- Matrice de confusion
- Visualisation des résultats

**Concept clé** : CLIP permet de projeter images et texte dans un même espace vectoriel, ce qui permet de faire de la classification sans entraînement spécifique (zero-shot).

**Bibliothèques utilisées** : `open_clip`, `torch`, `datasets`, `sklearn`, `matplotlib`

---

## PARTIE 3 : Synthèse Technique

### Technologies et Frameworks

| Catégorie | Technologies |
|-----------|-------------|
| Langage | Python 3.11+ |
| Deep Learning | PyTorch, PyTorch Lightning, Transformers (HuggingFace) |
| Fine-tuning | PEFT (LoRA/QLoRA), BitsAndBytes (quantification), TRL (SFT) |
| LLM APIs | OpenAI, Ollama, OpenRouter |
| RAG | Wikipedia-API, DuckDuckGo-Search, Sentence-Transformers, FAISS |
| Évaluation | rouge-score, sacrebleu, NLTK |
| Vision | torchvision (VGG, ResNet), OpenCLIP (ViT) |
| Suivi | MLflow |
| Visualisation | Matplotlib, Seaborn |
| Rapport | LaTeX, Jinja2 |
| CLI | Typer, Rich |
| Gestion de dépendances | pyproject.toml (hatchling), uv |

### Dépendances (pyproject.toml)

Le projet est packagé proprement avec `hatchling` et définit :
- 30+ dépendances principales couvrant l'intégralité du pipeline
- Des groupes optionnels : `dev` (tests, linting), `gpu` (PyTorch GPU), `cpu` (PyTorch CPU)
- Des points d'entrée CLI : `medqa-data`, `medqa-generate`, `medqa-rag`, `medqa-finetune`, `medqa-eval`, `medqa-report`

### Coûts estimés

| Backend | Modèle | Coût / 500 exemples |
|---------|--------|---------------------|
| OpenAI | gpt-4o-mini | ~0.50 $ |
| OpenAI | gpt-4o | ~5.00 $ |
| OpenRouter | mistral:free | 0.00 $ |
| Ollama | llama3.1:8b | 0.00 $ (local) |
| HuggingFace | flan-t5-base | 0.00 $ (local) |
| Fine-tuning | Colab T4 | 0.00 $ (free tier) |

---

## Conclusion

Ce dépôt représente un **projet pédagogique complet et bien structuré** pour un cours de Deep Learning de niveau Master 2. Il couvre un spectre large de techniques modernes du NLP :

1. **L'utilisation d'APIs de LLM** (OpenAI, OpenRouter) pour du question answering
2. **L'inférence locale** (Ollama, HuggingFace) comme alternative gratuite
3. **Le prompt engineering** avec des stratégies variées
4. **Le fine-tuning efficient** avec LoRA/QLoRA sur GPU consommateur
5. **Le RAG** pour ancrer les réponses dans des connaissances vérifiables
6. **L'évaluation rigoureuse** combinant métriques automatiques et jugement LLM
7. **Le transfer learning** en vision (VGG, ResNet sur Fashion-MNIST)
8. **Les embeddings multimodaux** (CLIP pour la classification zero-shot)

Le tout est enveloppé dans un pipeline reproductible, bien documenté, avec un rapport LaTeX auto-généré et une attention particulière aux considérations éthiques (avertissements médicaux, transparence sur les limitations).
