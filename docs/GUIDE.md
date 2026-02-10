# Guide Pédagogique - Medical Question Answering

Ce guide explique les concepts clés du projet et comment interpréter les résultats.

## 1. Stratégies d'Approche

### 1.1 Closed-Book Generation

**Principe**: Le modèle répond uniquement à partir de ses connaissances acquises pendant l'entraînement.

```
Question → LLM → Réponse
```

**Avantages**:
- Simple à implémenter
- Pas de dépendance externe
- Réponse rapide

**Inconvénients**:
- Peut halluciner (inventer des faits)
- Connaissances figées (date de training)
- Pas de source vérifiable

### 1.2 Prompt Engineering

**Principe**: Modifier la formulation de la demande pour améliorer les réponses.

**Stratégies testées**:

1. **One-sentence strict**: Force une réponse concise
   ```
   Answer in exactly one sentence.
   ```
   
2. **Flashcard style**: Imite le format du dataset
   ```
   Complete this medical flashcard...
   ```
   
3. **Uncertainty allowed**: Permet d'exprimer le doute
   ```
   If uncertain, say "Insufficient information."
   ```

**Impact attendu**:
- Format: Meilleur respect du format single-sentence
- Confiance: Moins d'hallucinations avec uncertainty allowed
- Trade-off: Taux de réponses vs précision

### 1.3 Fine-tuning avec LoRA/QLoRA

**Principe**: Adapter un modèle pré-entraîné au domaine médical avec peu de paramètres.

```
Modèle de base + LoRA adapters → Modèle adapté
```

**LoRA (Low-Rank Adaptation)**:
- Ajoute des matrices de rang faible aux couches attention
- ~1-2% des paramètres totaux
- Entraînable sur GPU consommateur

**QLoRA (Quantized LoRA)**:
- Modèle de base en 4-bit
- Réduit la mémoire GPU de 4x
- Qualité préservée pour l'inférence

**Hyperparamètres clés**:
- `lora_r`: Rang des matrices (16 recommandé)
- `lora_alpha`: Scaling factor (32 recommandé)
- `train_size`: Nombre d'exemples d'entraînement

### 1.4 RAG (Retrieval-Augmented Generation)

**Principe**: Récupérer des informations pertinentes avant de générer.

```
Question → Retriever → [Documents] → LLM + Context → Réponse
```

**Avantages**:
- Réduit les hallucinations
- Connaissances actualisables
- Sources traçables

**Inconvénients**:
- Latence supplémentaire
- Qualité dépendante de la retrieval
- Coût en tokens (prompt plus long)

**Retrievers implémentés**:
1. **Wikipedia**: Stable, fiable, gratuit
2. **Web (DuckDuckGo)**: Plus large mais bruité

## 2. Métriques d'Évaluation

### 2.1 ROUGE-L

**Définition**: Plus longue sous-séquence commune (LCS) entre prédiction et référence.

**Formule**:
```
Precision = LCS / len(prediction)
Recall = LCS / len(reference)
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Interprétation**:
- 0.0 = Aucun mot en commun dans l'ordre
- 1.0 = Correspondance parfaite
- Bon pour: Capture de la structure générale

**Limites pour QA médical**:
- Pénalise les paraphrases valides
- "The heart pumps blood" vs "Blood is pumped by the heart" = score faible

### 2.2 BLEU

**Définition**: Précision des n-grammes avec pénalité de brièveté.

**Interprétation**:
- Mesure la précision des n-grammes (1-4)
- Score normalisé 0-1
- Plus strict que ROUGE pour l'ordre exact

**Limites**:
- Très strict pour QA courtes
- Une réponse correcte avec synonymes = score faible

### 2.3 LLM-as-a-Judge

**Principe**: Un LLM évalue l'équivalence sémantique.

**Scores**:
- **2 (Correct)**: Même information médicale
- **1 (Partial)**: Information partiellement correcte
- **0 (Wrong)**: Incorrect ou non-pertinent

**Avantages**:
- Capture la sémantique
- Tolère les paraphrases
- Proche du jugement humain

**Limites**:
- Coût API
- Biais du modèle juge
- Instabilité (vérifier avec double-judge)

### 2.4 Pourquoi combiner les métriques?

| Situation | ROUGE | BLEU | Judge |
|-----------|-------|------|-------|
| Paraphrase correcte | Bas | Bas | 2 |
| Copie exacte | Haut | Haut | 2 |
| Hallucination plausible | Moyen | Moyen | 0 |
| Réponse vide | 0 | 0 | 0 |

**Recommandation**: Ne jamais utiliser une seule métrique. La combinaison donne une vue plus complète.

## 3. Interprétation des Résultats

### 3.1 Lire les tableaux de résultats

```
| Experiment | ROUGE-L | BLEU | Judge | Correct% |
|------------|---------|------|-------|----------|
| baseline   | 0.42    | 0.28 | 1.45  | 52%      |
| rag_wiki   | 0.48    | 0.31 | 1.62  | 68%      |
```

**Comparaisons à faire**:
1. RAG améliore-t-il le Judge score?
2. Le ROUGE suit-il la même tendance?
3. Y a-t-il des cas où ROUGE et Judge divergent?

### 3.2 Analyser les exemples qualitatifs

**Best cases**: Où le modèle excelle
- Quels types de questions?
- Quelle longueur de réponse?

**Worst cases**: Où le modèle échoue
- Questions ambiguës?
- Termes rares?
- Multi-concepts?

**Controversial**: Désaccord métrique/juge
- ROUGE haut + Judge bas = hallucination fluide
- ROUGE bas + Judge haut = bonne paraphrase

### 3.3 Ablations à interpréter

**Taille des données (fine-tuning)**:
```
1k → 5k → 20k exemples
```
- Courbe de progression?
- Plateau atteint?
- Overfitting possible?

**Top-k (RAG)**:
```
top_k=3 vs top_k=5
```
- Plus de contexte = meilleure qualité?
- Ou plus de bruit?

## 4. Écrire la Discussion

### 4.1 Structure recommandée

1. **Résumé des résultats principaux**
   - Quelle méthode fonctionne le mieux?
   - Écarts significatifs ou non?

2. **Analyse par stratégie**
   - Closed-book: limites observées?
   - Prompting: impact réel?
   - Fine-tuning: scaling avec les données?
   - RAG: apport du retrieval?

3. **Limites méthodologiques**
   - Taille du test set
   - Biais du judge
   - Dataset spécifique

4. **Implications pratiques**
   - Coût vs performance
   - Cas d'usage recommandés

### 4.2 Honnêteté requise

**À inclure**:
- Résultats négatifs (si une méthode ne marche pas)
- Variances observées
- Limitations du dataset
- Risques d'hallucinations

**À éviter**:
- Inventer des chiffres
- Cacher des échecs
- Sur-interpréter de petites différences
- Généraliser au-delà des données

## 5. Considérations Éthiques

### 5.1 Hallucinations en médecine

Les LLMs peuvent générer du contenu **plausible mais faux**. En médecine:
- Informations incorrectes = danger potentiel
- Toujours avertir les utilisateurs
- Ne jamais présenter comme conseil médical

### 5.2 Biais des données

Le dataset peut contenir:
- Biais géographiques (médecine occidentale)
- Informations datées
- Simplifications excessives

### 5.3 Transparence

Documenter:
- Sources des données
- Modèles utilisés
- Coûts réels
- Limitations connues

## 6. Checklist Finale

### Avant l'exécution
- [ ] API keys configurées
- [ ] Splits créés (seed=42)
- [ ] Espace disque suffisant

### Après l'exécution
- [ ] Tous les fichiers de prédiction générés
- [ ] Métriques calculées et sauvegardées
- [ ] Figures lisibles et informatives

### Pour le rapport
- [ ] Tables auto-générées incluses
- [ ] Figures avec légendes claires
- [ ] Discussion des limitations
- [ ] Avertissement médical présent
- [ ] Aucun résultat inventé

## 7. FAQ

**Q: Mes scores ROUGE sont très bas, c'est normal?**
A: Pour du QA médical, ROUGE-L autour de 0.3-0.5 est typique. Les réponses peuvent être correctes avec des formulations différentes.

**Q: Le judge donne toujours 2, c'est suspect?**
A: Vérifiez le prompt du judge. Un prompt trop permissif peut tout accepter. Ajoutez des exemples de score 0 et 1.

**Q: Comment savoir si c'est une hallucination?**
A: Comparez avec la référence ET vérifiez factuellement. Un score Judge=2 avec ROUGE bas peut indiquer une bonne paraphrase ou une hallucination plausible.

**Q: Quelle méthode choisir en pratique?**
A: Dépend du contexte:
- Coût minimal → Local (Ollama)
- Meilleure qualité → RAG + GPT-4
- Données disponibles → Fine-tuning

**Q: Comment gérer les erreurs API?**
A: Le code inclut retry automatique avec backoff. Vérifiez les logs dans `logs/`.
