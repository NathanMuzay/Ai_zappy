# README_CURRICULUM_LEARNING.md

## Problème identifié

Avec le système de récompense initial, les agents ne pouvaient **jamais accumuler assez de nourriture** pour atteindre 200+ food (seuil Fork).
Résultat : **zéro Fork, pool épuisé toutes les ~17 minutes**.

## Solution : Curriculum Learning en 2 phases

### Phase 1 (0 à 500k steps) — SURVIE D'ABORD

**Objectif** : Apprendre à trouver et manger de la nourriture régulièrement.

**Reward modifié** :
- `Ramasser food < 30` : **+10.0** (WAS 2.0) → **+400%**
- `Ramasser food 30-126` : **+5.0** (WAS 0.5) → **+900%**
- `Food critique (< 15)` : **-0.2** (WAS -0.05) → **-4x urgence**
- `Mort` : **-20.0** (WAS -10.0) → **pénalité totale pour famine**
- `Fork` : **+0.5** (ignoré, pas l'heure)

**Résultat attendu** :
- Agents survivent 5-10 min (au lieu de 2-3 min)
- Accumulent 50-100 food progressivement
- Meurent moins → pool se vide plus lentement

### Phase 2 (500k à 2M steps) — REPRODUCTION

**Objectif** : Reproduction = Fork régulier pour régénérer le pool.

**Reward modifié** :
- `Ramasser food` : **+2.0** (réduit, déjà su faire)
- `Fork réussi` : **+8.0** (WAS 0.5) → **16x plus incité**
- `Mort` : **-10.0** (réduit, moins critique)
- `Incantation` : Bonus inchangé (progression naturelle)

**Résultat attendu** :
- Agents tentent Fork quand ils ont assez de food
- Chaque Fork → +1 agent + régénère 1 oeuf du pool
- Pool se maintient → entraînement stable

## Implémentation technique

### 1. `src/rewards.py`
- Fonction `set_global_timesteps(ts)` : appelée par CurriculumCallback
- Fonction `get_phase()` : retourne phase 1 ou 2
- `compute_reward()` : deux branches d'exécution selon phase

### 2. `src/train.py`
- Classe `CurriculumCallback` : log la transition 1→2 à 500k steps
- Classe `ZappyCallback` : log métrique toutes les 60k steps

### 3. `src/env.py` & `supervisor.py`
- Inchangés (déjà fonctionnels)

## Résultat observé

Avant (sans curriculum) :
