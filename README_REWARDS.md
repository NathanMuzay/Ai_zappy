# Système de Récompenses — Documentation Technique

## Table des matières
1. [Vue d'ensemble](#1-vue-densemble)
2. [Signature de `compute_reward`](#2-signature-de-compute_reward)
3. [Etat interne au module](#3-etat-interne-au-module)
4. [Phase DÉBUT (level 1-3)](#4-phase-début-level-1-3)
5. [Phase FIN (level 4-7)](#5-phase-fin-level-4-7)
6. [Anti-FORK-SPAM](#6-anti-fork-spam)
7. [Comparaison avant / après](#7-comparaison-avant--après)
8. [Aucune modification de env.py](#8-aucune-modification-de-envpy)

---

## 1. Vue d'ensemble

Le nouveau système de récompenses organise le comportement de l'agent en **3 phases** :

| Phase | Niveau | Objectif | Mécanique clé |
|-------|--------|----------|---------------|
| DÉBUT | 1-3 | Survie + collecte pierres élévation | Reward fort nourriture critique, collecte pierres palier |
| FIN | 4-7 | Élévation + assemblage pierres | Reward colossal montee, gather stones manquantes |
| ANTI-FORK | TOUJOURS | Empêcher le spam Fork | Penalité linéaire croissante sur Fork répétés |

---

## 2. Signature de `compute_reward`

**Signature inchangée** — compatible avec l'appel existant dans `env.py` :

```python
def compute_reward(prev: dict, state: dict, event: dict) -> float:
```

| Argument | Contenu | Provenance dans env.py |
|----------|---------|----------------------|
| `prev` | État **AVANT** action : level, inventory, alive, food_dist | construit dans `step()` ligne `prev = {...}` |
| `state` | État **APRÈS** action : level, inventory, alive, food_dist | `self.state` après traitement de la réponse |
| `event` | Résultat action : action, ok, ko, level_up, old_level, death, win | construit dans `step()` selon commande |

Aucune modification de `env.py` n'est requise.

---

## 3. État interne au module

Le module utilise **2 variables de module** (survit entre les appels) :

```python
_fork_count   : int   — nombre de Fork consécutifs réussis (reset à 0 sur autre action)
_last_level   : int   — dernier niveau observé (détecte les montées)
```

Ces variables permettent :
- De **détecter le spam Fork** sans avoir besoin de `team_state` (non passé à `compute_reward`)
- De **réinitialiser le compteur** après une montee de niveau ou une action non-Fork

---

## 4. Phase DÉBUT (level 1-3)

### 4a. Nourriture (survie)

| Situation | Reward |
|-----------|--------|
| food < 30 et alimentation récupérée | **+2.0** |
| food < 80 et alimentation récupérée | **+0.5** |
| food >= 80 et alimentation récupérée | **+0.05** |
| food < 30 (pression constante) | **-0.1** par step |
| Se rapproche de la nourriture visible | **+0.3** |
| S'éloigne de la nourriture visible | **-0.1** |
| Sur case food sans manger | **-0.2** (gaspillage) |
| Take food réussi (cas prioritaire) | **+2.0 / +0.5 / +0.05** selon niveau |

**Pourquoi** : En début de partie, mourir de faim est le risque principal. L'agent doit prioriser la nourriture avant tout. Le bonus proportionnel au déficit crée une pression forte mais graduée.

### 4b. Pierres (collecte)

| Action | Condition | Reward |
|--------|-----------|--------|
| Take pierre manquante (palier courant) | `have <= needed` | **+2.0** |
| Take pierre future (palier futur) | `needed > 0` | **+0.3** |
| Take pierre inutile | hors élévation | **+0.05** |
| Set pierre manquante sur case | `on_tile <= needed` | **+2.5** |
| Set pierre pas prioritaire | `needed > 0` | **-0.2** |
| Set pierre inutile | hors élévation | **-0.1** |

**Pourquoi** : Linemate pour niveau 2, linemate + deraumere + sibur pour niveau 3. Chaque prise de pierre manquante est récompensée 4x plus qu'une pierre inutile.

### 4c. Incantation

| Résultat | Reward |
|----------|--------|
| Élévation réussie | **+50.0** |
| Échec (ko) | **-3.0** |
| Cout de l'incantation | **-0.43** |

**Pourquoi** : En phase début, il faut 1 à 2 joueurs. L'agent peut tenter l'élévation assez tôt si les pierres sont collectées. Le reward de +50 rend la montee très attractive sans être excessive.

### 4d. Mort

| Situation | Reward |
|-----------|--------|
| Mort (death) | **-10.0** |

---

## 5. Phase FIN (level 4-7)

### 5a. Élévation (action critique)

| Résultat | Reward |
|----------|--------|
| Victoire finale (level 8) | **+500.0** |
| Niveau max atteint (non-victoire) | **+200.0** |
| Élévation réussie niveau 4-7 | **+80 + 20 × delta** |
| Échec (ko) | **-5.0** (plus cher qu'en phase début) |
| Cout incantation ratee | **-0.43** |

**Pourquoi** : Plus le niveau est élevé, plus le défi est grand (plus de joueurs requis, plus de pierres). Le reward croissant reflète la difficulté croissante. La pénalité d'échec plus forte (-5 au lieu de -3) décourage les tentatives prématurées.

### 5b. Gather pierres (assemblage pour élévation)

| Action | Condition | Reward |
|--------|-----------|--------|
| Take pierre manquante | `have <= needed` | **+3.0 − 1.5 × shortage** |
| Take pierre future | `needed > 0` | **+0.5** |
| Take pierre inutile | hors élévation | **+0.05** |
| Set pierre manquante sur case | `on_tile <= needed` | **+3.5** |
| Set pierre pas prioritaire | `needed > 0` | **-0.3** |
| Set pierre inutile | hors élévation | **-0.2** |

**Pourquoi du `shortage`** : La formule `+3.0 - 1.5 × shortage` donne un reward de **+3.0** quand il manque tout (`shortage=1`) et **+1.5** quand il ne manque qu'un peu (`shortage=0`). Cela incite l'agent à gather les pierres dont il a **le plus besoin**, pas seulement n'importe laquelle.

### 5c. Nourriture

Même logique qu'en phase début mais avec pression légèrement supérieure (`-0.15` au lieu de `-0.1`) car mourir en phase fin est plus coûteux (progression longue perdue).

### 5d. Reward de Gather (shaping transversal)

```python
r += (1.0 - shortage) × 0.05   # phase début
r += (1.0 - shortage) × 0.15   # phase fin
```

**Pourquoi** : Ce shaping récompense passivement l'agent quand il **détient** les bonnes pierres du palier courant, même sans avoir fait d'action liée. Cela guide l'agent vers l'objectif (gather les pierres) même quand il explore ou se déplace.

---

## 6. Anti-FORK-SPAM

### Principe

Fork ne doit **jamais être l'action la plus rentable par défaut**. Le compteur `_fork_count` suit les Fork réussis consécutifs.

| Fork # | Reward | Explication |
|--------|--------|-------------|
| 1 | 0.0 | Neutre — peut être utile pour coop early game |
| 2 | -0.5 | Premier Fork supplémentaire — légère pénalité |
| 3 | -1.0 | Deuxième Fork supplémentaire |
| 4+ | -1.5, -2.0... | Pénalité linéaire croissante |

### Reset du compteur

Le compteur est remis à **0** quand :
- Une action **autre que Fork** est effectuée
- L'agent **monte de niveau** (récompense le focus sur l'élévation)

### Pourquoi un Fork gratuit ?

Le premier Fork est neutre car :
- En début de partie (level 1-3), un deuxième joueur peut aider à l'élévation niv1→2
- Cela permet la coordination sans punir la reproduction légitime
- Mais le 2e Fork est déjà pénalisé, donc l'agent ne va pas spammer

---

## 7. Comparaison avant / après

| Aspect | Avant | Après |
|--------|-------|-------|
| Nourriture critique | reward statique (+2.0 si < 30) | reward proportionnel au besoin (+2.0 à +0.05) |
| Collecte pierres | +1.5 / +0.1 / +0.05 | +2.0 / +0.3 / +0.05 (phase début), +3.0 / +0.5 / +0.05 (phase fin) |
| Dépose pierres | +2.0 / -0.2 / -0.1 | +2.5 / -0.2 / -0.1 (début), +3.5 / -0.3 / -0.2 (fin) |
| Élévation réussie | +10 + 5 × difficulty × delta | +50 (début), +80+ (fin), +500 (victoire) |
| Échec élévation | -2.0 | -3.0 (début), -5.0 (fin) |
| Fork spam | aucune pénalité | -0.5 × (fork_count - 1) après 1er Fork |
| Phase spécifique | NON | OUI — récompenses différentes selon level |
| Nourriture visible (shaping) | OUI (+0.3/-0.1) | OUI — conservé dans les 2 phases |

---

## 8. Aucune modification de env.py

Le fichier `env.py` **n'a pas été modifié** pour les raisons suivantes :

1. La signature de `compute_reward(prev, state, event)` est **identique** à l'appel existant dans `env.py` (ligne `reward = compute_reward(prev, self.state, event)`)
2. Toutes les variables passées (`prev`, `self.state`, `event`) sont **déjà construites** et alimentées par `env.py`
3. `food_dist` est déjà populated dans `prev["food_dist"]` et `state["food_dist"]` avant l'appel
4. `team_state` n'est pas nécessaire car `compute_reward` utilise un **état interne au module** (`_fork_count`, `_last_level`) pour le tracking anti-Fork

Pour déployer : remplacer `src/rewards.py` par le nouveau fichier, sans toucher à `src/env.py`.
