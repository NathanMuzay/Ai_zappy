# Zappy PPO — Notes d'entraînement

## 1. Contexte

Ce dépôt entraîne deux agents PPO (équipes `Br` et `of`) à jouer au jeu **Zappy**
(sujet Epitech `G-YEP-400`) via un environnement Gymnasium (`zappy_env.py`) qui
communique en TCP avec `zappy_server`.

- `train.py` : entraîne un agent à partir d'une config YAML (`configs/*.yaml`).
- `train_dual.py` : lance le serveur + les deux agents en parallèle, et
  redémarre tout en cas de crash (variable `cycle`).
- `eval.py` : évalue un modèle entraîné sur N épisodes déterministes.
- `zappy_env.py` : wrapper Gym qui parle le protocole Zappy (`ZappyClient`).

---

## 2. Bug corrigé : épuisement des slots (`No slot for team`)

### Symptôme
Après ~2048 steps (1 rollout PPO complet), l'agent recevait
`RuntimeError: No slot for team 'Br'`, `train.py` crashait, et
`train_dual.py` redémarrait tout depuis zéro. Résultat : 23h d'entraînement
pour seulement 218 "cycles" de redémarrage, sans jamais sauvegarder le modèle.

### Cause
- Le serveur est lancé avec `-c 20` (20 œufs/slots par équipe).
- Chaque `ZappyEnv.reset()` après une mort (`dead`) **consomme un œuf** en
  rouvrant une connexion (`ZappyClient.connect()`).
- `2048 steps / ~105 steps par épisode ≈ 20 épisodes` → exactement le nombre
  d'œufs disponibles. Une fois les 20 œufs consommés, plus aucune
  reconnexion n'est possible.

### Fix (`zappy_env.py`, `ZappyClient.connect()`)
Après chaque connexion réussie, le client envoie immédiatement **`Fork`**
(commande du protocole, page 11 du sujet : *"Once the egg is laid, a new
slot is added to the team"*). Chaque connexion reconstitue ainsi le slot
qu'elle vient de consommer → le pool d'œufs reste constant, peu importe le
nombre de morts/reconnexions.

---

## 3. Nouvel espace d'actions

`ZappyEnv.ACTIONS` est passé de 8 à **19 actions**, pour donner accès à
l'ensemble des ressources du jeu (page 12 : `Take <object>` / `Set <object>`) :

```python
ACTIONS = (
    "Forward", "Left", "Right", "Look", "Inventory",
    "Take food", "Take linemate", "Take deraumere", "Take sibur",
    "Take mendiane", "Take phiras", "Take thystame",
    "Set linemate", "Set deraumere", "Set sibur",
    "Set mendiane", "Set phiras", "Set thystame",
    "Incantation",
)
```

⚠️ Le modèle déjà entraîné (`models/zappy_ppo_*.zip`) a un `action_space`
de taille 8 — il est **incompatible** avec ce nouvel environnement. Il faut
**réentraîner depuis zéro** (supprimer/renommer les `.zip` existants, sinon
`PPO.load` plantera ou tronquera les sorties du réseau).

---

## 4. Nouveau système de reward

Le reward est désormais calé sur les mécaniques **réelles** du sujet plutôt
que sur des valeurs arbitraires.

### a) Coût du temps (page 10 : `action / f`)
Chaque action coûte du temps réel : `7/f` pour les actions simples,
`300/f` pour une `Incantation` (~43x plus cher).

| Action | Coût (`time units`) | Malus de step |
|---|---|---|
| `Forward`, `Left`, `Right`, `Look`, `Inventory`, `Take ...` | 7 | `-0.01` |
| `Incantation` | 300 | `-0.43` |

### b) Mort (page 3 : 1 food = 126 unités de vie)
Mourir = être tombé à 0 food, l'échec total de la survie.

| Événement | Reward |
|---|---|
| `dead` | **-10.0** (terminé) |

### c) Ramasser de la nourriture
La récompense dépend de l'urgence réelle de la situation :

| Niveau de food | Reward |
|---|---|
| `< 30` (zone de danger) | **+2.0** |
| `< 126` (sous la réserve "1 unité de survie") | **+0.5** |
| `>= 126` (surplus) | **+0.05** |

Un malus permanent de **-0.1** s'applique tant que `food < 30`, pour pousser
l'agent à anticiper plutôt qu'à mourir brutalement.

### d) Ramasser / poser une pierre (`Take <pierre>` / `Set <pierre>`)

La table d'élévation (page 5) donne, pour le niveau courant, les quantités
de chaque pierre **nécessaires sur la case** pour le rituel suivant.

**Take `<pierre>` réussi** (la pierre passe au sol → inventaire) :

| Situation | Reward |
|---|---|
| La pierre est encore requise et l'inventaire n'en a pas encore assez (`have <= needed`) | **+1.5** |
| La pierre est requise mais l'inventaire en a déjà assez (surplus) | **+0.1** |
| La pierre n'est pas requise au niveau actuel | **+0.05** |

**Set `<pierre>` réussi** (la pierre passe de l'inventaire → au sol, étape
indispensable car le rituel page 5 exige les pierres *sur la case*, pas dans
l'inventaire) :

| Situation | Reward |
|---|---|
| La case n'a pas encore assez de cette pierre pour le rituel (`on_tile <= needed`) | **+2.0** |
| La case en a déjà assez (pose superflue, pierre "gâchée") | **-0.2** |
| La pierre n'est pas requise au niveau actuel | **-0.1** |

### e) Action `Take`/`Set` ratée (`ko`)
Rien à ramasser, ou rien dans l'inventaire à poser → action gaspillée : **-0.05**.

### f) Incantation

| Résultat | Reward |
|---|---|
| Échec (`ko`) — 300/f bloqués pour rien | **-2.0** |
| Succès, niveau `k → k+1` | `10 + 5 × difficulté(k) × Δniveau` |
| Atteinte du niveau 8 (`MAX_LEVEL`) | **+100** (terminé) |

`difficulté(k)` = somme des joueurs + pierres requis par la table
d'élévation du sujet (page 5) pour passer du niveau `k` à `k+1` :

| Élévation | Joueurs | linemate | deraumere | sibur | mendiane | phiras | thystame | difficulté | reward incantation |
|---|---|---|---|---|---|---|---|---|---|
| 1→2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | **20** |
| 2→3 | 2 | 1 | 1 | 1 | 0 | 0 | 0 | 5 | **35** |
| 3→4 | 2 | 2 | 0 | 1 | 0 | 2 | 0 | 7 | **45** |
| 4→5 | 4 | 1 | 1 | 2 | 0 | 1 | 0 | 9 | **55** |
| 5→6 | 4 | 1 | 2 | 1 | 3 | 0 | 0 | 11 | **65** |
| 6→7 | 6 | 1 | 2 | 3 | 0 | 1 | 0 | 13 | **75** |
| 7→8 | 6 | 2 | 2 | 2 | 2 | 2 | 1 | 17 | **95** (+100 bonus) |

---

## 5. Score "virtuellement parfait"

Avec le nouvel espace d'actions, **tout le jeu devient accessible** (les 6
types de pierres peuvent être ramassées et posées). Le plafond précédent au
niveau 2 (`~+3`) n'existe plus.

### a) Coeur du score : les incantations (inchangé)

```
Somme incantations 1→7 : 20+35+45+55+65+75+95 = 390
Bonus niveau 8          : +100
─────────────────────────────────────────────
Sous-total incantations ≈ 490
```

### b) Bonus Take/Set pour réunir les pierres de chaque palier

Quantités totales de pierres nécessaires sur les paliers 1→8 (table page 5,
sommée colonne par colonne) :

| Pierre | Total requis (1→8) |
|---|---|
| linemate | 9 |
| deraumere | 8 |
| sibur | 10 |
| mendiane | 5 |
| phiras | 6 |
| thystame | 1 |
| **Total** | **39** |

Pour chaque unité, dans le meilleur des cas : 1×`Take` (≈+1.5) + 1×`Set`
(≈+2.0) ≈ **+3.5**. Soit environ `39 × 3.5 ≈ +136`.

### c) Nourriture
Sur un run complet (beaucoup plus long, car il faut explorer/ramasser 39
pierres), il faut se nourrir régulièrement. En comptant ~10-20 `Take food`
utiles (entre +0.5 et +2.0 selon l'urgence) : **≈ +10 à +20**.

### d) Coûts de step
Un run complet prend beaucoup plus de steps qu'avant (collecte de 39
pierres + déplacements + 7 incantations) :
- coût des 7 incantations : `7 × -0.43 ≈ -3`
- coût des steps "normaux" (déplacements, look, etc.) : dépend fortement du
  nombre de steps réellement utilisés, de l'ordre de **-5 à -15**.

### Estimation finale

```
390 (incantations 1-7) + 100 (bonus niveau 8) + ~136 (Take/Set pierres)
+ ~15 (food) - ~10 (coûts de step/incantations)
≈ 600 - 630
```

**Score "virtuellement parfait" ≈ 600-630**, contre ~3 avec l'ancien espace
d'actions limité à `food`/`linemate`. La composante dominante reste les
**incantations (390+100)**, mais la collecte des pierres représente
désormais un bonus substantiel (~20% du total) au lieu d'être totalement
hors de portée.

---

## 6. Lancer l'entraînement

⚠️ **Supprimez d'abord les anciens modèles**, incompatibles avec le nouvel
`action_space` (19 actions au lieu de 8) :

```bash
rm -f models/zappy_ppo_Br.zip models/zappy_ppo_of.zip
```

```bash
make            # build zappy_server / zappy_gui / zappy_ai
python3 src/train_dual.py --config-a configs/agent_br.yaml --config-b configs/agent_of.yaml
```

Les logs sont écrits dans `data/train_Br.log` et `data/train_of.log`.
Le modèle n'est sauvegardé (`models/zappy_ppo_*.zip`) qu'à la fin d'un
`total_timesteps` complet (500 000 steps par défaut) — surveillez
`data/train_*.log` pour vérifier la progression (`X/500000 (Y%)`).
