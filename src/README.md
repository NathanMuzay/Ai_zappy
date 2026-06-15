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

## 3. Nouvel espace d'observation et d'actions

### Actions (20, au lieu de 8 initialement)

`ZappyEnv.ACTIONS` donne désormais accès à l'ensemble des ressources du jeu
(page 12 : `Take <object>` / `Set <object>`), ainsi qu'à `Broadcast` pour la
coordination multi-agent (section 7) :

```python
ACTIONS = (
    "Forward", "Left", "Right", "Look", "Inventory",
    "Take food", "Take linemate", "Take deraumere", "Take sibur",
    "Take mendiane", "Take phiras", "Take thystame",
    "Set linemate", "Set deraumere", "Set sibur",
    "Set mendiane", "Set phiras", "Set thystame",
    "Incantation", "Broadcast",
)
```

### Observation (21 dims)

| Champ | Dims | Description |
|---|---|---|
| `inventory` | 7 | Inventaire normalisé (page 12), une valeur par ressource. |
| `tile` | 7 | Ressources au sol sur la case courante (page 6, tile 0). |
| `level` | 1 | Niveau courant / `MAX_LEVEL`. |
| `step_ratio` | 1 | Progression dans l'épisode (`step / max_steps`). |
| `players_here` | 1 | Nb de joueurs sur la case (soi inclus, page 6), normalisé /6. |
| `players_needed` | 1 | Nb de joueurs requis pour la prochaine élévation (table page 5), normalisé /6. |
| `broadcast` | 3 | Canal de coordination : message reçu ? direction ? niveau annoncé ? (voir section 7). |

`players_here` / `players_needed` donnent à la politique l'information
nécessaire pour savoir si les conditions de **groupe** de la table
d'élévation (page 5) sont réunies — condition indispensable pour viser les
niveaux 2 à 8.

⚠️ Le modèle déjà entraîné (`models/zappy_ppo_*.zip`) a un `action_space`
de taille 8 et un `observation_space` de taille 16 — **incompatible** avec
ce nouvel environnement. Il faut **réentraîner depuis zéro** (supprimer/
renommer les `.zip` existants, sinon `PPO.load` plantera ou tronquera les
sorties du réseau).

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

### g) Broadcast (coordination, page 6-7 et 12-13)

Diffuse `"Broadcast LVL<niveau>"` (entendu par **tous** les joueurs, toutes
équipes confondues, page 7) pour signaler qu'on est prêt pour une élévation
de groupe.

| Situation | Reward |
|---|---|
| Coût de base (action simple, `7/f`) | **-0.01** |
| + élévation suivante nécessite >1 joueur **et** les pierres requises sont déjà au sol sur la case | **+0.5** (donc net **+0.49**) |

C'est un "appel à l'aide" : ne rapporte un bonus que si broadcaster a
vraiment du sens à ce moment précis (tout est prêt, il ne manque que des
coéquipiers).

---

## 5. Tuning PPO (suite au premier run complet de 500k steps)

Le premier run avec le nouveau reward s'est **terminé normalement** (fix du
`Fork` validé : `Training complete`, modèles sauvegardés) en ~22h, avec
`ep_rew_mean` passant de **-16 à +4/+15**. Mais `max_level` restait bloqué
à **1** quasiment tout le run. Deux ajustements `algo:` dans
`agent_br.yaml` / `agent_of.yaml` :

| Paramètre | Avant | Après | Pourquoi |
|---|---|---|---|
| `ent_coef` | `0.05` | `0.01` | 5x la valeur par défaut de PPO. Avec 19 actions (vs 8), une entropie élevée étale la politique sur des `Take`/`Set` hors-sujet. La réduire laisse l'agent converger vers les comportements payants (manger, ramasser/poser les bonnes pierres, incanter). |
| `gamma` | `0.99` | `0.995` | Une `Incantation` réussie ne paie que si l'agent a fait `Take`→`Set` plusieurs steps avant. À `0.99`, une récompense à 50 steps est pondérée par `0.99^50≈0.61` ; à `0.995`, par `0.995^50≈0.78` — meilleur lien de causalité sur les séquences longues. |

`clip_range` et `vf_coef` ont été retirés de `agent_br.yaml` : `train.py`
ne les lit pas (seuls `learning_rate`, `n_steps`, `batch_size`, `gamma`,
`gae_lambda`, `ent_coef`, `n_epochs` sont passés au constructeur `PPO`), ces
clés étaient donc sans effet.

---

## 6. Score "virtuellement parfait" — par joueur

Avec le nouvel espace d'actions, **tout le jeu devient accessible** (les 6
types de pierres peuvent être ramassées et posées). Le plafond précédent au
niveau 2 (`~+3`) n'existe plus — *par joueur*, indépendamment du nombre de
joueurs réellement présents (voir section 7 pour le multi-agent).

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

**Score "virtuellement parfait" par joueur ≈ 600-630**, contre ~3 avec
l'ancien espace d'actions limité à `food`/`linemate`. La composante
dominante reste les **incantations (390+100)**, mais la collecte des
pierres représente désormais un bonus substantiel (~20% du total) au lieu
d'être totalement hors de portée.

⚠️ Ce chiffre suppose qu'un seul joueur atteint le niveau 8 — or 6→7 et
7→8 demandent **6 joueurs au même niveau, même case** (table page 5).
Sans coordination multi-joueurs, ce plafond reste théorique. C'est l'objet
de la section suivante.

---

## 7. Architecture multi-agent (parameter sharing)

### Le problème résolu

Chaque connexion client = un joueur (page 9). Avec **un seul client par
équipe**, l'agent ne peut jamais réunir les `nb_players ≥ 2` requis par la
table d'élévation (page 5) à partir du niveau 2→3 — il reste bloqué au
niveau 1 (incantation 1→2 réalisable seul).

### La solution : `n_players`

`agent_br.yaml` / `agent_of.yaml` ont un nouveau champ :

```yaml
n_players: 4
```

`train.py` ouvre **`n_players` connexions simultanées** à la même équipe
(`make_vec_env(ZappyEnv, n_envs=n_players, ...)`). Concrètement :

- `n_players` joueurs distincts apparaissent sur la carte pour l'équipe.
- **Une seule politique PPO** contrôle les `n_players` joueurs en parallèle
  (*parameter sharing* — l'approche multi-agent standard compatible avec
  Stable-Baselines3, sans framework MARL dédié).
- Le pool d'œufs (`-c 20` dans `train_dual.py`) + le `Fork` automatique
  (section 2) absorbent largement `n_players=4` par équipe.
- Sous `DummyVecEnv` (par défaut), les sous-environnements sont exécutés
  **séquentiellement** : le temps mur pour `total_timesteps` reste
  globalement le même qu'avec `n_players=1`, mais chaque pas de simulation
  fait avancer `n_players` joueurs au lieu d'un seul.

### Coordination via Broadcast

Avec plusieurs joueurs contrôlés par la même politique mais sans canal de
communication, les faire converger sur la même case au même niveau reste
largement hasardeux. L'action `Broadcast` (section 4g) + le canal
d'observation correspondant (section 3) donnent à la politique un outil
explicite : un joueur "prêt" (pierres au sol, juste besoin de coéquipiers)
diffuse son niveau ; les autres joueurs — y compris ceux de l'équipe
adverse, le protocole ne distingue pas (page 7) — perçoivent la direction
et le niveau annoncés et peuvent s'orienter en conséquence.

### Niveaux atteignables avec `n_players=4`

| Élévation | Joueurs requis | Atteignable avec 4 joueurs/équipe ? |
|---|---|---|
| 1→2 | 1 | ✅ |
| 2→3 | 2 | ✅ |
| 3→4 | 2 | ✅ |
| 4→5 | 4 | ✅ (tous les 4 doivent être niveau 4, même case) |
| 5→6 | 4 | ✅ (idem, plus exigeant) |
| 6→7 | 6 | ❌ (nécessiterait une convergence inter-équipe via Broadcast, hasardeux) |
| 7→8 | 6 | ❌ (idem) |

Pour viser 6→7 / 7→8, augmenter `n_players` à 6 (toujours absorbable par
`-c 20` + `Fork`), au prix d'un monde plus chargé (plus de connexions
simultanées, charge CPU serveur en hausse).

### Score réaliste avec `n_players=4` (objectif : atteindre le niveau 6)

```
Incantations 1→6 : 20+35+45+55+65 = 220
Pierres requises 1→6 (somme table page 5) : 21 unites x ~3.5 (Take+Set) ≈ 73.5
Food (survie sur un episode plus long)     : ≈ +10 a +15
Couts de step/incantations                 : ≈ -5 a -10
─────────────────────────────────────────────
Par joueur ≈ 290-300
```

`team_reward` (somme sur les 4 joueurs, métrique affichée par `eval.py`)
≈ **4 × 290-300 ≈ 1150-1200** si les 4 joueurs atteignent le niveau 6 — un
scénario optimiste mais réaliste comme objectif d'entraînement, bien plus
proche de "l'attendu" du sujet que le plafond solo (~+3 à +20).

### Évaluation multi-joueurs (`eval.py`)

`eval_br.yaml` / `eval_of.yaml` ont aussi `n_players: 4` : `eval.py`
instancie `n_players` `ZappyEnv` et fait avancer chaque joueur avec la même
politique chargée, puis affiche un `team_reward` (somme des rewards
individuels) en plus du détail par joueur — c'est la métrique pertinente
pour juger si des élévations de groupe ont vraiment lieu.

---

## 8. Lancer l'entraînement

⚠️ **Supprimez d'abord les anciens modèles**, incompatibles avec le nouvel
`action_space` (20 actions au lieu de 8/19) et `observation_space` (21 dims
au lieu de 16/18) :

```bash
rm -f models/zappy_ppo_Br.zip models/zappy_ppo_of.zip
```

```bash
make            # build zappy_server / zappy_gui / zappy_ai
python3 src/train_dual.py --config-a configs/agent_br.yaml --config-b configs/agent_of.yaml
```

Les logs sont écrits dans `data/train_Br.log` et `data/train_of.log` —
attendez-vous à une ligne `Multi-agent: 4 joueur(s) simultane(s) pour
l'equipe '...'` au démarrage de chaque agent.
Le modèle n'est sauvegardé (`models/zappy_ppo_*.zip`) qu'à la fin d'un
`total_timesteps` complet (500 000 steps par défaut) — surveillez
`data/train_*.log` pour vérifier la progression (`X/500000 (Y%)`).

Pour évaluer ensuite (et voir le `team_reward`) :

```bash
python3 src/run_eval.py --config configs/eval_br.yaml
python3 src/run_eval.py --config configs/eval_of.yaml
```
