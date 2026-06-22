# Système de récompense — Zappy RL

## Objectif du jeu

Victoire = 6 joueurs au niveau 8. Récompenses denses (guidage) + creuses (objectif final).

Le système fonctionne en 2 phases :
- **DÉBUT** (level 1–3) : survie + collecte
- **FIN** (level 4–7) : élévation + assemblage des pierres

## Composantes (définies dans `src/rewards.py`)

| Constante / règle | Valeur | Rôle |
|---|---|---|
| Élévation (`event["elevation"]`) | `+30 + 10 × level` | Signal de progression principal, de plus en plus récompensé |
| Victoire (`event["win"]`, niveau 8) | `+200` | Récompense terminale dominante |
| Mort (`alive = False`) | `-20` | Famine, fortement pénalisée |
| Take food | `+2.0` si `food ≤ 30`, sinon `+0.4` | Pression survie en crise famine |
| Take pierre utile au palier | `+2.0` | Guidage direct vers les pierres nécessaires |
| Take pierre non (encore) utile | `+0.1` | Évite l'immobilisme, pas de collecte inutile |
| Set pierre requise au palier | `+1.0` | Encourage à poser avant Incantation |
| Set pierre non requise | `-0.2` | Dissuade les actions stériles |
| Take/Set raté (`ok = False`) | `-0.05` | Pénalité douce pour action impossible |
| Pression famine | `-0.1 / step` quand `food ≤ 30` | Incite à manger activement |
| Coût temps | `-0.005 / step` | Évite la passivité |
| Fork n°1 | `+1.00` | Bonne pratique : les œufs rendent les élévations haut niveau possibles |
| Fork n°2 | `+0.45` | Toujours encouragé, moins urgent |
| Fork n°3 | `+0.20` | Devient du bruit |
| Fork n°4 | `-0.41` | Début du spam |
| Fork n°5 | `-0.91` | Spam assumé |
| Fork n°6+ | `-1.41 …` | Quasi nul, contre-productif |

## Logique de "pierre utile"

La fonction `_req(level)` lit la table `ELEVATION` (issue du PDF Zappy) et retourne les pierres requises pour atteindre le niveau suivant. La fonction `_stone_shortage()` compare l'inventaire aux pierres du palier courant : seules les pierres avec un manque relatif sont valorisées. Cela évite que l'agent gaspille des steps à ramasser des pierres superflues.

## Façonnage (reward shaping)

- **Récompenses denses** : Take food, Take pierre utile, Set pierre requise, pression famine, Fork dégressif → signal constant.
- **Récompenses creuses** : Élévation, Victoire → vrai objectif.
- **Coût temps** : empêche la passivité.
- **Fork dégressif** : apprend que forker est utile sans encourages le spam.

## Réglage

Toutes les valeurs sont centralisées en haut de `rewards.py` :

```
_FORK_BASE    = 1.0    _FORK_DECAY  = 0.45    _FORK_FREE  = 2    _FORK_PENALTY = 0.5
_FOOD_CRISIS  = 30
_poids        = 0.05 si level ≤ 3, 0.15 sinon
```

- Augmenter le reward d'élévation → accélère la priorisation de l'élévation.
- Augmenter `Take food` → renforce la survie.
- Baisser `_FORK_DECAY` ou `_FORK_FREE` → réduit plus vite l'incitation à forker.
