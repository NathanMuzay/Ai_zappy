# Système de récompense — Zappy RL

## Objectif du jeu
La victoire = **6 joueurs au niveau maximum (8)**. Le reward est donc
construit pour guider l'agent vers ce but, étape par étape (récompenses
denses) tout en valorisant fortement l'objectif final (récompense creuse).

## Composantes (définies dans `src/rewards.py`)

| Constante         | Valeur | Rôle |
|-------------------|--------|------|
| `W_LEVEL_UP`      | +100   | Franchir un palier d'élévation. C'est le signal de progression principal. |
| `W_WIN`           | +1000  | Atteindre le niveau max. Récompense terminale dominante. |
| `W_DEATH`         | -200   | Mort par famine. Fortement pénalisée car termine l'épisode. |
| `W_TAKE_USEFUL`   | +5     | Ramasser une pierre **utile** au prochain palier. |
| `W_TAKE_FOOD`     | +2     | Ramasser de la nourriture (survie). |
| `W_SET_WRONG`     | -3     | Poser une pierre non nécessaire (gaspillage). |
| `W_STARVE_RISK`   | -0.05  | Malus proportionnel quand food < 50 (anticipation famine). |
| `W_TIME`          | -0.01  | Pression temporelle : pousse à l'efficacité. |

## Logique de "pierre utile"
`_useful_stones(level)` lit `ELEVATION_REQUIREMENTS` (issu du PDF) et ne
retient que les pierres dont la quantité requise pour le prochain palier
est > 0. Cela évite que l'agent collectionne des ressources inutiles.

## Façonnage (reward shaping)
- Les récompenses **denses** (take food/stones, malus famine) maintiennent
  un signal d'apprentissage constant malgré la rareté des élévations.
- Les récompenses **creuses** (level up, win) définissent le vrai objectif.
- Le malus temporel évite les comportements passifs (tourner en rond).

## Réglage
Toutes les valeurs sont centralisées en haut de `rewards.py`. Augmenter
`W_LEVEL_UP` accélère la priorisation de l'élévation ; augmenter
`W_TAKE_FOOD` renforce la survie si l'agent meurt trop tôt.
