# Zappy RL — Entraînement par Reinforcement Learning

Système d'entraînement d'une IA jouant à Zappy, basé sur **PPO**
(Proximal Policy Optimization) via *stable-baselines3*.

## 1. Pourquoi PPO ?

| Critère | PPO (choisi) | DQN | A2C |
|---------|-------------|-----|-----|
| Espace d'action discret | ✅ | ✅ | ✅ |
| Stabilité d'apprentissage | ✅ (clipping) | ⚠️ instable | ⚠️ |
| Récompenses creuses | ✅ (avec shaping) | ⚠️ | ⚠️ |
| Échantillonnage | on-policy efficace | replay buffer lourd | on-policy bruité |
| Maturité / support | excellent (SB3) | bon | bon |

**Justification :** Zappy a un espace d'action **discret** (22 commandes) et
des récompenses partiellement creuses (élévations rares). PPO offre le
meilleur compromis stabilité/performance grâce à son *clipping* du ratio de
politique, qui évite les effondrements fréquents avec DQN sur ce type de
problème. Il ne s'agit **pas** de "double triks" (exclu de la consigne).

## 2. Installation

```bash
make install        # crée .venv et installe les dépendances Python
