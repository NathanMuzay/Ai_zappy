"""elevation_guide.py — Guide d'élévation pour Zappy (reward shaping dense).

Transforme la récompense creuse (incantation réussie) en signal dense :
à chaque step on connaît l'action qui rapproche de la prochaine élévation.

Recettes officielles (niveau courant -> exigences pour le niveau suivant) :
  1->2 : 1 joueur,  linemate=1
  2->3 : 2 joueurs, linemate=1 deraumere=1 sibur=1
  3->4 : 2 joueurs, linemate=2 sibur=1 phiras=2
  4->5 : 4 joueurs, linemate=1 deraumere=1 sibur=2 phiras=1
  5->6 : 4 joueurs, linemate=1 deraumere=2 sibur=1 mendiane=3
  6->7 : 6 joueurs, linemate=1 deraumere=2 sibur=3 phiras=1
  7->8 : 6 joueurs, linemate=2 deraumere=2 sibur=2 mendiane=2 phiras=2 thystame=1
"""
from __future__ import annotations

ELEVATION_REQUIREMENTS = {
    1: (1, {"linemate": 1, "deraumere": 0, "sibur": 0, "mendiane": 0, "phiras": 0, "thystame": 0}),
    2: (2, {"linemate": 1, "deraumere": 1, "sibur": 1, "mendiane": 0, "phiras": 0, "thystame": 0}),
    3: (2, {"linemate": 2, "deraumere": 0, "sibur": 1, "mendiane": 0, "phiras": 2, "thystame": 0}),
    4: (4, {"linemate": 1, "deraumere": 1, "sibur": 2, "mendiane": 0, "phiras": 1, "thystame": 0}),
    5: (4, {"linemate": 1, "deraumere": 2, "sibur": 1, "mendiane": 3, "phiras": 0, "thystame": 0}),
    6: (6, {"linemate": 1, "deraumere": 2, "sibur": 3, "mendiane": 0, "phiras": 1, "thystame": 0}),
    7: (6, {"linemate": 2, "deraumere": 2, "sibur": 2, "mendiane": 2, "phiras": 2, "thystame": 1}),
}

STONES = ["linemate", "deraumere", "sibur", "mendiane", "phiras", "thystame"]
MAX_LEVEL = 8

_ACTIONS_CACHE: list[str] = []


def set_actions_reference(actions: list[str]) -> None:
    """Injecte protocol.ACTIONS une fois au démarrage (évite l'import circulaire)."""
    global _ACTIONS_CACHE
    _ACTIONS_CACHE = list(actions)


def missing_stones(level: int, inventory: dict) -> dict:
    """Pierres encore manquantes dans l'inventaire pour élever."""
    if level >= MAX_LEVEL or level not in ELEVATION_REQUIREMENTS:
        return {}
    _, req = ELEVATION_REQUIREMENTS[level]
    out = {}
    for s in STONES:
        need = req.get(s, 0) - inventory.get(s, 0)
        if need > 0:
            out[s] = need
    return out


def required_players(level: int) -> int:
    if level not in ELEVATION_REQUIREMENTS:
        return 1
    return ELEVATION_REQUIREMENTS[level][0]


def tile_has_required_stones(level: int, tile: dict) -> bool:
    if level >= MAX_LEVEL or level not in ELEVATION_REQUIREMENTS:
        return False
    _, req = ELEVATION_REQUIREMENTS[level]
    return all(tile.get(s, 0) >= req.get(s, 0) for s in STONES)


def recommended_action(level: int, inventory: dict, current_tile: dict,
                       food: int, players_on_tile: int = 1) -> str:
    """Action conseillée selon l'état. Priorités :
    survie -> collecte -> dépôt -> appel d'alliés (Broadcast) -> incantation.
    """
    # 1. Survie
    if food < 15 and current_tile.get("food", 0) > 0:
        return "Take food"
    if food < 8:
        return "Forward"  # explorer pour trouver de la nourriture

    if level >= MAX_LEVEL:
        return "Look"

    miss = missing_stones(level, inventory)

    # 2. Collecte des pierres manquantes
    if miss:
        for s in STONES:
            if s in miss and current_tile.get(s, 0) > 0:
                return f"Take {s}"
        return "Forward"  # rien d'utile ici -> explorer

    # 3. Dépôt des pierres sur la case
    _, req = ELEVATION_REQUIREMENTS[level]
    for s in STONES:
        if current_tile.get(s, 0) < req.get(s, 0):
            return f"Set {s}"

    # 4. Pierres prêtes : faut-il appeler des alliés ?
    need_players = required_players(level)
    if players_on_tile < need_players:
        if "Broadcast join_incant" in _ACTIONS_CACHE:
            return "Broadcast join_incant"
        return "Connect_nbr"

    # 5. Tout est prêt -> incanter
    return "Incantation"


def guidance_reward(action_name: str, level: int, inventory: dict,
                    current_tile: dict, food: int, players_on_tile: int = 1,
                    bonus: float = 1.5, malus: float = -0.05) -> float:
    """+bonus si l'action == action conseillée, malus léger sinon.
    Les actions neutres d'exploration ne sont pas pénalisées.
    """
    reco = recommended_action(level, inventory, current_tile, food, players_on_tile)
    if action_name == reco:
        return bonus
    if action_name in ("Look", "Inventory", "Forward", "Right", "Left"):
        return 0.0
    return malus
