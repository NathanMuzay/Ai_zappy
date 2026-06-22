"""elevation_guide.py — Guide d'élévation Zappy (reward shaping dense).

Ralliement directionnel optimal pour TOUS les paliers (1->8).
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

# Conversion son K (0..8) -> action optimale d'approche.
# On tourne d'abord vers la source puis on avance (convergence multi-step).
_DIRECTION_TO_ACTION = {
    0: "Incantation",   # émetteur ici
    1: "Forward",       # devant
    2: "Forward",       # devant-gauche : avancer rapproche
    3: "Left",          # gauche : pivoter d'abord
    4: "Left",          # arrière-gauche
    5: "Left",          # arrière : demi-tour amorcé
    6: "Right",         # arrière-droite
    7: "Right",         # droite : pivoter d'abord
    8: "Forward",       # devant-droite : avancer rapproche
}


def set_actions_reference(actions: list[str]) -> None:
    global _ACTIONS_CACHE
    _ACTIONS_CACHE = list(actions)


def missing_stones(level: int, inventory: dict) -> dict:
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


def rally_action(incant_call_dir: int | None) -> str | None:
    """Action pour se rapprocher de l'émetteur du join_incant."""
    if incant_call_dir is None:
        return None
    return _DIRECTION_TO_ACTION.get(incant_call_dir, "Forward")


def recommended_action(level: int, inventory: dict, current_tile: dict,
                       food: int, players_on_tile: int = 1,
                       incant_call_dir: int | None = None,
                       ready_to_join: bool = False) -> str:
    """Priorités : survie -> collecte -> dépôt -> ralliement -> appel -> incanter."""
    # 1. Survie
    if food < 15 and current_tile.get("food", 0) > 0:
        return "Take food"
    if food < 8:
        return "Forward"

    if level >= MAX_LEVEL:
        return "Look"

    miss = missing_stones(level, inventory)

    # 2. Collecte des pierres manquantes
    if miss:
        for s in STONES:
            if s in miss and current_tile.get(s, 0) > 0:
                return f"Take {s}"
        return "Forward"

    # 3. Dépôt des pierres sur la case
    _, req = ELEVATION_REQUIREMENTS[level]
    for s in STONES:
        if current_tile.get(s, 0) < req.get(s, 0):
            return f"Set {s}"

    need_players = required_players(level)

    # 4. RALLIEMENT : si prêt et qu'un appel arrive, converger vers l'émetteur
    if ready_to_join and incant_call_dir is not None and players_on_tile < need_players:
        ral = rally_action(incant_call_dir)
        if ral is not None:
            return ral

    # 5. Pierres prêtes mais pas assez de joueurs : appeler les alliés
    if players_on_tile < need_players:
        if "Broadcast join_incant" in _ACTIONS_CACHE:
            return "Broadcast join_incant"
        return "Connect_nbr"

    # 6. Tout est prêt -> incanter
    return "Incantation"


def guidance_reward(action_name: str, level: int, inventory: dict,
                    current_tile: dict, food: int, players_on_tile: int = 1,
                    incant_call_dir: int | None = None,
                    ready_to_join: bool = False,
                    bonus: float = 1.5, malus: float = -0.05) -> float:
    reco = recommended_action(level, inventory, current_tile, food,
                              players_on_tile, incant_call_dir, ready_to_join)
    if action_name == reco:
        return bonus
    if action_name in ("Look", "Inventory", "Forward", "Right", "Left"):
        return 0.0
    return malus
