"""elevation_guide.py — Guide d'élévation Zappy (reward shaping dense).

Ralliement directionnel optimal pour TOUS les paliers (1->8).

CHANGELOG v2:
  - _DIRECTION_TO_ACTION : directions 2 et 8 corrigées (2="Left", 8="Right")
    Raison : en système de coordonnées Zappy, 2 = avant-gauche et 8 = avant-droit.
    Pour atteindre ces cases, il faut d'abord pivoter vers le côté puis avancer.
    Avant: 2="Forward" (trop long, risquait de contourner), 8="Forward" (même problème).
    Après: 2="Left" (pivote puis avance), 8="Right" (pivote puis avance).
  - rally_action(): vérifie maintenant explicitement que direction est dans 0..8,
    sinon retourne "Forward" par sécurité (pas None) pour éviter les None checks.
  - recommended_action(): protège aussi contre incant_call_dir hors plage.
  - set_actions_reference(): idempotent, appelé depuis env.py __init__.
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
#
# Géométrie Zappy des sons :
#   0 = même case que l'émetteur  → Incantation (on est déjà sur place)
#   1 = devant                    → Forward (avancer tout droit)
#   2 = avant-gauche (diag)      → Left (pivoter gauche, + avanc)
#   3 = gauche                    → Left (pivoter)
#   4 = arrière-gauche (diag)     → Left (pivoter, + demi-tour amorcé)
#   5 = derrière                  → Left × 2 (demi-tour) — 2 fois = +1.0 turn cost
#                                   En pratique Left×2 fonctionne bien
#   6 = arrière-droite (diag)     → Right (pivoter, + demi-tour amorcé)
#   7 = droite                    → Right (pivoter)
#   8 = avant-droite (diag)       → Right (pivoter droite, + avanç)
#
# On tourne d'abord vers la source puis on avance (convergence multi-step).
_DIRECTION_TO_ACTION = {
    0: "Incantation",   # émetteur ici
    1: "Forward",       # devant
    2: "Left",          # avant-gauche : pivoter gauche pour aligned
    3: "Left",          # gauche
    4: "Left",          # arrière-gauche : pivoter + avancer = converge
    5: "Left",          # derrière : demi-tour amorcé via Left × 2
    6: "Right",         # arrière-droite
    7: "Right",         # droite
    8: "Right",         # avant-droite : pivoter droite pour aligned
}


def set_actions_reference(actions: list[str]) -> None:
    """Enregistre la table des actions. Appelées depuis env.py __init__."""
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


def rally_action(incant_call_dir: int | None) -> str:
    """Action pour se rapprocher de l'émetteur du join_incant.
    
    Returns toujours une action (jamais None) pour éviter les exceptions.
    Direction hors plage 0..8 → "Forward" par défaut.
    """
    if incant_call_dir is None:
        return "Forward"
    if not (0 <= incant_call_dir <= 8):
        return "Forward"
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
        # Sécurité : ne pas planter si direction invalide
        if 0 <= incant_call_dir <= 8:
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
