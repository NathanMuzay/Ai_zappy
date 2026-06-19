"""rewards.py — Systeme de points avec CURRICULUM LEARNING."""

RESOURCES = ("food", "linemate", "deraumere", "sibur",
             "mendiane", "phiras", "thystame")
MAX_LEVEL = 8

ELEVATION = {
    1: {"players": 1, "linemate": 1, "deraumere": 0, "sibur": 0, "mendiane": 0, "phiras": 0, "thystame": 0},
    2: {"players": 2, "linemate": 1, "deraumere": 1, "sibur": 1, "mendiane": 0, "phiras": 0, "thystame": 0},
    3: {"players": 2, "linemate": 2, "deraumere": 0, "sibur": 1, "mendiane": 0, "phiras": 2, "thystame": 0},
    4: {"players": 4, "linemate": 1, "deraumere": 1, "sibur": 2, "mendiane": 0, "phiras": 1, "thystame": 0},
    5: {"players": 4, "linemate": 1, "deraumere": 2, "sibur": 1, "mendiane": 3, "phiras": 0, "thystame": 0},
    6: {"players": 6, "linemate": 1, "deraumere": 2, "sibur": 3, "mendiane": 0, "phiras": 1, "thystame": 0},
    7: {"players": 6, "linemate": 2, "deraumere": 2, "sibur": 2, "mendiane": 2, "phiras": 2, "thystame": 1},
}
STONES = ("linemate", "deraumere", "sibur", "mendiane", "phiras", "thystame")

# Gestion globale de la phase d'apprentissage
_GLOBAL_TIMESTEPS = 0
_PHASE1_THRESHOLD = 500000
_CURRENT_PHASE = 1

def set_global_timesteps(ts: int):
    """Appelé par le callback CurriculumCallback."""
    global _GLOBAL_TIMESTEPS, _CURRENT_PHASE
    _GLOBAL_TIMESTEPS = ts
    _CURRENT_PHASE = 1 if ts < _PHASE1_THRESHOLD else 2

def get_phase() -> int:
    """Retourne la phase actuelle (1 ou 2)."""
    return _CURRENT_PHASE

def difficulty(level: int) -> int:
    """Somme joueurs + pierres requises pour k -> k+1."""
    req = ELEVATION.get(level, {})
    return req.get("players", 0) + sum(req.get(s, 0) for s in STONES)

def needed_on_tile(level: int, stone: str) -> int:
    return ELEVATION.get(level, {}).get(stone, 0)

def compute_reward(prev: dict, state: dict, event: dict) -> float:
    if event is None:
        event = {}

    r = 0.0
    action = event.get("action", "")
    level = state.get("level", 1)
    food = state.get("inventory", {}).get("food", 0)
    phase = get_phase()

    # === a) Coût du temps (réduit) ===
    if action == "Incantation":
        r -= 0.1
    # plus de pénalité par action : on veut que survivre soit rentable

    # === b) Mort ===
    if event.get("death"):
        if phase == 1:
            return r - 10.0
        return r - 5.0

    # === SHAPING NOURRITURE ===
    prev_dist = prev.get("food_dist")
    cur_dist = state.get("food_dist")
    if cur_dist is not None and prev_dist is not None:
        if cur_dist < prev_dist:
            r += 0.3
        elif cur_dist > prev_dist:
            r -= 0.05

    # Baseline survie : VIVRE EST POSITIF (clef de la phase 1)
    r += 0.1

    # === c) Ramasser nourriture ===
    prev_food = prev.get("inventory", {}).get("food", 0)
    if food > prev_food:
        if phase == 1:
            if food < 30:
                r += 10.0
            elif food < 126:
                r += 5.0
            else:
                r += 2.0
        else:
            if food < 30:
                r += 2.0
            elif food < 126:
                r += 0.5
            else:
                r += 0.05

    # Famine (seuil bas seulement)
    if food < 10:
        r -= 0.1

    # === d) Take / Set pierre ===
    if action.startswith("Take ") and event.get("ok"):
        stone = action.split(" ", 1)[1]
        if stone in STONES:
            needed = needed_on_tile(level, stone)
            have = state.get("inventory", {}).get(stone, 0)
            if needed > 0 and have <= needed:
                r += 1.5
            elif needed > 0:
                r += 0.1
            else:
                r += 0.05
    elif action.startswith("Set ") and event.get("ok"):
        stone = action.split(" ", 1)[1]
        if stone in STONES:
            needed = needed_on_tile(level, stone)
            on_tile = event.get("tile_count", 0)
            if needed > 0 and on_tile <= needed:
                r += 2.0
            elif needed > 0:
                r -= 0.2
            else:
                r -= 0.1

    if action.startswith(("Take ", "Set ")) and event.get("ok") is False:
        r -= 0.05

    # === f) Incantation ===
    if action == "Incantation":
        if event.get("ko"):
            r -= 1.0
        elif event.get("level_up"):
            old = event.get("old_level", level)
            delta = state.get("level", old) - old
            if state.get("level") == MAX_LEVEL:
                r += 100.0
            else:
                r += 10 + 5 * difficulty(old) * max(delta, 1)

    # === g) Fork ===
    if action == "Fork" and event.get("ok"):
        if phase == 1:
            r += 0.5
        else:
            r += 8.0

    return r
