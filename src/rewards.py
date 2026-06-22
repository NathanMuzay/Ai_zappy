"""rewards.py — Système de points : CURRICULUM + COORDINATION + GUIDAGE D'ÉLÉVATION."""

from src import elevation_guide

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

# ═══════════════════════════════════════════════════════════════
#  CONSTANTES DE REWARD (réglables)
# ═══════════════════════════════════════════════════════════════

# --- Nourriture phase 1 / phase 2 ---
_FOOD_PHASE1_TAKE_ABUNDANT   = 10.0
_FOOD_PHASE1_TAKE_MODERATE   = 5.0
_FOOD_PHASE1_TAKE_LOW        = 2.0
_FOOD_PHASE2_TAKE_ABUNDANT   = 2.0
_FOOD_PHASE2_TAKE_MODERATE   = 0.5
_FOOD_PHASE2_TAKE_LOW        = 0.05

# --- Pierres utiles (Take) ---
_STONE_USEFUL_TAKE  = 1.5
_STONE_SIDE_TAKE    = 0.1
_STONE_NEVER_TAKE   = 0.05

# --- Pierres sur case (Set) ---
_STONE_SET_DEPOSIT  = 2.0
_STONE_SET_REGRET   = -0.2

# --- Incantation ---
_INCANTATION_KO      = -1.0
_INCANTATION_COST    = -0.1
_INCANTATION_SUCCESS = 10.0
_INCANTATION_FINAL   = 100.0
_DIFFICULTY_MULTIPLIER = 5

# --- Survie / temps ---
_SURVIVAL_BONUS     = 0.1
_FOOD_CRISIS        = -0.1

# --- Fork ---
_FORK_PHASE1  = 0.5
_FORK_PHASE2  = 8.0
_FORK_BASE    = 1.0
_FORK_DECAY   = 0.001
_FORK_FREE    = 400
_FORK_PENALTY = 5.0

# --- Coordination ---
_BROADCAST_COORD_SEND     = 0.3   # reward quand on émet un Broadcast utile
_BROADCAST_COORD_RECV     = 0.05  # reward quand un message de frère est reçu
_BROADCAST_USELESS        = -0.05 # malus si Broadcast émis sans raison
_GROUP_REWARD_PER_PLAYER  = 0.2
_PREINCANT_READY          = 3.0
_PREINCANT_LOST           = -0.5

# --- Guidage d'élévation (reward shaping dense) ---
# bonus à RÉDUIRE progressivement entre phases (voir README).
_GUIDE_BONUS_PHASE1 = 1.5
_GUIDE_BONUS_PHASE2 = 0.7
_GUIDE_MALUS        = -0.05

# ═══════════════════════════════════════════════════════════════
#  Gestion globale de phase (curriculum)
# ═══════════════════════════════════════════════════════════════

_GLOBAL_TIMESTEPS = 0
_PHASE1_THRESHOLD = 500000
_CURRENT_PHASE = 1

# ═══════════════════════════════════════════════════════════════
#  Gestion fork (agent_id -> fork_start_time)
# ═══════════════════════════════════════════════════════════════
_FORK_COUNTS = {}


def set_global_timesteps(ts: int):
    """Appelé par CurriculumCallback pour piloter la phase."""
    global _GLOBAL_TIMESTEPS, _CURRENT_PHASE
    _GLOBAL_TIMESTEPS = ts
    _CURRENT_PHASE = 1 if ts < _PHASE1_THRESHOLD else 2


def get_phase() -> int:
    return _CURRENT_PHASE


def get_global_timesteps() -> int:
    return _GLOBAL_TIMESTEPS


def difficulty(level: int) -> int:
    """Somme joueurs + pierres requises pour k -> k+1."""
    req = ELEVATION.get(level, {})
    return req.get("players", 0) + sum(req.get(s, 0) for s in STONES)


def needed_on_tile(level: int, stone: str) -> int:
    return ELEVATION.get(level, {}).get(stone, 0)


# ── helpers coordination ──────────────────────────────────────

def _check_preincant_ready(state: dict, event: dict) -> float:
    """Reward si toutes les pierres du palier + assez de joueurs sont sur la case."""
    level = state.get("level", 1)
    if level >= MAX_LEVEL:
        return 0.0
    req = ELEVATION.get(level, {})
    vision = state.get("vision", [])
    if not vision:
        return 0.0
    cur_tile = vision[0]

    players_here = cur_tile.get("player", 0)
    players_needed = req.get("players", 1)

    all_stones_ok = True
    for stone in STONES:
        if cur_tile.get(stone, 0) < req.get(stone, 0):
            all_stones_ok = False
            break

    if players_here >= players_needed and all_stones_ok:
        return _PREINCANT_READY
    return 0.0


def compute_reward(prev: dict, state: dict, event: dict) -> float:
    if event is None:
        event = {}

    r = 0.0
    action = event.get("action", "")
    level = state.get("level", 1)
    inventory = state.get("inventory", {})
    food = inventory.get("food", 0)
    phase = get_phase()

    # case courante (pour le guidage)
    vision = state.get("vision", [])
    cur_tile = vision[0] if vision else {}
    players_here = cur_tile.get("player", 0)

    # ── a) Coût du temps ──────────────────────────────────────
    if action == "Incantation":
        r += _INCANTATION_COST

    # ── b) Mort ───────────────────────────────────────────────
    if event.get("death"):
        aid = event.get("agent_id", 0)
        _FORK_COUNTS.pop(aid, None)
        return r + (-10.0 if phase == 1 else -5.0)

    # ── c) Survie ─────────────────────────────────────────────
    r += _SURVIVAL_BONUS

    # ── d) Shaping nourriture ──────────────────────────────────
    prev_dist = prev.get("food_dist")
    cur_dist = state.get("food_dist")
    if cur_dist is not None and prev_dist is not None:
        if cur_dist < prev_dist:
            r += 0.3
        elif cur_dist > prev_dist:
            r -= 0.05

    prev_food = prev.get("inventory", {}).get("food", 0)
    if food > prev_food:
        if phase == 1:
            if food < 30:
                r += _FOOD_PHASE1_TAKE_ABUNDANT
            elif food < 126:
                r += _FOOD_PHASE1_TAKE_MODERATE
            else:
                r += _FOOD_PHASE1_TAKE_LOW
        else:
            if food < 30:
                r += _FOOD_PHASE2_TAKE_ABUNDANT
            elif food < 126:
                r += _FOOD_PHASE2_TAKE_MODERATE
            else:
                r += _FOOD_PHASE2_TAKE_LOW

    if food < 10:
        r += _FOOD_CRISIS

    # ── e) Take / Set pierre ──────────────────────────────────
    if action.startswith("Take ") and event.get("ok"):
        stone = action.split(" ", 1)[1]
        if stone in STONES:
            needed = needed_on_tile(level, stone)
            if stone == "thystame" and level < 7 and needed == 0:
                r += _STONE_NEVER_TAKE
            elif needed > 0:
                r += _STONE_USEFUL_TAKE
            else:
                r += _STONE_SIDE_TAKE

    elif action.startswith("Set ") and event.get("ok"):
        stone = action.split(" ", 1)[1]
        if stone in STONES:
            needed = needed_on_tile(level, stone)
            if needed > 0:
                r += _STONE_SET_DEPOSIT
            else:
                r -= 0.1

    if action.startswith(("Take ", "Set ")) and event.get("ok") is False:
        r -= 0.05

    # ── f) Fork dégressif ─────────────────────────────────────
    if action == "Fork" and event.get("ok"):
        if phase == 1:
            r += _FORK_PHASE1
        else:
            aid = event.get("agent_id", 0)
            fork_time = _FORK_COUNTS.get(aid, 0)
            elapsed = fork_time - _FORK_FREE
            if elapsed > 0:
                penalty = min(elapsed * _FORK_DECAY, _FORK_PENALTY)
                r += _FORK_BASE - penalty
            else:
                r += _FORK_BASE
            _FORK_COUNTS[aid] = fork_time + 1

    # ── g) Incantation ───────────────────────────────────────
    if action == "Incantation":
        if event.get("ko"):
            r += _INCANTATION_KO
        elif event.get("elevation") or event.get("level_up"):
            old = event.get("old_level", level)
            delta = state.get("level", old) - old
            if state.get("level") == MAX_LEVEL:
                r += _INCANTATION_FINAL
            else:
                r += _INCANTATION_SUCCESS + _DIFFICULTY_MULTIPLIER * difficulty(old) * max(delta, 1)

    # ── h) Coordination : regroupement ───────────────────────
    if players_here > 1:
        r += (players_here - 1) * _GROUP_REWARD_PER_PLAYER

    # ── i) Pré-incantation prête ──────────────────────────────
    r += _check_preincant_ready(state, event)

    # ── j) Broadcast émis (NOUVEAU) ───────────────────────────
    if action.startswith("Broadcast") and event.get("ok"):
        # n'est utile que si l'élévation requiert plus de joueurs
        # qu'il n'y en a actuellement sur la case.
        need_players = ELEVATION.get(level, {}).get("players", 1)
        stones_ready = elevation_guide.tile_has_required_stones(level, cur_tile)
        if stones_ready and players_here < need_players:
            r += _BROADCAST_COORD_SEND  # appel justifié
        else:
            r += _BROADCAST_USELESS     # spam de broadcast inutile

    # ── k) Messages de coordination reçus ─────────────────────
    msg = event.get("message")
    if msg and ("incant" in msg.lower() or "join" in msg.lower()
                or "elevation" in msg.lower() or "ready" in msg.lower()
                or "collect" in msg.lower()):
        r += _BROADCAST_COORD_RECV

       # ── l) GUIDAGE D'ÉLÉVATION (reward shaping dense) ─────────
    guide_bonus = _GUIDE_BONUS_PHASE1 if phase == 1 else _GUIDE_BONUS_PHASE2
    miss = elevation_guide.missing_stones(level, inventory)
    ready_to_join = (len(miss) == 0 and level < MAX_LEVEL)
    incant_call_dir = event.get("incant_call_dir")
    r += elevation_guide.guidance_reward(
        action_name=action,
        level=level,
        inventory=inventory,
        current_tile=cur_tile,
        food=food,
        players_on_tile=players_here if players_here > 0 else 1,
        incant_call_dir=incant_call_dir,
        ready_to_join=ready_to_join,
        bonus=guide_bonus,
        malus=_GUIDE_MALUS,
    )

    return r
