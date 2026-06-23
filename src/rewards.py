"""rewards.py — Système de points avec CURRICULUM LEARNING + COORDINATION SHAPING.

CHANGELOG v2 — Blocage niveau 2 (2→3) corrigé :
  Le reward de rallyiement était BUGGÉ : récompense TOUT message reçu.
  Nouvelle logique ACTIVE UNIQUEMENT si level >= 2 :
    - call_dir = event["incant_call_dir"] (injecté par env.py depuis parse_broadcast)
    - direction 0 (sur place) : +8.0 bonus d'arrivée (_RALLY_ARRIVED)
    - suivre la direction optimale : +2.0 (_RALLY_FOLLOW_CLOSER)
    - suivre une mauvaise direction : -1.0 (_RALLY_FOLLOW_FARTHER)
    - émettre "Broadcast join_incant" en niveau 2+ : +1.5 (_BROADCAST_CALL_VALID)
  
  Ces rewards PPOcessitent que incant_call_dir soit injecté dans event par env.py
  (via parse_broadcast sur les messages "join_incant" reçus).

  Le reward solo niveau 1 (1→2) n'est PAS modifié.
"""

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
_FOOD_PHASE1_TAKE_ABUNDANT   = 10.0   # food < 30, phase 1
_FOOD_PHASE1_TAKE_MODERATE   = 5.0    # 30 ≤ food < 126, phase 1
_FOOD_PHASE1_TAKE_LOW        = 2.0    # food ≥ 126, phase 1
_FOOD_PHASE2_TAKE_ABUNDANT   = 2.0    # food < 30, phase 2
_FOOD_PHASE2_TAKE_MODERATE   = 0.5    # 30 ≤ food < 126, phase 2
_FOOD_PHASE2_TAKE_LOW        = 0.05   # food ≥ 126, phase 2

# --- Pierres utiles (Take) ---
_STONE_USEFUL_TAKE  = 1.5    # pierre requise pour élévation en cours
_STONE_SIDE_TAKE    = 0.1    # pierre non requise
_STONE_NEVER_TAKE   = 0.05   # pierre jamais requise (thystame niveau < 7)

# --- Pierres sur case (Set) ---
_STONE_SET_DEPOSIT  = 2.0    # on dépose une pierre requise
_STONE_SET_REGRET   = -0.2   # on retire une pierre requise de la case

# --- Incantation ---
_INCANTATION_KO      = -1.0   # échec
_INCANTATION_COST    = -0.1   # coût en temps
_INCANTATION_SUCCESS = 10.0   # succès (avant bonus difficulty)
_INCANTATION_FINAL   = 100.0  # élévation au niveau max
_DIFFICULTY_MULTIPLIER = 5    # bonus = _DIFFICULTY_MULTIPLIER * difficulty * delta

# --- Survie / temps ---
_SURVIVAL_BONUS     = 0.1    # chaque step où l'agent est vivant
_FOOD_CRISIS        = -0.1   # quand food < 10

# --- Fork ---
_FORK_PHASE1  = 0.5    # fork phase 1 (réplication encouragée)
_FORK_PHASE2  = 8.0    # fork phase 2 (équilibrage population)
_FORK_BASE    = 1.0    # reward fork initial
_FORK_DECAY   = 0.001  # pénalité par unité de temps (fork_start_time)
_FORK_FREE    = 400    # pas de pénalité avant cette durée
_FORK_PENALTY = 5.0    # pénalité max

# --- Coordination (v1) ---
_BROADCAST_COORD_SEND     = 0.3   # reward quand on Broadcast (v1)
_GROUP_REWARD_PER_PLAYER  = 0.2   # par joueur supplémentaire sur la case (hors soi)
_PREINCANT_READY          = 3.0   # bonus quand la case est prête pour incantation
_PREINCANT_LOST           = -0.5  # malus quand on perd cette état "prêt"

# ═══════════════════════════════════════════════════════════════
#  NOUVELLES CONSTANTES v2 — RALLYIEMENT MULTI-JOUEURS
# ═══════════════════════════════════════════════════════════════
_RALLY_ARRIVED       = 8.0   # direction 0 = on est sur la case émetteur
_RALLY_FOLLOW_CLOSER = 2.0   # action = optimale pour rejoindre appelant
_RALLY_FOLLOW_FARTHER= -1.0  # action = mouvement mais pas l'optimal
_BROADCAST_CALL_VALID= 1.5   # Broadcast join_incant réussi (level>=2, pas assez joueurs)

# ═══════════════════════════════════════════════════════════════
#  Gestion globale de phase (curriculum)
# ═══════════════════════════════════════════════════════════════

_GLOBAL_TIMESTEPS = 0
_PHASE1_THRESHOLD = 500000
_CURRENT_PHASE = 1

# ═══════════════════════════════════════════════════════════════
#  Gestion fork (compteur de种群 / agents vivants)
# ═══════════════════════════════════════════════════════════════
_FORK_COUNTS = {}   # agent_id -> fork_start_time

def set_global_timesteps(ts: int):
    """Appelé par le callback CurriculumCallback pour piloter la phase."""
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
    """Renvoie un reward si toutes les pierres du palier sont sur la case."""
    level = state.get("level", 1)
    if level >= MAX_LEVEL:
        return 0.0
    req = ELEVATION.get(level, {})
    vision = state.get("vision", [])
    if not vision:
        return 0.0
    cur_tile = vision[0] if vision else {}

    # nb joueurs requis présents sur la case
    players_here = cur_tile.get("player", 0)
    players_needed = req.get("players", 1)

    # toutes les pierres du palier présentes sur la case ?
    all_stones_ok = True
    for stone in STONES:
        needed = req.get(stone, 0)
        on_tile = cur_tile.get(stone, 0)
        if on_tile < needed:
            all_stones_ok = False
            break

    if players_here >= players_needed and all_stones_ok:
        return _PREINCANT_READY
    return 0.0


def _compute_rally_reward(action: str, call_dir: int | None, level: int) -> float:
    """Reward de rallyiement (v2). N'ACTIVE que si level >= 2.
    
    Logique :
      - call_dir == 0  → on est déjà sur la case émetteur → +8.0
      - action == optimal (rally_action(call_dir)) → +2.0
      - action est un mouvement mais pas l'optimal → -1.0 (pénalise l'erreur)
    
    Import local pour éviter dépendance circulaire avec elevation_guide.
    """
    if level < 2 or call_dir is None:
        return 0.0
    
    # 1) Arrivé sur place (direction 0)
    if call_dir == 0:
        return _RALLY_ARRIVED
    
    # 2) import local pour éviter circulaire
    # (elevation_guide n'importe pas rewards.py, donc pas de cycle)
    from src.elevation_guide import rally_action as _rally_action
    optimal = _rally_action(call_dir)
    
    if action == optimal:
        return _RALLY_FOLLOW_CLOSER
    
    # Pénalise les mouvements non-optimaux vers l'appelant
    # (Forward/Left/Right quand ce n'est pas le bon)
    if action in ("Forward", "Left", "Right"):
        return _RALLY_FOLLOW_FARTHER
    
    return 0.0


def compute_reward(prev: dict, state: dict, event: dict) -> float:
    if event is None:
        event = {}

    r = 0.0
    action = event.get("action", "")
    level = state.get("level", 1)
    food = state.get("inventory", {}).get("food", 0)
    phase = get_phase()

    # ── a) Coût du temps ──────────────────────────────────────
    if action == "Incantation":
        r += _INCANTATION_COST  # -0.1

    # ── b) Mort ───────────────────────────────────────────────
    if event.get("death"):
        # reset fork counter for this agent
        aid = event.get("agent_id", 0)
        _FORK_COUNTS.pop(aid, None)
        return r + (-10.0 if phase == 1 else -5.0)

    # ── c) Survie ─────────────────────────────────────────────
    r += _SURVIVAL_BONUS  # +0.1

    # ── d) Shaping nourriture ──────────────────────────────────
    prev_dist = prev.get("food_dist")
    cur_dist = state.get("food_dist")
    if cur_dist is not None and prev_dist is not None:
        if cur_dist < prev_dist:
            r += 0.3  # on se rapproche de la nourriture
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

    # Famine
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
                r += _STONE_SET_DEPOSIT  # dépôt utile pour incantation
            else:
                r -= 0.1

    if action.startswith(("Take ", "Set ")) and event.get("ok") is False:
        r -= 0.05

    # ── f) Fork dégressif ─────────────────────────────────────
    if action == "Fork" and event.get("ok"):
        if phase == 1:
            r += _FORK_PHASE1
        else:
            # penalite proportionnelle au temps depuis fork_start_time
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
    vision = state.get("vision", [])
    if vision:
        cur_tile = vision[0]
        players_here = cur_tile.get("player", 0)
        if players_here > 1:
            r += (players_here - 1) * _GROUP_REWARD_PER_PLAYER

    # ── i) Pré-incantation prête ──────────────────────────────
    r += _check_preincant_ready(state, event)

    # ── j) RALLIEMENT ACTIF v2 (NOUVEAU) ─────────────────────
    # Récupère la direction de l'appel join_incant (injecté par env.py)
    call_dir = event.get("incant_call_dir")
    r += _compute_rally_reward(action, call_dir, level)

    # ── k) Broadcast join_incant émis en niveau 2+ ───────────
    if action == "Broadcast join_incant" and event.get("ok") and level >= 2:
        # Vérifie qu'on a pas assez de joueurs (condition d'appel valide)
        vision = state.get("vision", [])
        if vision:
            players_here = vision[0].get("player", 0)
            needed_players = ELEVATION.get(level, {}).get("players", 1)
            if players_here < needed_players:
                r += _BROADCAST_CALL_VALID

    # ── l) Ancien reward de message (buggué v1) REMPLACÉ ─────
    # L'ancien bloc récompensait TOUT message contenant "incant"/"join".
    # Nouveau : récompense résiduelle MINIME uniquement si pas de call_dir
    # (cas où le message a été reçu mais parse_broadcast n'a pasmatché)
    if not event.get("incant_call_dir"):
        msg = event.get("message", "")
        if msg and ("join_incant" in msg.lower() or "ready" in msg.lower()):
            # reward très léger — l'agent est conscient qu'il y a un appel
            # mais le vrai signal vient de _compute_rally_reward
            r += 0.05

    return r
