"""Parsing et encodage du protocole client AI <-> serveur Zappy."""
from __future__ import annotations
import re

# Niveaux d'élévation : (nb_joueurs, linemate, deraumere, sibur, mendiane, phiras, thystame)
ELEVATION_REQUIREMENTS = {
    1: (1, 1, 0, 0, 0, 0, 0),
    2: (2, 1, 1, 1, 0, 0, 0),
    3: (2, 2, 0, 1, 0, 2, 0),
    4: (4, 1, 1, 2, 0, 1, 0),
    5: (4, 1, 2, 1, 3, 0, 0),
    6: (6, 1, 2, 3, 0, 1, 0),
    7: (6, 2, 2, 2, 2, 2, 1),
}

RESOURCES = ["food", "linemate", "deraumere", "sibur",
             "mendiane", "phiras", "thystame"]

# Une vie = 126 unités de temps par unité de food, départ a 10 food
FOOD_LIFE_UNITS = 126
MAX_LEVEL = 8


def parse_inventory(line: str) -> dict[str, int]:
    """Parse '[food 345, linemate 3, ...]' -> dict. Tolerant aux lignes invalides."""
    inv = {r: 0 for r in RESOURCES}
    content = line.strip().strip("[]")
    for item in content.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.rsplit(" ", 1)
        if len(parts) == 2 and parts[0] in inv and parts[1].lstrip("-").isdigit():
            inv[parts[0]] = int(parts[1])
    return inv



def parse_look(line: str) -> list[dict[str, int]]:
    """Parse '[player, food, thystame food,,]' -> liste de tuiles (compteurs)."""
    content = line.strip().strip("[]")
    tiles = []
    for tile in content.split(","):
        counts = {"player": 0, **{r: 0 for r in RESOURCES}}
        for tok in tile.strip().split():
            if tok in counts:
                counts[tok] += 1
        tiles.append(counts)
    return tiles


def parse_broadcast(line: str) -> tuple[int, str] | None:
    """Parse 'message K, text' -> (direction, text)."""
    m = re.match(r"message\s+(\d+),\s*(.*)", line.strip())
    if m:
        return int(m.group(1)), m.group(2)
    return None


def vision_tile_count(level: int) -> int:
    """Nombre de tuiles visibles : sum_{k=0}^{level} (2k+1)."""
    return sum(2 * k + 1 for k in range(level + 1))


# Broadcast à émettre : on encode des messages fixes de coordination.
# Le serveur attend "Broadcast <texte>". On limite à un vocabulaire
# restreint pour que l'agent apprenne à les utiliser.
BROADCAST_MESSAGES = [
    "join_incant",     # appel : venez sur ma case pour incanter
    "ready",           # je suis prêt / sur place
]

ACTIONS = [
    "Forward",
    "Right",
    "Left",
    "Look",
    "Inventory",
    "Connect_nbr",
    "Fork",
    "Eject",
    "Take food",
    "Take linemate",
    "Take deraumere",
    "Take sibur",
    "Take mendiane",
    "Take phiras",
    "Take thystame",
    "Set linemate",
    "Set deraumere",
    "Set sibur",
    "Set mendiane",
    "Set phiras",
    "Set thystame",
    "Incantation",
    "Broadcast join_incant",
    "Broadcast ready",
]


def encode_action(action_id: int) -> str:
    """ID discret -> commande serveur (sans \\n)."""
    return ACTIONS[action_id]
