"""Etat partagé de l'équipe : suivi des niveaux et condition de victoire."""
from __future__ import annotations
import threading
from src.protocol import MAX_LEVEL


class TeamState:
    """Partagé entre tous les agents pour détecter la victoire globale."""

    def __init__(self, nb_clients: int, win_count: int = 6):
        self.nb_clients = nb_clients
        self.win_count = win_count          # 6 joueurs niveau max = victoire
        self.levels: dict[int, int] = {}    # agent_id -> niveau courant
        self._lock = threading.Lock()
        self.victory = False

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def update(self, agent_id: int, level: int):
        with self._lock:
            self.levels[agent_id] = level
            at_max = sum(1 for lv in self.levels.values() if lv >= MAX_LEVEL)
            if at_max >= self.win_count:
                self.victory = True

    def is_victory(self) -> bool:
        with self._lock:
            return self.victory

    def max_level_count(self) -> int:
        with self._lock:
            return sum(1 for lv in self.levels.values() if lv >= MAX_LEVEL)

    def snapshot(self) -> dict[int, int]:
        with self._lock:
            return dict(self.levels)
