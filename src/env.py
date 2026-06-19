"""Environnement Gymnasium connectant l'agent au serveur Zappy via TCP."""
from __future__ import annotations
import socket
import time
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src import protocol
from src.rewards import compute_reward

logger = logging.getLogger("zappy.env")


class ZappyEnv(gym.Env):
    """Un agent = un drone connecté à une équipe."""

    metadata = {"render_modes": []}

    def __init__(self, host="localhost", port=4242, team="ia",
                 timeout=10.0, max_steps=5000,
                 agent_id=0, team_state=None):
        super().__init__()
        self.host = host
        self.port = port
        self.team = team
        self.timeout = timeout
        self.max_steps = max_steps
        self.agent_id = agent_id
        self.team_state = team_state

        self.sock: socket.socket | None = None
        self.buffer = ""
        self.world = (0, 0)
        self.steps = 0
        self.state = self._empty_state()

        self.action_space = spaces.Discrete(len(protocol.ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(24,), dtype=np.float32)

    # ----- gestion socket -----
    def _open_socket(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        self.buffer = ""

    def _close_socket(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self.buffer = ""

    def _send(self, cmd: str):
        """Envoie une commande (ajoute le \\n)."""
        self.sock.sendall((cmd + "\n").encode())

    def _readline(self) -> str:
        """Lit une ligne complete depuis le buffer (bloquant)."""
        while "\n" not in self.buffer:
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("Socket fermée par le serveur")
            self.buffer += data.decode(errors="ignore")
        line, self.buffer = self.buffer.split("\n", 1)
        return line

    def _expect(self, expected: str):
        """Lit une ligne et verifie qu'elle correspond a 'expected'."""
        line = self._readline().strip()
        if line != expected:
            raise ConnectionError(
                "Attendu '%s', recu '%s'" % (expected, line)
            )

    def _connect(self):
        """Connexion handshake. Retry tant que slots epuises (slots <= 0)."""
        max_retries = 50
        for attempt in range(max_retries):
            try:
                self._open_socket()
                self._expect("WELCOME")          # 1) le serveur salue
                self._send(self.team)            # 2) on annonce l'equipe
                slots = self._readline().strip()  # 3) nb de slots libres

                if not slots.lstrip("-").isdigit() or int(slots) <= 0:
                    self._close_socket()
                    time.sleep(0.3)
                    continue                     # sature/parasite -> retry

                self.client_slots = int(slots)
                dims = self._readline().strip()   # 4) "X Y"
                parts = dims.split()
                if len(parts) == 2 and all(p.isdigit() for p in parts):
                    self.world = (int(parts[0]), int(parts[1]))
                return
            except (ConnectionError, OSError):
                self._close_socket()
                time.sleep(0.3)
        raise ConnectionError(
            "Impossible de se connecter apres %d essais" % max_retries
        )

    def _reconnect_internal(self):
        """Rejoint le serveur en reutilisant notre propre slot fraichement libere.
        
        Comportement : try _connect() qui a 50 essais. Si saturation,
        l'exception remonte et l'episode termine (reset() retentera).
        """
        self._connect()
        self._set_level(1)
        self._refresh_inventory()
        self._refresh_vision()

    def _send_and_wait(self, cmd: str) -> str:
        self._send(cmd)
        while True:
            line = self._readline().strip()
            if line == "dead":
                self.state["alive"] = False
                return "dead"
            if line.startswith("message") or line.startswith("eject"):
                continue
            return line

    def _read_next(self) -> str:
        """Lit la prochaine ligne serveur SANS rien emettre (filtre broadcasts)."""
        while True:
            line = self._readline().strip()
            if line.startswith("message") or line.startswith("eject"):
                continue
            return line

    # ----- état -----
    @staticmethod
    def _empty_state():
        return {
            "level": 1,
            "inventory": {r: 0 for r in protocol.RESOURCES},
            "vision": [],
            "alive": True,
        }

    # ----- helpers vision -----
    def _food_distance(self) -> float | None:
        """Distance a la nourriture visible la plus proche.

        parse_look retourne list[dict] avec cle 'food' pour chaque tuile.
        Les tuiles sont en ordre radial :
          tile 0 = case actuelle (distance 0)
          tiles 1+ = anneaux externes...

        Retourne l'index (float) de la premiere tuile avec food > 0,
        ou None si aucune food visible.
        """
        vision = self.state["vision"]
        if not vision:
            return None
        for idx, tile in enumerate(vision):
            if tile.get("food", 0) > 0:
                return float(idx)
        return None

    def _refresh_inventory(self):
        line = self._send_and_wait("Inventory")
        if line not in ("dead", "ko", "") and line.startswith("["):
            self.state["inventory"] = protocol.parse_inventory(line)

    def _refresh_vision(self):
        line = self._send_and_wait("Look")
        if line != "dead":
            self.state["vision"] = protocol.parse_look(line)
            self.state["food_dist"] = self._food_distance()

    def _set_level(self, level: int):
        self.state["level"] = level
        if self.team_state is not None:
            self.team_state.update(self.agent_id, level)

    def _build_obs(self) -> np.ndarray:
        inv = self.state["inventory"]
        obs = np.zeros(24, dtype=np.float32)
        for i, r in enumerate(protocol.RESOURCES):
            obs[i] = min(inv[r] / 20.0, 1.0)
        obs[7] = self.state["level"] / protocol.MAX_LEVEL
        obs[8] = min(inv["food"] / 100.0, 1.0)
        vision = self.state["vision"]
        for offset, tile_idx in ((9, 0), (16, 1)):
            if tile_idx < len(vision):
                tile = vision[tile_idx]
                for i, r in enumerate(protocol.RESOURCES):
                    obs[offset + i] = min(tile[r] / 5.0, 1.0)
        if self.team_state is not None:
            obs[23] = self.team_state.max_level_count() / self.team_state.win_count
        return obs

    # ----- API gym -----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        was_dead = not self.state["alive"]
        self.state = self._empty_state()
        if was_dead:
            self._close_socket()
        if self.sock is None:
            self._connect()
        self._set_level(1)
        self._refresh_inventory()
        self._refresh_vision()
        return self._build_obs(), {}

    def step(self, action: int):
        self.steps += 1

        if not isinstance(action, (int, np.integer)) or not (0 <= int(action) < len(protocol.ACTIONS)):
            return self._build_obs(), -1.0, False, False, {"invalid": True}

        cmd = protocol.encode_action(int(action))
        prev = {
            "level": self.state["level"],
            "inventory": dict(self.state["inventory"]),
            "alive": self.state["alive"],
            "food_dist": self.state.get("food_dist"),
        }
        event = {"action": cmd}

        try:
            response = self._send_and_wait(cmd)

            if response == "dead":
                self.state["alive"] = False
                event["death"] = True
                reward = compute_reward(prev, self.state, event)
                logger.info("Agent %s MORT (reward=%.1f)", self.agent_id, reward)
                try:
                    self._reconnect_internal()
                    return self._build_obs(), reward, False, False, event
                except ConnectionError as exc:
                    logger.warning("Reconnexion impossible apres mort (agent %s): %s", 
                                   self.agent_id, exc)
                    self._close_socket()
                    return self._build_obs(), reward, True, False, event

            if cmd.startswith("Take"):
                event.update({"ok": response == "ok"})
                self._refresh_inventory()
            elif cmd.startswith("Set"):
                event.update({"ok": response == "ok"})
                self._refresh_inventory()
            elif cmd == "Inventory":
                self.state["inventory"] = protocol.parse_inventory(response)
            elif cmd == "Look":
                self.state["vision"] = protocol.parse_look(response)
                self.state["food_dist"] = self._food_distance()
            elif cmd == "Connect_nbr":
                event.update({"type": "connect_nbr",
                              "slots": int(response) if response.isdigit() else 0})
            elif cmd == "Fork":
                event.update({"type": "fork", "ok": response == "ok"})
            elif cmd == "Eject":
                event.update({"type": "eject", "ok": response == "ok"})
            elif cmd == "Incantation":
                old_level = self.state["level"]
                if response.startswith("Elevation"):
                    final = self._read_next()
                    m = final.split(":")[-1].strip() if ":" in final else None
                    if m and m.isdigit():
                        self._set_level(int(m))
                        event.update({"level_up": True, "old_level": old_level})
                        if self.state["level"] >= protocol.MAX_LEVEL:
                            event["win"] = True
                    else:
                        event.update({"ko": True})
                else:
                    event.update({"ko": True})

        except (BrokenPipeError, ConnectionError, OSError) as exc:
            logger.warning("Connexion perdue (agent %s): %s", self.agent_id, exc)
            self.state["alive"] = False
            event["death"] = True
            reward = compute_reward(prev, self.state, event)
            self._close_socket()
            try:
                self._reconnect_internal()
            except ConnectionError:
                logger.warning("Reconnexion impossible apres loss (agent %s)", 
                               self.agent_id)
            return self._build_obs(), reward, False, False, event

        reward = compute_reward(prev, self.state, event)
        team_won = self.team_state is not None and self.team_state.is_victory()
        terminated = (not self.state["alive"]) or team_won
        truncated = self.steps >= self.max_steps
        return self._build_obs(), reward, terminated, truncated, event

    def close(self):
        self._close_socket()
