#!/usr/bin/env python3
from __future__ import annotations

import socket
import time
from collections import deque
from typing import Deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

RESOURCE_NAMES = ("food", "linemate", "deraumere", "sibur", "mendiane", "phiras", "thystame")
STONE_NAMES = RESOURCE_NAMES[1:]  # toutes les ressources sauf food
MAX_LEVEL = 8

# Table de la "rite d'elevation" (page 5 du sujet) : pour passer du niveau
# courant au niveau superieur, il faut reunir sur la meme case ce nombre
# de joueurs de meme niveau ainsi que ces quantites de pierres.
ELEVATION_REQUIREMENTS = {
	1: {"players": 1, "linemate": 1, "deraumere": 0, "sibur": 0, "mendiane": 0, "phiras": 0, "thystame": 0},
	2: {"players": 2, "linemate": 1, "deraumere": 1, "sibur": 1, "mendiane": 0, "phiras": 0, "thystame": 0},
	3: {"players": 2, "linemate": 2, "deraumere": 0, "sibur": 1, "mendiane": 0, "phiras": 2, "thystame": 0},
	4: {"players": 4, "linemate": 1, "deraumere": 1, "sibur": 2, "mendiane": 0, "phiras": 1, "thystame": 0},
	5: {"players": 4, "linemate": 1, "deraumere": 2, "sibur": 1, "mendiane": 3, "phiras": 0, "thystame": 0},
	6: {"players": 6, "linemate": 1, "deraumere": 2, "sibur": 3, "mendiane": 0, "phiras": 1, "thystame": 0},
	7: {"players": 6, "linemate": 2, "deraumere": 2, "sibur": 2, "mendiane": 2, "phiras": 2, "thystame": 1},
}

# Un unite de food permet de survivre 126 unites de temps (page 3 du sujet).
# On considere qu'un agent est "en securite alimentaire" au-dela de cette
# reserve, et "en danger" en-dessous d'un seuil plus bas.
FOOD_SAFE_THRESHOLD = 126
FOOD_DANGER_THRESHOLD = 30


class ZappyClient:
	def __init__(self, host: str, port: int, team_name: str, timeout: float = 10.0) -> None:
		self.host = host
		self.port = port
		self.team_name = team_name
		self.timeout = timeout
		self.socket: socket.socket | None = None
		self._reader = None
		self._writer = None
		self._pending: Deque[str] = deque()

	def connect(self) -> int:
		self.close()
		self.socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
		self.socket.settimeout(self.timeout)
		self._reader = self.socket.makefile("r", encoding="utf-8", newline="\n")
		self._writer = self.socket.makefile("w", encoding="utf-8", newline="\n")
		greeting = self._readline()
		if greeting != "WELCOME":
			raise RuntimeError(f"Unexpected greeting: {greeting!r}")
		self._writeline(self.team_name)
		slots_line = self._readline()
		if slots_line == "ko":
			raise RuntimeError(f"No slot for team {self.team_name!r}")
		slots = int(slots_line)
		self._readline()  # ignore "15 10" map dimensions

		# La connexion consomme un oeuf/slot pour l'equipe. Le protocole Zappy
		# permet de reconstituer un slot via "Fork" (pond un nouvel oeuf).
		# Sans ca, le pool d'oeufs (-c N) s'epuise apres N connexions/morts
		# et toute reconnexion future echoue avec "No slot for team ...".
		try:
			self._writeline("Fork")
			self._readline()  # reponse attendue: "ok"
		except (RuntimeError, OSError, TimeoutError):
			pass

		return slots

	def close(self) -> None:
		for h in (self._reader, self._writer):
			if h:
				try: h.close()
				except: pass
		if self.socket:
			try: self.socket.close()
			except: pass
		self.socket = self._reader = self._writer = None
		self._pending.clear()

	def _writeline(self, text: str) -> None:
		self._writer.write(text + "\n")
		self._writer.flush()

	def _readline(self) -> str:
		while True:
			line = self._reader.readline()
			if line == "":
				raise RuntimeError("Connection closed by server")
			line = line.rstrip("\r\n")
			if not line:
				continue
			if line.startswith("message "):
				self._pending.append(line)
				continue
			return line

	def request(self, cmd: str) -> str:
		self._writeline(cmd)
		resp = self._readline()
		if resp == "Elevation underway":
			return resp + "\n" + self._readline()
		return resp

	def pop_messages(self) -> list[str]:
		"""Renvoie et vide la file des 'message K, text' recus (page 13)."""
		msgs = list(self._pending)
		self._pending.clear()
		return msgs


class ZappyEnv(gym.Env):
	metadata = {"render_modes": ["none"]}

	# Actions : on peut maintenant Take/Set chaque ressource du jeu (page 12 :
	# "Take <object>" / "Set <object>"), et plus seulement food/linemate.
	# "Broadcast" permet aux joueurs d'une meme equipe (entraines avec une
	# politique partagee) de se signaler mutuellement quand ils sont prets
	# pour une elevation de groupe (pages 6-7 et 12-13 : sound transmission).
	# Index :
	#  0=Forward 1=Left 2=Right 3=Look 4=Inventory
	#  5=Take food 6=Take linemate 7=Take deraumere 8=Take sibur
	#  9=Take mendiane 10=Take phiras 11=Take thystame
	#  12=Set linemate 13=Set deraumere 14=Set sibur
	#  15=Set mendiane 16=Set phiras 17=Set thystame
	#  18=Incantation 19=Broadcast
	ACTIONS = (
		"Forward", "Left", "Right", "Look", "Inventory",
		"Take food", "Take linemate", "Take deraumere", "Take sibur",
		"Take mendiane", "Take phiras", "Take thystame",
		"Set linemate", "Set deraumere", "Set sibur",
		"Set mendiane", "Set phiras", "Set thystame",
		"Incantation", "Broadcast",
	)

	def __init__(self, host="127.0.0.1", port=3000, team_name="Br",
	             timeout=10.0, max_steps=2000, **kwargs):
		super().__init__()
		# **kwargs absorbe les options non utilisees ici (ex: "render_mode"
		# envoye par eval.py via _build_env_config) pour eviter un
		# TypeError a l'instanciation.
		self.client = ZappyClient(host, port, team_name, timeout)
		self.team_name = team_name
		self.max_steps = max_steps

		# obs: inventory(7) + tile(7) + level(1) + step_ratio(1)
		#      + players_on_tile(1) + players_needed_next_level(1)
		#      + broadcast: heard(1) + direction(1) + announced_level(1) = 21
		# Le canal broadcast donne a la politique partagee un moyen de
		# percevoir "un coequipier signale qu'il est pret pour une elevation
		# de groupe" (pages 6-7, 12-13), pour aider les joueurs a converger
		# vers la meme case.
		self.observation_space = spaces.Box(0.0, 1.0, shape=(21,), dtype=np.float32)
		self.action_space = spaces.Discrete(len(self.ACTIONS))

		self._step = 0
		self._level = 1
		self._inventory = np.zeros(7, dtype=np.float32)
		self._tile = np.zeros(7, dtype=np.float32)
		self._players_on_tile = 0.0
		# (heard_recently, direction_norm, announced_level_norm)
		self._last_broadcast = (0.0, 0.0, 0.0)

	def reset(self, *, seed=None, options=None):
		super().reset(seed=seed)
		self._step = 0
		self._level = 1
		self._players_on_tile = 0.0
		self._last_broadcast = (0.0, 0.0, 0.0)

		if self.client.socket is None:
			self._connect_with_retry(max_retries=10, delay=2.0)
		else:
			try:
				self.client.socket.getpeername()
			except OSError:
				self.client.close()
				self._connect_with_retry(max_retries=10, delay=2.0)

		try:
			self._inventory = self._get_inventory()
			self._tile = self._get_tile()
		except (RuntimeError, OSError, TimeoutError):
			# Connexion perdue juste apres connect -> retry complet
			self.client.close()
			self._connect_with_retry(max_retries=10, delay=2.0)
			self._inventory = self._get_inventory()
			self._tile = self._get_tile()
		return self._obs(), {}

	def _connect_with_retry(self, max_retries: int, delay: float) -> None:
		for attempt in range(max_retries):
			try:
				self.client.connect()
				return
			except Exception as exc:
				if attempt >= max_retries - 1:
					raise
				print(f"[{self.team_name}] Retry {attempt+1}/{max_retries} ({exc})...")
				time.sleep(delay)
				self.client.close()

	def step(self, action: int):
		self._step += 1
		cmd = self.ACTIONS[int(action) % len(self.ACTIONS)]
		prev_level = self._level
		# Cout du temps : chaque commande "simple" coute 7/f, l'incantation
		# en coute 300/f (~43x plus). On reflete ca par un cout proportionnel
		# au temps reellement consomme par l'action (cf. page 10 du sujet).
		action_time_units = 300 if cmd == "Incantation" else 7
		reward = -0.01 * (action_time_units / 7.0)
		terminated = False
		truncated = self._step >= self.max_steps
		response = ""

		try:
			# "Broadcast" annonce le niveau courant du joueur (format "LVL<k>"),
			# pour que les coequipiers puissent reperer un partenaire pret pour
			# une elevation de groupe (pages 6-7, 12-13).
			wire_cmd = f"Broadcast LVL{self._level}" if cmd == "Broadcast" else cmd
			response = self.client.request(wire_cmd)

			if response == "dead":
				# Mourir signifie etre tombe a 0 unite de nourriture :
				# c'est l'echec ultime de la gestion de survie (page 3).
				reward = -10.0
				terminated = True

			elif cmd == "Incantation":
				if "Current level:" in response:
					new_level = self._parse_level(response)
					if new_level > prev_level:
						# Recompense proportionnelle a la difficulte reelle du
						# rituel (nb de joueurs + pierres requises, page 5) :
						# plus le niveau est haut, plus le rituel est exigeant
						# et donc plus il rapporte.
						req = ELEVATION_REQUIREMENTS.get(prev_level, {})
						difficulty = sum(req.values())
						reward += 10.0 + 5.0 * difficulty * (new_level - prev_level)
						self._level = new_level
						if self._level >= MAX_LEVEL:
							# Objectif final du sujet : un joueur au niveau
							# max (la victoire d'equipe demande 6 joueurs a
							# ce niveau).
							reward += 100.0
							terminated = True
				else:
					# "ko" : conditions non remplies. Le rituel a quand meme
					# coute 300/f de temps bloque pour rien (page 5).
					reward -= 2.0

			elif response == "ko" and (cmd.startswith("Take ") or cmd.startswith("Set ")):
				# Rien a ramasser / rien dans l'inventaire a poser : action gaspillee.
				reward -= 0.05

			elif cmd == "Broadcast" and response == "ok":
				# Diffuser son niveau n'est vraiment utile que si l'elevation
				# suivante necessite plusieurs joueurs ET que les pierres sont
				# deja au sol sur la case : on "appelle a l'aide" au bon
				# moment (page 5 : conditions de groupe + pierres requises).
				req = ELEVATION_REQUIREMENTS.get(self._level, {})
				stones_ready = all(
					self._tile[RESOURCE_NAMES.index(s)] >= req.get(s, 0)
					for s in STONE_NAMES
				)
				if req.get("players", 1) > 1 and stones_ready:
					reward += 0.5

			if not terminated:
				try:
					self._inventory = self._get_inventory()
					self._tile = self._get_tile()
				except (RuntimeError, OSError, TimeoutError):
					# connexion perdue apres l action -> termine proprement
					reward = -5.0
					terminated = True
					self.client.close()
				else:
					food_idx = RESOURCE_NAMES.index("food")
					food_level = self._inventory[food_idx]
					req = ELEVATION_REQUIREMENTS.get(self._level, {})

					if cmd == "Take food" and response == "ok":
						# Ramasser de la nourriture est d'autant plus utile
						# que la reserve de survie (126 unites = 126 cycles,
						# page 3) est basse.
						if food_level < FOOD_DANGER_THRESHOLD:
							reward += 2.0
						elif food_level < FOOD_SAFE_THRESHOLD:
							reward += 0.5
						else:
							reward += 0.05  # surplus, peu utile

					elif cmd.startswith("Take ") and response == "ok":
						stone = cmd.split(" ", 1)[1]
						if stone in STONE_NAMES:
							needed = req.get(stone, 0)
							have = self._inventory[RESOURCE_NAMES.index(stone)]
							if needed > 0 and have <= needed:
								# Cette pierre rapproche directement de la
								# prochaine elevation (table page 5).
								reward += 1.5
							elif needed > 0:
								# Stock deja suffisant pour la prochaine elevation.
								reward += 0.1
							else:
								# Pierre non requise au niveau actuel.
								reward += 0.05

					elif cmd.startswith("Set ") and response == "ok":
						stone = cmd.split(" ", 1)[1]
						if stone in STONE_NAMES:
							needed = req.get(stone, 0)
							on_tile = self._tile[RESOURCE_NAMES.index(stone)]
							if needed > 0 and on_tile <= needed:
								# Pose une pierre encore manquante sur la case
								# d'incantation : progres reel vers le rituel
								# (page 5 : les pierres doivent etre AU SOL).
								reward += 2.0
							elif needed > 0:
								# La case a deja assez de cette pierre : pose
								# superflue, la pierre est "perdue" pour rien.
								reward -= 0.2
							else:
								# Pierre non requise au niveau actuel : posee
								# inutilement.
								reward -= 0.1

					# Petit malus si la nourriture descend sous le seuil de
					# danger : pousse l'agent a anticiper sa survie plutot
					# que de mourir brutalement.
					if food_level < FOOD_DANGER_THRESHOLD:
						reward -= 0.1

		except (RuntimeError, OSError, TimeoutError):
			reward = -5.0
			terminated = True
			self.client.close()

		if not terminated:
			# Met a jour le canal de broadcast a partir des "message K, text"
			# accumules pendant les requetes ci-dessus (page 13). On ne garde
			# que le plus recent pour rester simple.
			self._last_broadcast = (0.0, 0.0, 0.0)
			for line in self.client.pop_messages():
				parsed = self._parse_message(line)
				if parsed is not None:
					self._last_broadcast = parsed
		else:
			self._last_broadcast = (0.0, 0.0, 0.0)

		return self._obs(), reward, terminated, truncated, {"level": self._level, "response": response}

	def _obs(self) -> np.ndarray:
		inv_norm = np.clip(self._inventory / np.array([200, 50, 50, 50, 50, 50, 50], dtype=np.float32), 0, 1)
		tile_norm = np.clip(self._tile / np.array([200, 50, 50, 50, 50, 50, 50], dtype=np.float32), 0, 1)
		level = np.array([self._level / MAX_LEVEL], dtype=np.float32)
		step_ratio = np.array([self._step / self.max_steps], dtype=np.float32)
		# Nombre de joueurs sur la case courante (soi-meme inclus), normalise
		# par le maximum requis dans la table d'elevation (6, pour 6->7/7->8).
		players_here = np.array([min(self._players_on_tile / 6.0, 1.0)], dtype=np.float32)
		# Nombre de joueurs requis pour la prochaine elevation (table page 5).
		req = ELEVATION_REQUIREMENTS.get(self._level, {})
		players_needed = np.array([req.get("players", 1) / 6.0], dtype=np.float32)
		# Canal de coordination (page 6-7, 12-13) : un coequipier a-t-il
		# diffuse recemment son niveau, et dans quelle direction / a quel
		# niveau ? Permet a la politique partagee de converger vers les
		# joueurs prets pour une elevation de groupe.
		broadcast = np.array(self._last_broadcast, dtype=np.float32)
		return np.concatenate(
			[inv_norm, tile_norm, level, step_ratio, players_here, players_needed, broadcast]
		).astype(np.float32)

	def _get_inventory(self) -> np.ndarray:
		return self._parse_resources(self.client.request("Inventory"))

	def _get_tile(self) -> np.ndarray:
		resp = self.client.request("Look")
		content_all = resp.strip().strip("[]")
		tile0 = content_all.split(",")[0] if content_all else ""
		# La case 0 (page 6) liste toujours "player" pour soi-meme, et un
		# "player" supplementaire par autre joueur partageant la case. C'est
		# le seul moyen, via le protocole, de savoir si on est en groupe -
		# une condition necessaire pour les elevations 2->8 (page 5).
		self._players_on_tile = float(tile0.strip().split().count("player"))
		return self._parse_resources(f"[{tile0}]")

	def _parse_resources(self, resp: str) -> np.ndarray:
		out = np.zeros(7, dtype=np.float32)
		content = resp.strip().strip("[]")
		for part in content.split(","):
			tokens = part.strip().split()
			if len(tokens) >= 2 and tokens[0] in RESOURCE_NAMES:
				try:
					out[RESOURCE_NAMES.index(tokens[0])] = float(tokens[1])
				except ValueError:
					pass
		return out

	def _parse_level(self, resp: str) -> int:
		for line in resp.splitlines():
			if line.startswith("Current level:"):
				try:
					return int(line.split(":")[1].strip())
				except ValueError:
					pass
		return self._level

	def _parse_message(self, line: str) -> tuple[float, float, float] | None:
		"""Parse une ligne 'message K, LVL<k>' (page 13) -> (heard, dir, lvl).

		K est la direction du tile d'origine (0 = sur soi, page 7).
		On normalise K par 32 (vision max raisonnable) et le niveau par
		MAX_LEVEL. En cas d'echec de parsing, renvoie None (ligne ignoree).
		"""
		if not line.startswith("message "):
			return None
		try:
			head, _, text = line.partition(",")
			direction = int(head.split()[1])
			text = text.strip()
			level = 0.0
			if text.startswith("LVL"):
				level = float(int(text[3:])) / MAX_LEVEL
			direction_norm = min(abs(direction) / 32.0, 1.0)
			return (1.0, direction_norm, level)
		except (ValueError, IndexError):
			return None

	def close(self):
		self.client.close()
