#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from zappy_env import ZappyEnv


def load_config(path: str) -> dict:
	with open(path, encoding="utf-8") as f:
		return yaml.safe_load(f) or {}


class ProgressCallback(BaseCallback):
	def __init__(self, total_steps: int, log_every: int = 10000):
		super().__init__()
		self.total = total_steps
		self.log_every = log_every
		self.next_log = log_every
		self.t0 = 0.0
		self.rewards: list[float] = []
		self.levels: list[int] = []

	def _on_training_start(self):
		self.t0 = time.time()
		print(f"Training started: {self.total} steps")

	def _on_step(self) -> bool:
		for info in self.locals.get("infos", []):
			ep = info.get("episode")
			if ep:
				self.rewards.append(float(ep["r"]))
			if "level" in info:
				self.levels.append(int(info["level"]))

		if self.num_timesteps >= self.next_log:
			elapsed = max(time.time() - self.t0, 1e-9)
			fps = self.num_timesteps / elapsed
			pct = 100 * self.num_timesteps / self.total
			avg_r = sum(self.rewards[-20:]) / max(len(self.rewards[-20:]), 1)
			max_lvl = max(self.levels[-100:]) if self.levels else 1
			print(f"  {self.num_timesteps}/{self.total} ({pct:.0f}%) | fps={fps:.0f} | avg_reward={avg_r:.2f} | max_level={max_lvl}")
			self.next_log += self.log_every
		return True


def save_graph(rewards: list, path: str):
	if not rewards:
		return
	Path(path).parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(10, 4))
	plt.plot(rewards, linewidth=1)
	plt.title("Reward per episode")
	plt.xlabel("Episode")
	plt.ylabel("Reward")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(path)
	plt.close()
	print(f"Graph saved: {path}")


def train(config_path: str):
	cfg = load_config(config_path)
	algo = cfg.get("algo", {})

	env_kwargs = {
		"host": cfg.get("host", "127.0.0.1"),
		"port": int(cfg.get("port", 3000)),
		"team_name": cfg.get("team_name", "Br"),
		"timeout": float(cfg.get("timeout_seconds", 10.0)),
		"max_steps": int(cfg.get("max_steps_per_episode", 2000)),
	}

	total_steps = int(cfg.get("total_timesteps", 500000))
	model_path = cfg.get("model_path", "models/zappy_ppo.zip")
	graph_path = cfg.get("graph_path", "data/rewards.png")

	# Multi-agent par "parameter sharing" (page 9 : un client = un joueur).
	# n_players connexions simultanees a la meme equipe -> n_players joueurs
	# presents sur la carte, tous pilotes par UNE seule politique PPO.
	# C'est ce qui permet de reunir plusieurs joueurs de meme niveau sur une
	# case pour les elevations de groupe (table page 5).
	n_players = max(1, int(cfg.get("n_players", 1)))
	print(f"Multi-agent: {n_players} joueur(s) simultane(s) pour l'equipe '{env_kwargs['team_name']}'")

	vec_env = make_vec_env(ZappyEnv, n_envs=n_players, seed=cfg.get("seed", 42), env_kwargs=env_kwargs)

	# Charger modele existant ou en créer un nouveau
	if Path(model_path).exists():
		print(f"Resuming from {model_path}")
		model = PPO.load(model_path, env=vec_env)
	else:
		model = PPO(
			"MlpPolicy", vec_env,
			verbose=1,
			seed=cfg.get("seed", 42),
			learning_rate=float(algo.get("learning_rate", 3e-4)),
			n_steps=int(algo.get("n_steps", 2048)),
			batch_size=int(algo.get("batch_size", 64)),
			gamma=float(algo.get("gamma", 0.99)),
			gae_lambda=float(algo.get("gae_lambda", 0.95)),
			ent_coef=float(algo.get("ent_coef", 0.01)),
			n_epochs=int(algo.get("n_epochs", 10)),
		)

	cb = ProgressCallback(total_steps, log_every=512)
	model.learn(total_timesteps=total_steps, callback=cb, reset_num_timesteps=False)

	Path(model_path).parent.mkdir(parents=True, exist_ok=True)
	model.save(model_path)
	print(f"Model saved: {model_path}")
	save_graph(cb.rewards, graph_path)
	print("Training complete")
	vec_env.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", default="configs/agent_br.yaml")
	args = parser.parse_args()
	train(args.config)
