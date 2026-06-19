"""Wrapper PPO (stable-baselines3) avec sauvegarde des checkpoints."""
from __future__ import annotations
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

DATA_DIR = "data"


def build_model(env, tensorboard_log="logs/tb"):
    """Construit un PPO avec une politique MLP adaptée a notre observation."""
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,          # exploration : utile car récompenses creuses
        verbose=1,
        tensorboard_log=tensorboard_log,
    )


def checkpoint_callback(freq=10000):
    os.makedirs(DATA_DIR, exist_ok=True)
    return CheckpointCallback(
        save_freq=freq,
        save_path=os.path.join(DATA_DIR, "checkpoints"),
        name_prefix="zappy_ppo",
    )
