"""train.py — Entrainement multi-agents avec curriculum learning + guidage d'élévation."""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from src.env import ZappyEnv, TeamState
from src import protocol
from src import elevation_guide
from src.rewards import set_global_timesteps, get_phase

# ── Logging : console + fichier (logs/train.log) ─────────────────────────────
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("zappy.train")
logger.setLevel(logging.INFO)
logger.propagate = False

_fmt = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

_console = logging.StreamHandler()
_console.setFormatter(_fmt)

_file = logging.FileHandler("logs/train.log", mode="a", encoding="utf-8")
_file.setFormatter(_fmt)

if not logger.handlers:
    logger.addHandler(_console)
    logger.addHandler(_file)

# Injecte la liste des actions dans le guide (évite l'import circulaire)
elevation_guide.set_actions_reference(protocol.ACTIONS)

CURRICULUM_PHASE1_STEPS = 100000   # steps avant phase 2
CURRICULUM_PHASE2_STEPS = 500000   # steps avant phase 3 (final)
MAX_TIMESTEPS = 1000000


class CurriculumCallback(BaseCallback):
    """Callback qui synchronise la phase dans rewards.py ET log l'avancement."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.phase = 1
        self.phase1_end = CURRICULUM_PHASE1_STEPS
        self.phase2_end = CURRICULUM_PHASE2_STEPS

    def _on_step(self) -> bool:
        ts = self.num_timesteps
        set_global_timesteps(ts)

        if ts < self.phase1_end:
            new_phase = 1
        elif ts < self.phase2_end:
            new_phase = 2
        else:
            new_phase = 3

        if new_phase != self.phase:
            old = self.phase
            self.phase = new_phase
            if self.verbose:
                logger.info("Phase %d → %d @ %d steps", old, new_phase, ts)
            if new_phase == 2:
                logger.info(
                    "PHASE_TRANSITION phase1→phase2 | timesteps=%d | "
                    "server_config: small_map_high_food", ts
                )
            elif new_phase == 3:
                logger.info(
                    "PHASE_TRANSITION phase2→phase3 | timesteps=%d | "
                    "server_config: normal_map_normal_food", ts
                )
        return True


class ZappyCallback(BaseCallback):
    """Logging des métriques toutes les N secondes."""

    def __init__(self, interval_sec=30, verbose=0):
        super().__init__(verbose)
        self.interval_sec = interval_sec
        self.last_time = time.time()
        self.rewards = []

    def _on_step(self) -> bool:
        rews = self.locals.get("rewards")
        if rews is not None:
            self.rewards.extend(np.asarray(rews).tolist())

        now = time.time()
        if now - self.last_time >= self.interval_sec:
            mean_r = float(np.mean(self.rewards)) if self.rewards else 0.0
            logger.info(
                "steps=%d | mean_reward=%.3f | n=%d",
                self.num_timesteps, mean_r, len(self.rewards)
            )
            self.rewards.clear()
            self.last_time = now
        return True


def make_env(rank, host, port, team, client_id, team_state):
    """Factory pour SubprocVecEnv."""
    def _init():
        return ZappyEnv(
            host=host,
            port=port,
            team=team,
            timeout=10.0,
            max_steps=5000,
            agent_id=client_id + rank,
            team_state=team_state,
        )
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=4242)
    parser.add_argument("--team", default="ia")
    parser.add_argument("--clients", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--win_count", type=int, default=6)

    args = parser.parse_args()

    logger.info(f"Démarrage entraînement multi-agents : {vars(args)}")

    team_state = TeamState(win_count=args.win_count)

    env_fns = [
        make_env(rank, args.host, args.port, args.team, 0, team_state)
        for rank in range(args.clients)
    ]
    vec_env = SubprocVecEnv(env_fns)

    model_path = Path("models/zappy_ppo")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and Path(f"{args.resume}.zip").exists():
        logger.info(f"Reprise depuis {args.resume}.zip")
        model = PPO.load(args.resume, env=vec_env)
    else:
        from src.agent import build_model
        model = build_model(vec_env)

    from src.agent import checkpoint_callback
    callbacks = [
        CurriculumCallback(verbose=1),
        ZappyCallback(interval_sec=30, verbose=1),
        checkpoint_callback(freq=10000),
    ]

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=False,
        )
        logger.info("✅ Entraînement terminé avec succès")
    except KeyboardInterrupt:
        logger.info("⏹️  Entraînement interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur durant entraînement : {e}")
        raise
    finally:
        model.save(str(model_path))
        logger.info(f"Modèle sauvegardé : {model_path}.zip")
        vec_env.close()


if __name__ == "__main__":
    main()
