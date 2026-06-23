"""train.py — Entrainement multi-agents avec curriculum learning."""
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
from src.rewards import set_global_timesteps, get_phase

# CHANGELOG v2 : initialiser elevation_guide AVANT la création des envs
# pour que le cache d'actions soit disponible dès le premier step.
# L'appel dans env.__init__ est un filet de sécurité (idempotent).
try:
    from src.elevation_guide import set_actions_reference
    set_actions_reference(protocol.ACTIONS)
except ImportError:
    pass  # elevation_guide optionnel en dehors de l'arbre src/

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("zappy.train")

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
        # Met à jour le module rewards (état global)
        set_global_timesteps(ts)

        # Détermine la phase
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
            # Log fort pour analyse post-entrainement
            if new_phase == 2:
                logger.info(
                    "PHASE_TRANSITION phase1→phase2 | timesteps=%d | "
                    "server_config: small_map_high_food",
                    ts
                )
            elif new_phase == 3:
                logger.info(
                    "PHASE_TRANSITION phase2→phase3 | timesteps=%d | "
                    "server_config: normal_map_normal_food",
                    ts
                )
        return True


class ZappyCallback(BaseCallback):
    """Logging des métriques toutes les N steps."""

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
            team_state=team_state
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

    vec_env = SubprocVecEnv([
        make_env(i, args.host, args.port, args.team, 0, team_state)
        for i in range(args.clients)
    ])

    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)
    model_path = checkpoint_dir / "zappy_ppo"

    if args.resume and (checkpoint_dir / f"{args.resume}.zip").exists():
        logger.info(f"Reprise depuis {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device="cpu")
    else:
        logger.info("Nouveau modèle PPO")
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            device="cpu",
            verbose=0,
        )

    callbacks = [
        CurriculumCallback(verbose=0),
        ZappyCallback(interval_sec=30, verbose=0),
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
