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

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)
logger = logging.getLogger("zappy.train")

CURRICULUM_PHASE1_STEPS = 100000   # steps avant phase 2
MAX_TIMESTEPS = 500000

class CurriculumCallback(BaseCallback):
    """Callback qui ajuste les rewards en fonction de la phase d'apprentissage."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.phase = 1
        self.phase1_end = CURRICULUM_PHASE1_STEPS
    
    def _on_step(self) -> bool:
        # Déterminer la phase actuelle
        new_phase = 1 if self.num_timesteps < self.phase1_end else 2
        
        if new_phase != self.phase:
            self.phase = new_phase
            if self.phase == 2:
                logger.info(
                    f"🎯 TRANSITION PHASE 1→2 à {self.num_timesteps} steps. "
                    f"Focus : SURVIE → REPRODUCTION (Fork activation)"
                )
        
        return True

class ZappyCallback(BaseCallback):
    """Logging des metriques toutes les N steps."""
    
    def __init__(self, interval_sec=30, verbose=0):
        super().__init__(verbose)
        self.interval_sec = interval_sec
        self.last_time = time.time()
        self.rewards = []

    def _on_step(self) -> bool:
        # collecte des rewards du batch courant
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
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--win_count", type=int, default=6)
    
    args = parser.parse_args()
    
    logger.info(
        f"Demarrage entrainement multi-agents : {vars(args)}"
    )
    
    # État partagé de l'équipe
    team_state = TeamState(win_count=args.win_count)
    
    # Création des environnements en parallèle
    vec_env = SubprocVecEnv([
        make_env(i, args.host, args.port, args.team, 0, team_state)
        for i in range(args.clients)
    ])
    
    # Checkpoints
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)
    model_path = checkpoint_dir / "zappy_ppo"
    
    # Chargement ou création du modèle
    if args.resume and (checkpoint_dir / f"{args.resume}.zip").exists():
        logger.info(f"Resume depuis {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device="cpu")
    else:
        logger.info("Nouveau modele PPO")
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
    
    # Callbacks
    callbacks = [
        CurriculumCallback(verbose=0),
        ZappyCallback(interval_sec=30, verbose=0),
    ]

    
    try:
        # Entraînement principal
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=False,
        )
        logger.info("✅ Entrainement termine avec succes")
    except KeyboardInterrupt:
        logger.info("⏹️  Entrainement interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur durant entrainement: {e}")
        raise
    finally:
        # Sauvegarde
        model.save(str(model_path))
        logger.info(f"Modele sauvegarde : {model_path}.zip")
        vec_env.close()

if __name__ == "__main__":
    main()
