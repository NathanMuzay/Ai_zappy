"""
Zappy — Entraînement PPO multi-agents avec Stable Baselines3.
=====================================================================

Modifications from base version:
  - Ajout du CheckpointCallback (sauvegarde tous les 50 000 steps).
  - Support --resume pour reprendre un entraînement interrompu.
  - Le modèle est sauvegardé dans data/ckpt_XXXXX.zip à chaque checkpoint.

  ==============================================================
  ===  LIGNES AJOUTÉES — Reportez-les manuellement si vous   ===
  ===  avez déjà une version personnalisée de train.py :     ===
  ===                                                             ===
  ===  1) Import : voir section "# === AJOUT : imports ==="  ===
  ===  2) CheckpointCallback : voir classe plus bas         ===
  ===  3) Instanciation : voir dans main() vers "ckpt_cb"  ===
  ===  4) Passage au learn() : callback=[..., ckpt_cb]      ===
  ===  5) Chargement resume : dans le bloc try/if resume    ===
  ==============================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.env import ZappyEnv
from src.team_state import TeamState

LOG_DIR = "logs"
DATA_DIR = "data"

# === AJOUT : imports ========================================================
CHECKPOINT_DIR = DATA_DIR
CHECKPOINT_FREQ = 50_000   # sauvegarder tous les 50 000 steps


# === AJOUT : CheckpointCallback =============================================
class CheckpointCallback(BaseCallback):
    """
    Sauvegarde le modèle PPO dans data/ckpt_<step>.zip
    à intervalles réguliers (en nombre de steps cumulés).

    Le fichier le plus récent est aussiCopié vers
    data/ckpt_latest.zip pour faciliter la reprise.

    Utilisé pour survivre aux crashs serveur / reboot machine :
    `make train RESUME=data/ckpt_latest.zip` reprend exactement
    là où l'entraînement s'était arrêté.
    """

    def __init__(
        self,
        checkpoint_freq: int = CHECKPOINT_FREQ,
        checkpoint_dir: str = CHECKPOINT_DIR,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, step: int) -> None:
        path = self.checkpoint_dir / f"ckpt_{step}.zip"
        latest = self.checkpoint_dir / "ckpt_latest.zip"
        self.model.save(str(path))
        # copie "latest" pour方便 la reprise
        self.model.save(str(latest))
        if self.verbose:
            print(f"[CheckpointCallback] saved → {path}", flush=True)

    def _on_step(self) -> bool:
        # stable-baselines3 appelle _on_step après chaque batch de steps
        ts = self.num_timesteps
        # on calcule le reste modulo pour détecter le franchissement du seuil
        if ts > 0 and ts % self.checkpoint_freq == 0:
            self._save(ts)
        return True


# Custom callbacks


class MinuteLogger(BaseCallback):
    """
    Log le reward moyen par minute (60 secondes temps réel).

    ATTENTION : cette classe utilise BaseCallback mais ne l'importait pas
    dans le fichier original. Si vous portez cette version, ajoutez :
        from stable_baselines3.common.callbacks import BaseCallback
    en haut du fichier.
    """

    def __init__(self, total_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self._reward_window: list[float] = []
        self._window_start: float | None = None
        self._window_duration = 60.0          # seconds

    # Internal helpers
    def _log_window(self) -> None:
        ts = self.num_timesteps
        avg = sum(self._reward_window) / len(self._reward_window) if self._reward_window else 0.0
        stamp = datetime.now().strftime("%H:%M:%S")
        logging.getLogger("zappy.train").info(
            "steps=%d | reward/min=%.2f | elapsed=60s",
            ts, avg,
        )

    def _reset_window(self) -> None:
        self._reward_window.clear()
        self._window_start = time.time()

    # stable-baselines3 hook
    def _on_step(self) -> bool:
        # Initialise timer on first call
        if self._window_start is None:
            self._window_start = time.time()

        rewards = self.locals.get("rewards", [])
        self._reward_window.extend(rewards)

        elapsed = time.time() - self._window_start
        if elapsed >= self._window_duration:
            self._log_window()
            self._reset_window()

        return True


class EpisodeRewardRecorder(BaseCallback):
    """
    Enregistre le reward cumulé par épisode et sauvegarde un graphique
    à la fin de l'entraînement.

    a complete history at the end of training.
    On ``_on_training_end`` the list is written to logs/rewards.csv
    and a PNG is generated.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self._initialised = False
        self._per_env_reward: list[float] = []
        self._ep_rewards: list[float] = []

    # Helpers
    def _ensure_init(self, n_envs: int) -> None:
        if not self._initialised:
            self._per_env_reward = [0.0] * n_envs
            self._initialised = True

    def _save(self) -> None:
        os.makedirs(LOG_DIR, exist_ok=True)
        csv_path = os.path.join(LOG_DIR, "rewards.csv")
        png_path = os.path.join(LOG_DIR, "rewards.png")

        # CSV
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("episode,reward\n")
            for ep, r in enumerate(self._ep_rewards):
                f.write(f"{ep},{r}\n")

        # PNG
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 4))
            plt.title("Reward per episode")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path, dpi=120)
            plt.close()
            if self.verbose:
                logging.getLogger("zappy.train").info(
                    "Episode rewards saved: %s / %s", csv_path, png_path,
                )
        except Exception as exc:
            if self.verbose:
                logging.getLogger("zappy.train").warning(
                    "Could not save reward plot: %s", exc,
                )

    # stable-baselines3 hooks
    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones    = self.locals.get("dones", [])
        self._ensure_init(len(rewards))

        for i, (r, done) in enumerate(zip(rewards, dones)):
            self._per_env_reward[i] += float(r)
            if done:
                self._ep_rewards.append(self._per_env_reward[i])
                self._per_env_reward[i] = 0.0

        return True

    def _on_training_end(self) -> None:
        self._save()


# Utility helpers


def setup_logging() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
        ],
    )


def make_env_fn(idx: int, team_state, host: str, port: int, team: str):
    def _init():
        return ZappyEnv(
            host=host,
            port=port,
            team=team,
            agent_id=idx,
            team_state=team_state,
        )
    return _init


def build_model(vec_env) -> PPO:
    return PPO(
        "MlpPolicy",
        vec_env,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        verbose=1,
    )


def save_metadata(args, duration: float, victory: bool, model_path: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    meta = {
        "timesteps": args.timesteps,
        "host":      args.host,
        "port":      args.port,
        "team":      args.team,
        "clients":   args.clients,
        "duration_sec": round(duration, 1),
        "victory":   victory,
        "model_path": model_path,
    }
    meta_path = os.path.join(DATA_DIR, "training_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


# Main


def main() -> None:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",      default="localhost")
    parser.add_argument("--port",      default=4242, type=int)
    parser.add_argument("--team",      default="ia")
    parser.add_argument("--clients",   default=6,    type=int)
    parser.add_argument("--win_count", default=6,    type=int)
    parser.add_argument("--timesteps", default=2_000_000, type=int)
    parser.add_argument("--resume",    default=None)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("zappy.train")
    logger.info(
        "Demarrage entrainement multi-agents : %s",
        {
            "host":      args.host,
            "port":      args.port,
            "team":      args.team,
            "clients":   args.clients,
            "win_count": args.win_count,
            "timesteps": args.timesteps,
            "resume":    args.resume,
        },
    )

    team_state = TeamState(nb_clients=args.clients, win_count=args.clients)
    env_fns    = [
        make_env_fn(i, team_state, args.host, args.port, args.team)
        for i in range(args.clients)
    ]
    vec_env    = SubprocVecEnv(env_fns)

    victory    = False
    model_path = os.path.join(DATA_DIR, "ppo_zappy")

    try:
        # === AJOUT : Chargement resume (optionnel) ==========================
        if args.resume and os.path.exists(args.resume):
            logger.info("Reprise du modèle depuis : %s", args.resume)
            model = PPO.load(args.resume, env=vec_env)
        else:
            model = build_model(vec_env)
        # ====================================================================

        minute_logger = MinuteLogger(args.timesteps, verbose=1)
        ep_recorder   = EpisodeRewardRecorder(verbose=1)

        # === AJOUT : CheckpointCallback ====================================
        ckpt_cb = CheckpointCallback(
            checkpoint_freq=CHECKPOINT_FREQ,
            checkpoint_dir=CHECKPOINT_DIR,
            verbose=1,
        )
        # ====================================================================

        start = time.time()
        model.learn(
            total_timesteps=args.timesteps,
            callback=[minute_logger, ep_recorder, ckpt_cb],   # <-- ckpt_cb ajouté
        )
        duration = time.time() - start

        model.save(model_path)
        logger.info(
            "Entrainement termine | duree=%.1fs | modele=%s",
            duration, model_path,
        )

    except Exception as exc:
        raise
    finally:
        vec_env.close()

    save_metadata(args, duration, victory, model_path)


if __name__ == "__main__":
    main()
