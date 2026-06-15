#!/usr/bin/python3

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from tools import load_yaml_config
from zappy_env import ZappyEnv


def _build_env_config(config: dict) -> dict:
    env_cfg = dict(config.get("env", {}))
    env_cfg.setdefault("host", config.get("host", "127.0.0.1"))
    env_cfg.setdefault("port", int(config.get("port", 3000)))
    env_cfg.setdefault("team_name", config.get("team_name", "Br"))
    env_cfg.setdefault("timeout_seconds", float(config.get("timeout_seconds", 2.0)))
    env_cfg.setdefault("max_steps_per_episode", int(config.get("max_steps_per_episode", 2000)))
    env_cfg.setdefault("render_mode", config.get("render_mode", "none"))
    return env_cfg


def evaluate(config_path: str, episodes_override: int | None = None, render_override: bool | None = None, video: bool = False) -> None:
    config = load_yaml_config(config_path)
    env_cfg = _build_env_config(config)

    model_path = config.get("model_path", "models/zappy_ppo.zip")
    episodes = int(episodes_override if episodes_override is not None else config.get("episodes", 10))
    render_mode = "human" if (render_override if render_override is not None else bool(config.get("render", False))) else "none"
    deterministic = bool(config.get("deterministic", True))
    seed = int(config.get("seed", 42))
    # Meme politique partagee que pour l'entrainement (page 9 : un client =
    # un joueur) : on connecte n_players joueurs simultanement pour pouvoir
    # observer de vraies elevations de groupe (table page 5).
    n_players = max(1, int(config.get("n_players", 1)))

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first with run.py."
        )

    envs = [ZappyEnv(**env_cfg) for _ in range(n_players)]
    model = PPO.load(model_path, env=envs[0])
    if render_mode == "human":
        for env in envs:
            env.set_render_enabled(True)

    episode_team_rewards = []

    for episode in range(episodes):
        observations = [env.reset(seed=seed + episode * n_players + i)[0] for i, env in enumerate(envs)]
        done = [False] * n_players
        player_rewards = [0.0] * n_players
        max_level = 1

        while not all(done):
            for i, env in enumerate(envs):
                if done[i]:
                    continue
                action, _ = model.predict(observations[i], deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                observations[i] = obs
                player_rewards[i] += float(reward)
                max_level = max(max_level, int(info.get("level", 1)))
                done[i] = bool(terminated or truncated)
                if render_mode == "human":
                    env.render()

        team_reward = sum(player_rewards)
        episode_team_rewards.append(team_reward)
        if n_players == 1:
            print(f"Episode {episode + 1}/{episodes} reward: {player_rewards[0]:.2f} | max_level={max_level}")
        else:
            details = ", ".join(f"p{i+1}={r:.2f}" for i, r in enumerate(player_rewards))
            print(f"Episode {episode + 1}/{episodes} team_reward: {team_reward:.2f} ({details}) | max_level={max_level}")

    for env in envs:
        env.close()

    mean_reward = sum(episode_team_rewards) / len(episode_team_rewards)
    label = "team_reward" if n_players > 1 else "reward"
    print(f"Mean {label} over {episodes} episodes: {mean_reward:.2f}")
    if video:
        print("Video recording is not implemented for the socket-based Zappy environment.")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Zappy PPO model")
    parser.add_argument(
        "--config",
        default="configs/eval.yaml",
        help="Path to evaluation config YAML",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override the episode count from the config")
    parser.add_argument("--render", action="store_true", help="Print live environment traces")
    parser.add_argument("--video", action="store_true", help="Keep compatibility with the old CLI")
    args = parser.parse_args()
    evaluate(args.config, episodes_override=args.episodes, render_override=args.render, video=args.video)


if __name__ == "__main__":
    main()