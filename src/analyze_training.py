#!/usr/bin/env python3
"""
analyze_training.py — Analyse d'entraînement Zappy (PPO / SB3)

Usage:
  python3 analyze_training.py                                    # demo
  python3 analyze_training.py --log logs/server.log
  python3 analyze_training.py --monitor-dir logs/ --tb-logdir runs/
  python3 analyze_training.py --outdir /tmp/my_report

Sorties (--outdir, défaut /mnt/user-data/outputs/zappy_report) :
  - PNGs : un par graphique
  - report.pdf : toutes les figures
  - summary.md : statistiques + diagnostic auto
"""

import argparse
import os
import sys
import re
import warnings

# ─── Dépendances facultatives ────────────────────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import matplotlib
    matplotlib.use("Agg")          # headless
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


# ─── Parsing ─────────────────────────────────────────────────────────────────

def parse_train_log(path: str):
    """Extrait steps, mean_reward, n depuis un log texte style :
    INFO:zappy.train:steps=331074 | mean_reward=3.609 | n=30
    """
    pattern = re.compile(
        r"steps\s*=\s*(\d+)\s*\|\s*mean_reward\s*=\s*([-+]?\d*\.?\d+)\s*\|\s*n\s*=\s*(\d+)",
        re.IGNORECASE,
    )
    records = []
    if not os.path.exists(path):
        print(f"  [WARN] log introuvable : {path}")
        return records
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                records.append(
                    {"steps": int(m.group(1)),
                     "mean_reward": float(m.group(2)),
                     "n": int(m.group(3))}
                )
    print(f"  Train log : {len(records)} entrées lues depuis {path}")
    return records


def parse_monitor_csv(path: str):
    """Parse un fichier .monitor.csv de Stable-Baselines3."""
    if not os.path.exists(path):
        return []
    if not HAS_PANDAS:
        # fallback texte minimal
        records = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3 and parts[0]:
                    try:
                        records.append({"episode": len(records)+1,
                                        "r": float(parts[0]),
                                        "l": int(parts[1])})
                    except ValueError:
                        pass
        print(f"  Monitor CSV : {len(records)} épisodes (sans pandas)")
        return records

    df = pd.read_csv(path, comment="#", header=None)
    # Le fichier Monitor a une ligne header puis des données r,l,t
    if df.shape[1] < 3:
        return []
    df.columns = ["r", "l", "t"] + list(df.columns[3:]) if df.shape[1] > 3 else ["r", "l", "t"]
    records = df[["r", "l"]].copy()
    records["episode"] = range(1, len(records) + 1)
    print(f"  Monitor CSV : {len(records)} épisodes lus depuis {path}")
    return records.to_dict("records")


def find_monitor_csvs(monitor_dir: str):
    """Trouve tous les fichiers monitor.csv / *.monitor.csv dans un dossier."""
    results = []
    if not os.path.isdir(monitor_dir):
        return results
    for root, _, files in os.walk(monitor_dir):
        for f in files:
            if f == "monitor.csv" or f.endswith(".monitor.csv"):
                results.append(os.path.join(root, f))
    return results


def parse_eval_npz(path: str):
    if not os.path.exists(path):
        return None, None
    if not HAS_NUMPY:
        return None, None
    data = np.load(path)
    timesteps = data.get("timesteps", None)
    results = data.get("results", None)
    print(f"  Eval npz : timesteps={timesteps is not None}, results={results is not None}")
    return timesteps, results


def parse_tensorboard(logdir: str):
    """Extrait les scalaires loss depuis un dossier TensorBoard."""
    if not os.path.exists(logdir) or not HAS_TB:
        return {}
    try:
        ea = event_accumulator.EventAccumulator(
            logdir,
            size_guidance={event_accumulator.SCALARS: 0},
        )
        ea.Reload()
        scalars = {}
        tags = ea.Tags().get("scalars", [])
        # Filtre les tags loss pertinents
        wanted = ["loss", "policy_loss", "value_loss", "entropy_loss",
                  "explained_variance", "approx_kl", "clip_fraction",
                  "learning_rate"]
        for tag in tags:
            if any(w in tag.lower() for w in wanted):
                events = ea.Scalars(tag)
                scalars[tag] = [(e.step, e.value) for e in events]
        print(f"  TensorBoard : {len(scalars)} séries extraites")
        return scalars
    except Exception as e:
        print(f"  [WARN] TB parsing échoué : {e}")
        return {}


# ─── Génération de données de demo ───────────────────────────────────────────

def generate_synthetic_log(n_points: int = 3000, seed: int = 42):
    """Crée un log texte synthétique qui imite le format INFO:zappy.train."""
    if not HAS_NUMPY:
        return []
    rng = np.random.default_rng(seed)
    # Progression lente + bruit
    steps = np.cumsum(rng.integers(60, 300, size=n_points))
    trend = np.linspace(-0.5, 0.8, n_points) * 50
    noise = rng.normal(0, 1.2, size=n_points)
    mean_reward = trend + noise
    # Quelques spikes aléatoires (rare positive burst)
    spike_idx = rng.choice(n_points, size=max(1, n_points // 100), replace=False)
    mean_reward[spike_idx] += rng.uniform(3, 8, size=len(spike_idx))
    n = rng.integers(20, 400, size=n_points)
    records = [
        {"steps": int(s), "mean_reward": float(r), "n": int(nn)}
        for s, r, nn in zip(steps, mean_reward, n)
    ]
    return records


def write_synthetic_log(records, path: str):
    lines = [f"INFO:zappy.train:steps={r['steps']} | mean_reward={r['mean_reward']:.3f} | n={r['n']}\n"
             for r in records]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(lines)


# ─── Fenêtrage / lissage ─────────────────────────────────────────────────────

def rolling_mean(values, window: int):
    if HAS_NUMPY:
        return np.convolve(values, np.ones(window)/window, mode="same").tolist()
    else:
        out = []
        for i in range(len(values)):
            lo = max(0, i - window // 2)
            hi = min(len(values), i + window // 2 + 1)
            out.append(sum(values[lo:hi]) / (hi - lo))
        return out


# ─── Graphiques ───────────────────────────────────────────────────────────────

def plot_reward_vs_steps(records, outdir: str, demo: bool):
    if not HAS_MATPLOTLIB or not records:
        return
    steps = [r["steps"] for r in records]
    rewards = [r["mean_reward"] for r in records]
    smoothed = rolling_mean(rewards, 50)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(steps, rewards, alpha=0.25, s=8, label="mean_reward", zorder=1)
    ax.plot(steps, smoothed, color="orange", lw=2, label="moyenne glissante (fenêtre=50)", zorder=2)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward moyen au fil de l'entraînement" + (" [DEMO]" if demo else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "reward_vs_steps.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_episode_length(records, outdir: str, demo: bool):
    if not HAS_MATPLOTLIB or not records:
        return
    steps = [r["steps"] for r in records]
    lengths = [r["n"] for r in records]
    smoothed = rolling_mean(lengths, 50)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(steps, lengths, alpha=0.25, s=8, label="n (longueur épisodique)", zorder=1)
    ax.plot(steps, smoothed, color="green", lw=2, label="moyenne glissante (fenêtre=50)", zorder=2)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Episode length (n)")
    ax.set_title("Longueur des épisodes au fil de l'entraînement" + (" [DEMO]" if demo else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "episode_length_vs_steps.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_reward_histogram(records, outdir: str, demo: bool):
    if not HAS_MATPLOTLIB or not records:
        return
    rewards = [r["mean_reward"] for r in records]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rewards, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(rewards), color="red", lw=1.5, label=f"moyenne={np.mean(rewards):.2f}")
    ax.set_xlabel("Mean Reward")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des rewards moyens" + (" [DEMO]" if demo else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "reward_histogram.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_monitor_episodes(monitor_records, outdir: str, demo: bool):
    if not HAS_MATPLOTLIB or not monitor_records:
        return
    episodes = [r["episode"] for r in monitor_records]
    rewards_ep = [r["r"] for r in monitor_records]
    lengths_ep = [r["l"] for r in monitor_records]

    # reward par épisode
    smoothed_r = rolling_mean(rewards_ep, 20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(episodes, rewards_ep, alpha=0.3, s=6, label="reward épisodique")
    ax.plot(episodes, smoothed_r, color="purple", lw=2, label="moyenne glissante")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward épisodique (Monitor CSV)" + (" [DEMO]" if demo else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "monitor_episode_rewards.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")

    # longueur par épisode
    smoothed_l = rolling_mean(lengths_ep, 20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(episodes, lengths_ep, alpha=0.3, s=6, label="longueur épisodique")
    ax.plot(episodes, smoothed_l, color="teal", lw=2, label="moyenne glissante")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Longueur des épisodes (Monitor CSV)" + (" [DEMO]" if demo else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "monitor_episode_lengths.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_tb_scalars(tb_scalars, outdir: str, demo: bool):
    if not HAS_MATPLOTLIB or not tb_scalars:
        return
    short_names = {
        "loss": "loss_total",
        "policy_loss": "policy_loss",
        "value_loss": "value_loss",
        "entropy_loss": "entropy_loss",
        "explained_variance": "explained_variance",
        "approx_kl": "approx_kl",
        "clip_fraction": "clip_fraction",
        "learning_rate": "learning_rate",
    }
    for tag, data in tb_scalars.items():
        steps_tb, values_tb = zip(*data)
        short = short_names.get(os.path.basename(tag), os.path.basename(tag))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(steps_tb, values_tb, lw=1.2, alpha=0.7)
        smoothed = rolling_mean(list(values_tb), 50)
        ax.plot(steps_tb, smoothed, lw=2, color="orange", label="moyenne glissante")
        ax.set_xlabel("Steps")
        ax.set_ylabel(short)
        ax.set_title(f"{short} (TensorBoard)" + (" [DEMO]" if demo else ""))
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(outdir, f"tb_{short}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  saved: {path}")


# ─── PDF combiné ──────────────────────────────────────────────────────────────

def build_pdf(figures_dir: str, outdir: str, demo: bool):
    """Combine tous les PNG du dossier en un seul report.pdf."""
    if not HAS_MATPLOTLIB:
        return
    try:
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        return

    pdf_path = os.path.join(outdir, "report.pdf")
    png_files = sorted([
        f for f in os.listdir(figures_dir)
        if f.endswith(".png")
    ])
    if not png_files:
        print("  [WARN] aucun PNG à combiner en PDF")
        return

    with PdfPages(pdf_path) as pdf:
        for png in png_files:
            img = matplotlib.image.imread(os.path.join(figures_dir, png))
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(111)
            ax.imshow(img, aspect="auto")
            ax.axis("off")
            ax.set_title(png)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  saved: {pdf_path}")


# ─── Summary markdown ─────────────────────────────────────────────────────────

def write_summary(records, monitor_records, tb_scalars, outdir: str, demo: bool):
    if not records and not monitor_records:
        return

    lines = ["# 📊 Résumé d'entraînement — Zappy RL\n"]

    # — source —
    if demo:
        lines.append("> ⚠️ **MODE DEMO** — données générées artificiellement.\n")
        lines.append("Lancez ce script avec `--log <chemin>` pour voir les vraies données.\n")
    lines.append("---\n")

    # — stats principales —
    if records:
        rewards = [r["mean_reward"] for r in records]
        ns = [r["n"] for r in records]
        steps_list = [r["steps"] for r in records]

        total_steps = steps_list[-1] if steps_list else 0
        final_rolling = rolling_mean(rewards, min(50, len(rewards)))[-1] if len(rewards) >= 3 else 0
        best = max(rewards)
        worst = min(rewards)
        positive_pct = sum(1 for r in rewards if r > 0) / len(rewards) * 100
        mean_n = np.mean(ns) if HAS_NUMPY else sum(ns) / len(ns)

        # Comparaison premiers / derniers tiers
        third = len(rewards) // 3
        early_n = np.mean(ns[:third]) if HAS_NUMPY else sum(ns[:third]) / third
        late_n  = np.mean(ns[-third:]) if HAS_NUMPY else sum(ns[-third:]) / third
        n_trend = "↑ en hausse" if late_n > early_n * 1.05 else ("↓ en baisse" if late_n < early_n * 0.95 else "→ stable")

        lines.append(f"""
## Statistiques — Log d'entraînement

| Métrique | Valeur |
|---|---|
| Total steps simulés | **{total_steps:,}** |
| Points de log | **{len(records)}** |
| Reward moyen final (fenêtre 50) | **{final_rolling:+.3f}** |
| Best reward | **{best:+.3f}** |
| Worst reward | **{worst:+.3f}** |
| % points positifs | **{positive_pct:.1f}%** |
| Longueur épisodique moyenne | **{mean_n:.1f}** |
| Tendance longueur (1er vs 3e tiers) | **{n_trend}** ({early_n:.1f} → {late_n:.1f})
""")
    else:
        lines.append("> Aucun point de log d'entraînement disponible.\n")

    # — monitor stats —
    if monitor_records:
        ep_rewards = [r["r"] for r in monitor_records]
        ep_lengths = [r["l"] for r in monitor_records]
        lines.append(f"""
## Statistiques — Monitor CSV

| Métrique | Valeur |
|---|---|
| Nombre d'épisodes | **{len(monitor_records)}** |
| Reward moyen par épisode | **{np.mean(ep_rewards):+.2f}** |
| Longueur épisodique moyenne | **{np.mean(ep_lengths):.1f}** |

""")

    # — TB summary —
    if tb_scalars:
        lines.append("## Scalaires TensorBoard disponibles\n")
        for tag in tb_scalars:
            lines.append(f"- `{tag}`\n")
        lines.append("\n")

    # — diagnostic auto —
    diag = []
    if records:
        if final_rolling < -1.0:
            diag.append("- Le reward moyen reste négatif → l'agent reçoit des pénalités importantes (mort, famine, coûts de temps). Vérifiez que le serveur Zappy est joignable et que les commandes `Incantation` sont bien détectées.")
        elif -1.0 <= final_rolling < 0.5:
            diag.append("- Reward proche de zéro : l'agent survit mais ne progresse pas significativement. Il se peut que les incantations soient trop coûteuses par rapport aux récompenses de victoire ou de montée de niveau.")
        elif 0.5 <= final_rolling < 2.0:
            diag.append("- Reward modérément positif : l'agent apprend des comportements utiles (collecte, survie). La progression reste lente — envisagez d'ajuster les coefficients de reward ou le learning rate.")
        else:
            diag.append("- Reward élevé : l'agent semble bien apprendre. Surveillez qu'il ne s'appuie pas sur un comportement sous-optimal mais stable (ex. collecte excessive de nourriture au détriment de l'élévation).")

        if positive_pct < 30:
            diag.append("- Moins de 30 % des points ont un reward positif → les récompenses positives sont rares. Vérifiez les constantes de reward et la difficulté du serveur.")
        if n_trend == "↑ en hausse":
            diag.append("- La longueur des épisodes augmente → l'agent vit plus longtemps (bon signe de survie) mais ne finit pas les episodes plus vite.")
        elif n_trend == "↓ en baisse":
            diag.append("- La longueur des épisodes diminue → l'agent apprend à terminer plus vite, potentiellement en mourant précocement.")

    if monitor_records and ep_rewards:
        if np.mean(ep_rewards) < 0:
            diag.append("- Le reward épisodique moyen est négatif → l'agent est pénalisé davantage qu'il ne gagne. Réviser la balance des coûts/rewards de victoire/décès.")
    if tb_scalars.get("loss"):
        vals = [v for _, v in tb_scalars["loss"]]
        if vals and max(vals) > 100:
            diag.append("- La loss totale atteint des valeurs très élevées (>100) → instabilité d'entraînement. Essayez de réduire le learning rate ou le nombre de clients simulés.")
        if vals and max(vals) < 0.001:
            diag.append("- La loss totale est très faible (<0.001) → l'agent a peut-être cessé d'apprendre. Vérifiez qu'il reçoit toujours des gradients non-nuls.")

    if not diag:
        diag = ["- Données insuffisantes pour émettre un diagnostic automatique. Lancez l'entraînement plus longtemps et ré-exécutez ce script."]

    lines.append("## 🔍 Diagnostic automatique\n")
    for d in diag:
        lines.append(d + "\n")

    lines.append(f"""
---
*Rapport généré par `analyze_training.py` — { "données DEMO" if demo else "données réelles" }. Pour regénérer : `python3 analyze_training.py --log <fichier> --outdir <dossier>`*
""")

    path = os.path.join(outdir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved: {path}")
    print("\n" + "\n".join(lines))


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse l'entraînement Zappy (PPO/SB3). "
                    "Génère des graphiques PNG, un PDF combiné et un summary.md.",
    )
    p.add_argument("--log",        default=None,
                   help="Chemin vers le fichier log texte (INFO:zappy.train:...)")
    p.add_argument("--monitor-dir", default=None,
                   help="Dossier contenant des fichiers .monitor.csv (Stable-Baselines3)")
    p.add_argument("--eval",       default=None,
                   help="Chemin vers un fichier evaluations.npz (SB3)")
    p.add_argument("--tb-logdir",  default=None,
                   help="Dossier de logs TensorBoard (si tensorboard disponible)")
    p.add_argument("--outdir",     default="/mnt/user-data/outputs/zappy_report",
                   help="Dossier de sortie (défaut: /mnt/user-data/outputs/zappy_report)")
    p.add_argument("--demo",       action="store_true",
                   help="Force le mode demo (génère des données synthétiques)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    demo_mode = args.demo

    # ── Collecte des données ──────────────────────────────────────────────────
    records = []        # log principal : steps / mean_reward / n
    monitor_all = []    # tous les monitor CSV fusionnés
    tb_scalars = {}     # dict tag -> [(step, value), ...]

    sources_used = []

    # 1) log texte
    log_path = args.log or os.environ.get("ZAPPY_TRAIN_LOG", "")
    # AUTO-DÉTECTION : chemins probables avant de tomber en demo
    if not log_path:
        for candidate in ("logs/train.log", "logs/zappy.train.log"):
            if os.path.exists(candidate) and parse_train_log(candidate):
                log_path = candidate
                break
    if log_path:
        records = parse_train_log(log_path)
        if records:
            sources_used.append(f"log texte ({log_path})")


    # 2) monitor CSV
    mon_dir = args.monitor_dir
    if mon_dir:
        for mpath in find_monitor_csvs(mon_dir):
            monitor_all.extend(parse_monitor_csv(mpath))
        if monitor_all:
            sources_used.append(f"monitor CSV ({mon_dir})")

    # 3) eval npz
    if args.eval:
        _, eval_results = parse_eval_npz(args.eval)
        if eval_results is not None:
            sources_used.append(f"evaluations.npz ({args.eval})")

    # 4) TensorBoard
    if args.tb_logdir:
        tb_scalars = parse_tensorboard(args.tb_logdir)
        if tb_scalars:
            sources_used.append(f"TensorBoard ({args.tb_logdir})")

    # ── Mode demo si aucune source ───────────────────────────────────────────
    if not records and not monitor_all and not tb_scalars or args.demo:
        demo_mode = True
        print("=== MODE DEMO (aucune source de données trouvée) ===")
        print("Génération de données synthétiques…")
        demo_records = generate_synthetic_log(n_points=3000)
        demo_log_path = os.path.join(outdir, "synthetic_train.log")
        write_synthetic_log(demo_records, demo_log_path)
        records = demo_records
        sources_used.append(f"données synthétiques (log: {demo_log_path})")
        print(f"  synthetic log écrit : {demo_log_path}")

    # ── Rendu ────────────────────────────────────────────────────────────────
    print(f"\nSources utilisées : {', '.join(sources_used) if sources_used else 'aucune'}")
    print(f"Output directory  : {outdir}\n")

    print("Génération des graphiques…")
    plot_reward_vs_steps(records, outdir, demo_mode)
    plot_episode_length(records, outdir, demo_mode)
    plot_reward_histogram(records, outdir, demo_mode)
    plot_monitor_episodes(monitor_all, outdir, demo_mode)
    plot_tb_scalars(tb_scalars, outdir, demo_mode)

    print("Création du PDF combiné…")
    build_pdf(outdir, outdir, demo_mode)

    print("Écriture du summary.md…")
    write_summary(records, monitor_all, tb_scalars, outdir, demo_mode)

    print(f"\n✅ Terminé — outputs dans : {outdir}")
    print("Fichiers générés :")
    for f in sorted(os.listdir(outdir)):
        print(f"   {f}")


if __name__ == "__main__":
    main()