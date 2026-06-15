#!/usr/bin/env python3
"""
Lance 2 agents en parallele avec redemarrage automatique du serveur.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# train_dual.py est dans src/AI/src/ -> parent = src/AI/src, parent.parent = src/AI/
# zappy_server est dans src/AI/ = ROOT, donc ROOT est correct.

SERVER_CMD = [
	str(ROOT / "zappy_server"),
	"-p", "3000", "-x", "15", "-y", "10",
	"-n", "Br", "of",
	"-c", "20", "-f", "10000",
	"--auto-start", "on", "--display-eggs", "true",
]


def wait_for_server(timeout=30.0) -> bool:
	import socket
	deadline = time.time() + timeout
	while time.time() < deadline:
		try:
			with socket.create_connection(("127.0.0.1", 3000), timeout=1.0) as s:
				if "WELCOME" in s.recv(64).decode(errors="ignore"):
					return True
		except OSError:
			pass
		time.sleep(0.5)
	return False


def start_server() -> subprocess.Popen:
	server_bin = Path(SERVER_CMD[0])
	if not server_bin.exists():
		raise RuntimeError(f"zappy_server introuvable : {server_bin}\n  ROOT={ROOT}")
	log_path = ROOT / "data" / "server.log"
	log_path.parent.mkdir(parents=True, exist_ok=True)
	log_fd = open(log_path, "w")
	proc = subprocess.Popen(SERVER_CMD, cwd=str(ROOT), stdout=log_fd, stderr=log_fd)
	if not wait_for_server():
		log_fd.flush()
		raise RuntimeError(f"Serveur non demarré en 30s — voir {log_path}")
	print(f"► Serveur prêt (PID {proc.pid})")
	return proc


def stop_server(proc: subprocess.Popen):
	if proc.poll() is not None:
		return
	proc.terminate()
	try:
		proc.wait(timeout=5)
	except subprocess.TimeoutExpired:
		proc.kill()
		proc.wait()
	print("► Serveur arrêté")


def launch_agent(config: str, label: str, append: bool) -> subprocess.Popen:
	log = ROOT / "data" / f"train_{label}.log"
	log.parent.mkdir(parents=True, exist_ok=True)
	fd = open(log, "a" if append else "w", encoding="utf-8")
	proc = subprocess.Popen(
		[sys.executable, str(ROOT / "src" / "train.py"), "--config", config],
		cwd=str(ROOT), stdout=fd, stderr=subprocess.STDOUT, text=True,
	)
	print(f"  [{label}] PID {proc.pid} → {log}")
	return proc


def is_done(label: str) -> bool:
	log = ROOT / "data" / f"train_{label}.log"
	if not log.exists():
		return False
	return "Training complete" in log.read_text(encoding="utf-8", errors="ignore")


def has_crashed(label: str) -> bool:
	log = ROOT / "data" / f"train_{label}.log"
	if not log.exists():
		return False
	txt = log.read_text(encoding="utf-8", errors="ignore")
	return any(p in txt for p in ["RuntimeError", "TimeoutError", "OSError", "No slot"])


def tail(label: str):
	log = ROOT / "data" / f"train_{label}.log"
	if not log.exists():
		return
	lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
	# Afficher les 3 dernieres lignes significatives (pas les tracebacks)
	shown = [l for l in lines[-20:] if any(k in l for k in ["steps", "reward", "Progress", "level", "fps", "Training", "ep_rew", "ep_len", "value_loss", "policy"])]
	for l in shown[-3:]:
		print(f"  [{label}] {l}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config-a", default="configs/agent_br.yaml")
	parser.add_argument("--config-b", default="configs/agent_of.yaml")
	args = parser.parse_args()

	print("=" * 55)
	print("  Zappy Dual Training")
	print("=" * 55)

	server = start_server()
	cycle = 1
	append = False

	proc_a = launch_agent(args.config_a, "Br", append)
	time.sleep(5.0)
	proc_b = launch_agent(args.config_b, "of", append)
	append = True

	last_rewards = {"Br": None, "of": None}
	last_steps   = {"Br": 0, "of": 0}

	def extract_stats(label: str):
		"""Extrait steps et reward depuis les logs Progress: du ProgressCallback."""
		log = ROOT / "data" / f"train_{label}.log"
		if not log.exists():
			return None, None
		lines = log.read_text(encoding="utf-8", errors="ignore").splitlines()
		steps = reward = None
		for l in reversed(lines):
			# Format: "  1024/500000 (0.2%) | fps=6 | avg_reward=-8.32 | max_level=1"
			if "avg_reward=" in l and steps is None:
				try:
					steps = int(l.strip().split("/")[0])
					reward = float(l.split("avg_reward=")[1].split()[0])
				except:
					pass
				break
		return steps, reward

	try:
		while True:
			time.sleep(10)
			a_done = proc_a.poll() is not None
			b_done = proc_b.poll() is not None

			steps_a, rew_a = extract_stats("Br")
			steps_b, rew_b = extract_stats("of")

			print(f"\n[{time.strftime('%H:%M:%S')}] Cycle {cycle}")
			print(f"  Br: {'fini' if a_done else 'actif':<6} | steps={steps_a or '?':>7} | reward_moyen={rew_a if rew_a is not None else '?'}")
			print(f"  of: {'fini' if b_done else 'actif':<6} | steps={steps_b or '?':>7} | reward_moyen={rew_b if rew_b is not None else '?'}")

				# Watchdog : vérifie que le serveur est toujours vivant
			server_dead = (server.poll() is not None)
			if not server_dead:
				try:
					import socket as _sock
					with _sock.create_connection(("127.0.0.1", 3000), timeout=2.0) as s:
						server_dead = "WELCOME" not in s.recv(64).decode(errors="ignore")
				except OSError:
					server_dead = True

			if server_dead:
				print(f"\n⚠ Serveur mort détecté — redemarrage forcé (cycle {cycle + 1})...")
				for p in (proc_a, proc_b):
					if p.poll() is None:
						p.kill()
				stop_server(server)
				time.sleep(2.0)
				server = start_server()
				cycle += 1
				proc_a = launch_agent(args.config_a, "Br", append)
				time.sleep(5.0)
				proc_b = launch_agent(args.config_b, "of", append)
				continue

			if a_done and b_done:
				if is_done("Br") and is_done("of"):
					print("\n✓ Entrainement terminé normalement.")
					break
				else:
					print(f"\n► Crash détecté — redemarrage cycle {cycle + 1}...")
					stop_server(server)
					time.sleep(2.0)
					server = start_server()
					cycle += 1
					proc_a = launch_agent(args.config_a, "Br", append)
					time.sleep(5.0)
					proc_b = launch_agent(args.config_b, "of", append)

	except KeyboardInterrupt:
		print("\n► Arrêt manuel...")
		for p in (proc_a, proc_b):
			if p.poll() is None:
				p.terminate()
		for p in (proc_a, proc_b):
			try: p.wait(timeout=5)
			except subprocess.TimeoutExpired: p.kill()
		stop_server(server)

	else:
		stop_server(server)

	print("\n" + "=" * 55)
	print("  Modeles : models/zappy_ppo_Br.zip  models/zappy_ppo_of.zip")
	print("  Logs    : data/train_Br.log  data/train_of.log")
	print("=" * 55)


if __name__ == "__main__":
	main()