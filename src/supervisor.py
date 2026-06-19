#!/usr/bin/env python3
"""
Superviseur Zappy — relance automatique du serveur quand les接客 sont épuisés.
=====================================================================

Problème résolu :
  Le serveur Zappy démarre avec un pool fini d'oeufs (-c N).
  Chaque connexion d'agent consume un oeuf ; la mort ne le rend PAS.
  Une fois le pool épuisé, le serveur refuse toute nouvelle connexion,
  l'env se bloque dans reset()/_connect() → l'entraînement gèle.

Solution :
  Ce script surveille en permanence la santé du serveur.
  Si le processus meurt OU si le port devient inaccessible pendant
  plusieurs secondes (pointe = pool épuisé), on relance le serveur
  avec un nouveau pool d'oeufs. L'entraînement (PPO) vit dans train.py
  et n'est PAS touché par le redémarrage.

Dépendances : stdlib uniquement (socket, subprocess, time, os, signal).
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Configuration — miroir du Makefile original
# ----------------------------------------------------------------------
SERVER_BIN   = "./zappy_server"
PORT         = 4242
WIDTH        = 10
HEIGHT       = 10
TEAM         = "ia"
CLIENTS      = 200          # taille du pool d'oeufs
FREQ         = 100
LOG_FILE     = "logs/server.log"
PID_FILE     = ".server.pid"
SUP_PID_FILE = ".supervisor.pid"

# ----------------------------------------------------------------------
# Commandes de lancement du serveur
# ----------------------------------------------------------------------
SERVER_CMD = [
    SERVER_BIN,
    "-p", str(PORT),
    "-x", str(WIDTH),
    "-y", str(HEIGHT),
    "-n", TEAM,
    "-c", str(CLIENTS),
    "-f", str(FREQ),
    "--display-eggs", "true",
    "-v",
]

# ----------------------------------------------------------------------
# Log propre
# ----------------------------------------------------------------------
def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[supervisor {stamp}] {msg}", flush=True)


# ----------------------------------------------------------------------
# Vérification de santé TCP
# ----------------------------------------------------------------------
def server_is_alive(port: int, timeout: float = 1.0) -> bool:
    """
    Tente une connexion TCP rapide sur le port.
    Retourne True si on établit une connexion, False sinon.
    """
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except (OSError, socket.error):
        return False


def get_server_pid(pid_file: str) -> int | None:
    """Lit le PID du serveur depuis le fichier .server.pid."""
    try:
        with open(pid_file) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def is_process_alive(pid: int) -> bool:
    """Vérifie si un processus existe encore (évite les PID réutilisés)."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ----------------------------------------------------------------------
# Lancement / arrêt du serveur
# ----------------------------------------------------------------------
def launch_server() -> subprocess.Popen[bytes]:
    """
    Lance le serveur, écrit son PID dans .server.pid,
    et retourne le objet Popen.
    """
    os.makedirs("logs", exist_ok=True)
    log_file_fd = open(LOG_FILE, "a")

    proc = subprocess.Popen(
        SERVER_CMD,
        stdout=log_file_fd,
        stderr=subprocess.STDOUT,
    )

    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid) + "\n")

    log(f"Serveur lance (PID={proc.pid}) — log dans {LOG_FILE}")
    return proc


def kill_server(proc: subprocess.Popen[bytes] | None, pid: int | None) -> None:
    """Tente de tuer le serveur proprement, puis violemment si nécessaire."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=3)
            log("Serveur arrete proprement.")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            log("Serveur tue violemment (kill).")

    # Nettoyage supplémentaire via PID fichier
    if pid and is_process_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            if is_process_alive(pid):
                os.kill(pid, signal.SIGKILL)
                log(f"PID {pid} tue par SIGKILL.")
        except OSError:
            pass


# ----------------------------------------------------------------------
# Boucle de surveillance
# ----------------------------------------------------------------------
HEALTH_CHECK_INTERVAL = 5   # secondes entre chaque vérification
DOWN_GRACE            = 20   # secondes avant de considérer le serveur mort

def main() -> None:
    log("Demarrage du supervisor.")

    # Nettoyage d'un éventuel serveur orphelin au démarrage
    old_pid = get_server_pid(PID_FILE)
    if old_pid and is_process_alive(old_pid):
        log(f"Nettoyage serveur orphelin PID={old_pid}.")
        try:
            os.kill(old_pid, signal.SIGTERM)
            time.sleep(1)
            if is_process_alive(old_pid):
                os.kill(old_pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.remove(PID_FILE)
        except FileNotFoundError:
            pass

    # Lancement initial
    proc: subprocess.Popen[bytes] | None = None
    proc = launch_server()
    current_pid = proc.pid

    # Temps depuis lequel le serveur est considéré "down"
    down_since: float | None = None

    running = True

    def signal_handler(signum: int, _frame) -> None:
        nonlocal running, proc, current_pid
        running = False
        log("Signal recu, arret du supervisor.")
        kill_server(proc, current_pid)
        for f in (PID_FILE, SUP_PID_FILE):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass
        log("Supervisor arrete.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT,  signal_handler)

    # Écrire notre propre PID pour que le Makefile puisse nous tuer
    with open(SUP_PID_FILE, "w") as f:
        f.write(str(os.getpid()) + "\n")

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------
    while running:
        time.sleep(HEALTH_CHECK_INTERVAL)

        # 1) Le processus serveur est-il mort ?
        proc_poll = proc.poll() if proc else None
        if proc_poll is not None:
            # Le processus a terminé tout seul
            log("Le processus serveur est mort (code=%s). Relance." % proc_poll)
            kill_server(proc, current_pid)
            time.sleep(2)
            proc = launch_server()
            current_pid = proc.pid
            down_since = None
            continue

        # 2) Le port est-il joignable ?
        if not server_is_alive(PORT):
            if down_since is None:
                down_since = time.time()
                log("Serveur injoignable — debut du comptage.")
            else:
                elapsed = time.time() - down_since
                if elapsed >= DOWN_GRACE:
                    log("Serveur injoignable depuis > %ds — "
                        "relance serveur (oeufs epuises ?" % DOWN_GRACE)
                    kill_server(proc, current_pid)
                    time.sleep(2)
                    proc = launch_server()
                    current_pid = proc.pid
                    down_since = None
                # else: on attend encore, le serveur peut revenir
        else:
            # Serveur joignable → on reset le compteur
            if down_since is not None:
                log("Serveur a nouveau joignable.")
                down_since = None


if __name__ == "__main__":
    main()
