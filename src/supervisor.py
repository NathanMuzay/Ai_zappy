#!/usr/bin/env python3
"""
Superviseur Zappy — relance automatique du serveur quand les oeufs sont épuisés.
Avec support du curriculum learning (phase-aware).
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Configuration
SERVER_BIN   = "./zappy_server"
PORT         = 4242
WIDTH        = 10
HEIGHT       = 10
TEAM         = "ia"
CLIENTS      = 200        # taille max du pool
FREQ         = 100
LOG_FILE     = "logs/server.log"
PID_FILE     = ".server.pid"
SUP_PID_FILE = ".supervisor.pid"

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

def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[supervisor {stamp}] {msg}", flush=True)

def server_has_slots(port: int, team: str, timeout: float = 1.0) -> bool:
    """Handshake reel : WELCOME + lecture slots."""
    try:
        with socket.create_connection(("localhost", port), timeout=timeout) as s:
            s.settimeout(timeout)
            buf = b""
            while b"\n" not in buf:
                buf += s.recv(256)
            if not buf.startswith(b"WELCOME"):
                return False
            s.sendall((team + "\n").encode())
            buf = buf.split(b"\n", 1)[1]
            while b"\n" not in buf:
                buf += s.recv(256)
            slots = buf.split(b"\n", 1)[0].strip()
            return slots.lstrip(b"-").isdigit() and int(slots) > 0
    except (OSError, ValueError):
        return False

def get_server_pid(pid_file: str) -> int | None:
    try:
        with open(pid_file) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None

def is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def launch_server() -> subprocess.Popen[bytes]:
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
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=3)
            log("Serveur arrete proprement.")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            log("Serveur tue violemment (kill).")

    if pid and is_process_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            if is_process_alive(pid):
                os.kill(pid, signal.SIGKILL)
                log(f"PID {pid} tue par SIGKILL.")
        except OSError:
            pass

HEALTH_CHECK_INTERVAL = 5
DOWN_GRACE            = 20

def main() -> None:
    log("Demarrage du supervisor.")

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

    proc: subprocess.Popen[bytes] | None = None
    proc = launch_server()
    current_pid = proc.pid
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

    with open(SUP_PID_FILE, "w") as f:
        f.write(str(os.getpid()) + "\n")

    while running:
        time.sleep(HEALTH_CHECK_INTERVAL)

        proc_poll = proc.poll() if proc else None
        if proc_poll is not None:
            log("Le processus serveur est mort (code=%s). Relance." % proc_poll)
            kill_server(proc, current_pid)
            time.sleep(2)
            proc = launch_server()
            current_pid = proc.pid
            down_since = None
            continue

        if not server_has_slots(PORT, TEAM):
            if down_since is None:
                down_since = time.time()
                log("Serveur injoignable ou oeufs epuises — debut du comptage.")
            else:
                elapsed = time.time() - down_since
                if elapsed >= DOWN_GRACE:
                    log("Serveur injoignable depuis >= %ds — relance serveur." % DOWN_GRACE)
                    kill_server(proc, current_pid)
                    time.sleep(2)
                    proc = launch_server()
                    current_pid = proc.pid
                    down_since = None
        else:
            if down_since is not None:
                log("Serveur a nouveau joignable.")
                down_since = None

if __name__ == "__main__":
    main()
