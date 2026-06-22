# ═══════════════════════════════════════════════════════════════════
#  Makefile Zappy AI — Curriculum Learning (3 phases)
#
#  Phase 1 : carte 10×10 + nourriture abondante (600)
#            → 100 000 steps : l'agent apprend à survivre et se reproduire
#  Phase 2 : carte 20×20 + nourriture normale (100)
#            → 400 000 steps : l'agent monte en niveau, commence à collecter
#  Phase 3 : grande carte 42×42 + nourriture rare (10)
#            → 500 000 steps : performance finale, coordination multi-agents
#
#  Commandes principales :
#    make all          — exécute les 3 phases à la suite (1M steps total)
#    make phase1       — phase 1 seule
#    make phase2       — phase 2 seule (reprend depuis phase1)
#    make phase3       — phase 3 seule (reprend depuis phase2)
#    make train        — entraînement rapide (défaut : 1M steps)
#    make analyze      — analyse les logs existants
#    make stop         — arrête le serveur
#    make clean        — nettoyage léger
#    make fclean       — nettoyage complet (supprime aussi models + logs)
#
#  Variables d'environnement pour personnaliser :
#    SERVER_BIN        chemin vers le binaire zappy_server (défaut: ./zappy_server)
#    PORT              port du serveur (défaut: 4242)
#    CLIENTS           taille du pool d'œufs (défaut: 6)
#    TIMESTEPS         nombre de steps pour `make train` (défaut: 1000000)
#    RESUME            checkpoint à reprendre (ex: RESUME=models/zappy_ppo)
#    WIDTH, HEIGHT     taille de la carte (min 10, max 42 imposé par le serveur)
#    FREQ              fréquence de jeu
# ═══════════════════════════════════════════════════════════════════

PYTHON      = python3
VENV        = .venv
SERVER_BIN  ?= ./zappy_server
GUI         = ./zappygui.AppImage
PORT        ?= 4242
TEAM        ?= ia
CLIENTS     ?= 6
TIMESTEPS   ?= 1000000
RESUME      ?=

.PHONY: all phase1 phase2 phase3 train analyze server gui tensorboard stop clean fclean install help

# ── Macros ───────────────────────────────────────────────────────
define SUPERVISOR_START
    @echo "=== Supervisor : WIDTH=$(WIDTH) HEIGHT=$(HEIGHT) FREQ=$(FREQ) CLIENTS=$(CLIENTS) ==="
    -@kill `cat .supervisor.pid` 2>/dev/null || true
    -@kill `cat .server.pid` 2>/dev/null || true
    -@fuser -k $(PORT)/tcp 2>/dev/null || true
    @sleep 1
    @echo "Démarrage du supervisor..."
    $(PYTHON) src/supervisor.py &
    @echo $$! > .supervisor.pid
    @sleep 4
endef

define SUPERVISOR_STOP
    @echo "=== Arrêt du supervisor ==="
    -@kill `cat .supervisor.pid` 2>/dev/null || true
    -@kill `cat .server.pid` 2>/dev/null || true
    -@fuser -k $(PORT)/tcp 2>/dev/null || true
    -@rm -f .supervisor.pid .server.pid
endef

define TRAIN
    @echo "Démarrage entraînement (RESUME=$(RESUME), TIMESTEPS=$(TIMESTEPS))"
    $(VENV)/bin/python -m src.train \
        --host localhost \
        --port $(PORT) \
        --team $(TEAM) \
        --clients $(CLIENTS) \
        --timesteps $(TIMESTEPS) \
        $(if $(RESUME),--resume $(RESUME),)
endef

define ANALYZE
    @echo "=== Lancement de l'analyse ==="
    $(VENV)/bin/python src/analyze_training.py \
        --log logs/server.log \
        --outdir logs/report
endef

# ── Phases curriculum ────────────────────────────────────────────
phase1:
	$(call SUPERVISOR_START)
	$(call TRAIN)
	$(call SUPERVISOR_STOP)
	$(call ANALYZE)

phase2:
	$(call SUPERVISOR_START)
	$(call TRAIN)
	$(call SUPERVISOR_STOP)
	$(call ANALYZE)

phase3:
	$(call SUPERVISOR_START)
	$(call TRAIN)
	$(call SUPERVISOR_STOP)
	$(call ANALYZE)

# ── Curriculum complet ───────────────────────────────────────────
all:
	@echo "═══════════════════════════════ PHASE 1 / 3 ═══════════════════════════════"
	$(MAKE) phase1 WIDTH=10 HEIGHT=10 FREQ=600 TIMESTEPS=100000 RESUME=
	@echo ""
	@echo "═══════════════════════════════ PHASE 2 / 3 ═══════════════════════════════"
	$(MAKE) phase2 WIDTH=20 HEIGHT=20 FREQ=100 TIMESTEPS=400000 RESUME=models/zappy_ppo
	@echo ""
	@echo "═══════════════════════════════ PHASE 3 / 3 ═══════════════════════════════"
	$(MAKE) phase3 WIDTH=42 HEIGHT=42 FREQ=10 TIMESTEPS=500000 RESUME=models/zappy_ppo
	@echo ""
	@echo "═══════════════════════════════ ANALYSE FINALE ═══════════════════════════════"
	$(call ANALYZE)
	@echo "✅ Curriculum terminé — voir logs/report/summary.md"

# ── Entraînement rapide ──────────────────────────────────────────
train:
	$(call SUPERVISOR_START)
	$(call TRAIN)
	$(call SUPERVISOR_STOP)
	$(call ANALYZE)

# ── Analyse seule ────────────────────────────────────────────────
analyze:
	$(call ANALYZE)

# ── Serveur seul ─────────────────────────────────────────────────
server:
	$(call SUPERVISOR_START)

# ── GUI ──────────────────────────────────────────────────────────
gui:
	$(GUI) -h localhost -p $(PORT)

# ── TensorBoard ──────────────────────────────────────────────────
tensorboard:
	$(VENV)/bin/tensorboard --logdir logs/tb

# ── Arrêt ────────────────────────────────────────────────────────
stop:
	$(call SUPERVISOR_STOP)

# ── Nettoyage ────────────────────────────────────────────────────
clean:
	rm -rf __pycache__ src/__pycache__ .server.pid .supervisor.pid

fclean: clean stop
	rm -rf models logs $(VENV)

# ── Installation ─────────────────────────────────────────────────
install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

help:
	@echo "Zappy AI — Curriculum Learning"
	@echo ""
	@echo "  make all        — Curriculum complet (1M steps, 3 phases)"
	@echo "  make phase1     — Phase 1 : carte 10x10 + nourriture abondante"
	@echo "  make phase2     — Phase 2 : carte 20x20 + nourriture normale"
	@echo "  make phase3     — Phase 3 : carte 42x42 + nourriture rare"
	@echo "  make train      — Entraînement rapide (1M steps par défaut)"
	@echo "  make analyze    — Analyse les logs existants"
	@echo "  make stop       — Arrête le serveur"
	@echo "  make clean      — Nettoyage léger"
	@echo "  make fclean     — Nettoyage complet"
	@echo ""
	@echo "Variables : WIDTH HEIGHT FREQ CLIENTS TIMESTEPS RESUME PORT TEAM"
	@echo "  Ex: make train WIDTH=10 HEIGHT=10 TIMESTEPS=500000 RESUME=models/zappy_ppo"
