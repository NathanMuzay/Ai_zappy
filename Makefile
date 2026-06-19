PORT         ?= 4242
WIDTH        ?= 10
HEIGHT       ?= 10
TEAM         ?= ia
CLIENTS      ?= 6
SERVER_SLOTS := $(shell echo $$(($(CLIENTS)*2)))
FREQ         ?= 100
DURATION     ?= 3600
TIMESTEPS    ?= 2000000
RESUME       ?=

SERVER     = ./zappy_server
GUI        = ./zappygui.AppImage
PYTHON     = python3
VENV       = .venv


all: server
	@sleep 4
	$(MAKE) train
	$(MAKE) stop

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

server:
	-@kill `cat .supervisor.pid` 2>/dev/null || true
	-@kill `cat .server.pid` 2>/dev/null || true
	-@fuser -k $(PORT)/tcp 2>/dev/null || true
	@sleep 1
	@echo "Supervisor + serveur lances (logs -> logs/server.log)"
	$(PYTHON) src/supervisor.py &
	@echo $$! > .supervisor.pid
	@sleep 3

gui:
	$(GUI) -h localhost -p $(PORT)

train:
	@echo "Demarrage entrainement (RESUME=$(RESUME))"
	$(VENV)/bin/python -m src.train \
		--host localhost \
		--port $(PORT) \
		--team $(TEAM) \
		--clients $(CLIENTS) \
		--timesteps $(TIMESTEPS) \
		$(if $(RESUME),--resume $(RESUME),)

tensorboard:
	$(VENV)/bin/tensorboard --logdir logs/tb

stop:
	-@kill `cat .supervisor.pid` 2>/dev/null || true
	-@kill `cat .server.pid` 2>/dev/null || true
	-@fuser -k $(PORT)/tcp 2>/dev/null || true
	-@rm -f .supervisor.pid .server.pid
	@echo "Supervisor + serveur arretes"

clean:
	rm -rf __pycache__ src/__pycache__ .server.pid .supervisor.pid

fclean: clean stop
	rm -rf data logs $(VENV)

re: fclean all

help:
