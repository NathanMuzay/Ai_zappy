PYTHON = python3
SRC    = src
CONFIGS = configs

.PHONY: all train server gui eval clean fclean re help

all: train

train:
	@mkdir -p models data
	$(PYTHON) $(SRC)/train_dual.py \
		--config-a $(CONFIGS)/agent_br.yaml \
		--config-b $(CONFIGS)/agent_of.yaml

server:
	./zappy_server -p 3000 -x 15 -y 10 -n Br of -c 20 -f 10000 --auto-start on --display-eggs true

gui:
	./zappygui.AppImage -p 3000

clean:
	@rm -f data/*.log data/*.png

fclean: clean
	@rm -f models/zappy_ppo_Br.zip models/zappy_ppo_of.zip

re: fclean all

help:
	@echo "make          -> entrainement complet (2 agents)"
	@echo "make server   -> serveur seul"
	@echo "make gui      -> GUI"
	@echo "make clean    -> supprime logs/graphes"
	@echo "make fclean   -> supprime logs/graphes/modeles"
	@echo "make re       -> repart de zero"
