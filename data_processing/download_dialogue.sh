#!/bin/bash

python3 -m pip install parlai

python3 -m parlai.scripts.display_data --task dailydialog --datapath data/dialogue
python3 -m parlai.scripts.display_data --task blended_skill_talk --datapath data/dialogue
python3 -m parlai.scripts.display_data --task empathetic_dialogues --datapath data/dialogue
python3 -m parlai.scripts.display_data --task personachat --datapath data/dialogue