import json
import sqlite3
import process
import cv2
import yaml

import videosupport.video_player as player

def from_config(config,value):
    if value in config:
        return config[value]
    else:
        return input(f"Enter {value}:")




config = yaml.safe_load(open('config.yaml'))


db = sqlite3.connect(config['db'])


op = process.PROCESS[from_config(config,"operator")](**config["options"],processes=process.PROCESS)

count = 0
for frame in player.read_from_database(db,config['video'],8,config["start"],config["end"]):
    if count % 100 == 0:
        print(count)
    count += 1
    original = frame.copy()
    op.process(frame,original=original)

op.finalize()
cv2.destroyAllWindows()