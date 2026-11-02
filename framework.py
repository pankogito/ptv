import json
import sqlite3
import time

import process
import cv2
import yaml

import videosupport.video_player as player

def from_config(config,value):
    if value in config:
        return config[value]
    else:
        return input(f"Enter {value}:")


skip = 0

config = yaml.safe_load(open('config.yaml'))


db = sqlite3.connect(config['db'])


op = process.PipeLine(pipeline=config["pipeline"],processes=process.PROCESS,skip=skip)

start_time = time.time()
count = 0
for frame in player.read_from_database(db,config['video'],8,config["start"],config["end"]):
    if count % 100 == 0:
        print(count, time.time() - start_time)
    count += 1
    op.process(frame)

op.finalize()
cv2.destroyAllWindows()