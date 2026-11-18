import json
import sqlite3
import time

import process
import cv2
import yaml

import asyncio

import process_log
import videosupport.video_player as player

def from_config(config,value):
    if value in config:
        return config[value]
    else:
        return input(f"Enter {value}:")


skip = 0

config = yaml.safe_load(open('config.yaml'))


db = sqlite3.connect(config['db'])


op = process.PipeLine(pipeline=config["pipeline"],processes=process.PROCESS | process_log.PROCESS,skip=skip)

async def op_routine(frame):
    if frame is None:
        return
    else:
        op.process(frame)

async def iterate(gen):
    return next(gen,None)

async def main_routine():
    start_time = time.time()
    count = 0

    reader = player.read_from_database(db,config['video'],4,config["start"],config["end"])

    frame = next(reader,None)

    while frame is not None:
        if count % 100 == 0:
            print(count//20,"->", time.time() - start_time)
        count += 1

        _,frame = await asyncio.gather(op_routine(frame),iterate(reader))


asyncio.run(main_routine())

op.finalize()
cv2.destroyAllWindows()