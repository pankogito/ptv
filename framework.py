import json
import sqlite3
import process
import cv2


import videosupport.video_player as player

def from_config(config,value):
    if value in config:
        return config[value]
    else:
        return input(f"Enter {value}:")




config = json.load(open('config.json'))
db = sqlite3.connect(config['db'])

writer = cv2.VideoWriter(config["out"]+"video.avi",cv2.VideoWriter_fourcc(*'XVID'),20,[3072,2048])

start = "2026-06-03 11:43:45"
end = "2026-06-03 11:45:00"

op = process.PROCESS[from_config(config,"operator")](**config["options"],processes=process.PROCESS)

count = 0
for frame in player.read_from_database(db,config['video'],8,start,end):
    if count % 100 == 0:
        print(count)
    count += 1

    op.process(frame)

op.finalize()
writer.release()
cv2.destroyAllWindows()