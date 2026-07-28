import json
import sqlite3

import cv2

import numpy as np

import videosupport.video_player as player

def preprocess_frame(frame,*args):
    frame -= frame.min(0).min(0)
    frame = frame.astype(np.float16)/frame.max(0).max(0)
    frame *= 255
    return frame.astype(np.uint8)

def process_frame(frame,color):
    process = (frame-color)**2
    process = 0xff - np.sum(process,2)/3/0xff

    process = process**2/0xff

    process = process ** 2 / 0xff

    #process = cv2.threshold(process,240,255,cv2.THRESH_BINARY)[1]

    process = process.astype(np.uint8)
    return cv2.cvtColor(process,cv2.COLOR_GRAY2BGR)



config = json.load(open('config.json'))
db = sqlite3.connect(config['db'])




color =np.array([[[0x00,0x00,0xff]]])


start = "2026-06-03 11:42:50"
end = "2026-06-03 11:43:00"
#start = "2026-06-02 10:42:20"
#end = "2026-06-02 10:42:30"

hist = None

sum = None
sum_hsl = None
i = 0
for frame in player.read_from_database(db,config['video'],8,start,end):
    # writer.write(process_frame(frame,color))

    if sum is None:
        sum = np.zeros(frame.shape,np.uint64)
        sum_hsl = np.zeros(frame.shape,np.uint64)


    sum += frame
    sum_hsl += cv2.cvtColor(frame,cv2.COLOR_BGR2HLS)

    # h,e = np.histogramdd(np.reshape(cv2.cvtColor(frame,cv2.COLOR_BGR2HLS),(frame.shape[0]*frame.shape[1],3)),bins=15)
    # if hist is None:
    #     hist = h
    # else:
    #     hist += h

    i += 1

sum = (sum/i).astype(np.uint8)
cv2.imwrite(config['out']+"avg.png",sum)

sum_hsl = cv2.cvtColor((sum_hsl/i).astype(np.uint8),cv2.COLOR_HLS2BGR)
cv2.imwrite(config['out']+"avg_hsl.png",sum_hsl)

diff = cv2.absdiff(sum,sum_hsl)
cv2.imwrite(config['out']+"avg_diff.png",diff*5)

print(diff.max())

# # hist /=np.max(hist)
# hist = np.sqrt(hist)
# for i,h in enumerate(hist):
#     cv2.imwrite(config["out"]+f"hist_{i}.png",(255*h).astype(np.uint8) )