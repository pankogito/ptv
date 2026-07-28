from multiprocessing.spawn import prepare

import cv2
import numpy as np


class Process:
    def __init__(self,**kwargs):
        pass

    def process(self,frame,**kwargs):
        pass

    def finalize(self,**kwargs):
        pass

class AvgProcess(Process):
    def __init__(self,**kwargs):
        self.avg = None
        self.count = 0

    def process(self,frame,**kwargs):
        if self.avg is None:
            self.avg = frame.astype(np.int64)
        else:
            self.avg += frame
        self.count += 1

    def finalize(self,**kwargs):
        cv2.imwrite("log_avg.png", (self.avg/self.count).astype(np.uint8))

class ForeGround(Process):
    def __init__(self,**kwargs):
        self.mask = cv2.imread(kwargs["mask"])
        self.foreground = 64

    def process(self,frame,**kwargs):
        mask = self.mask-frame

        mask = 255 - mask.sum(axis=-1,keepdims=True)/3

        return np.concatenate((frame,mask),axis=-1)

class PipeLine(Process):
    def __init__(self,**kwargs):
        self.pipeline = list(map(lambda config:prepare(config,kwargs["processes"]),kwargs["pipeline"]))

    def prepare(self,config,processes):
        return processes[config["operator"]](config["options"])

    def process(self,frame,**kwargs):
        for process in self.pipeline:
            frame = process(frame,**kwargs)

    def finalize(self,**kwargs):
        for process in self.pipeline:
            process.finalize(**kwargs)


PROCESS = {
    "none":Process,
    "avg":AvgProcess,
    "fore":ForeGround

}
