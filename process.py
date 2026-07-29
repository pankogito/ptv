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
        self.pipeline = list(map(lambda config:self.prepare(config,kwargs["processes"]),kwargs["pipeline"]))

    def prepare(self,config,processes):
        return processes[config["operator"]](**config.get("options",{}))

    def process(self,frame,**kwargs):
        for process in self.pipeline:
            frame = process.process(frame,**kwargs)

    def finalize(self,**kwargs):
        for process in self.pipeline:
            process.finalize(**kwargs)

class Logger(Process):
    def __init__(self,**kwargs):
        self.name = kwargs["name"]
        self.period = kwargs.get("period",1)
        self.counter = 0
    def process(self,frame,**kwargs):
        if self.counter % self.period == 0:
            cv2.imwrite(self.name,frame.astype(np.uint8))
        self.counter += 1
        return frame

class ColorCut(Process):
    def __init__(self,**kwargs):
        self.color = np.array(kwargs["color"])

    def process(self,frame,**kwargs):
        return np.sum(255-np.abs(frame-self.color),axis=-1)/frame.shape[-1]

class HeightMap(Process):
    def process(self,frame,**kwargs):
        x = np.gradient(frame,1,axis=0) + 127
        y = np.gradient(frame,1,axis=1) + 127

        return np.stack([frame,x,y],axis=-1)


class CliffCut(Process):
    def __init__(self,color,gradient,**kwargs):
        self.color = color
        self.gradient = gradient

    def process(self,frame,**kwargs):
        body = np.sign(frame[:,:,0]-self.color)
        grad_1 = np.sign((frame[:,:,1]-127)**2+(frame[:, :, 2]-127)**2 - self.gradient**2)

        return 127*np.stack([body, grad_1, grad_1], axis=-1)+127



PROCESS = {
    "none":Process,
    "avg":AvgProcess,
    "fore":ForeGround,
    "pipe":PipeLine,
    "log":Logger,
    "color":ColorCut,
    "hmap":HeightMap,
    "cliff":CliffCut,

}