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

class VideoLogger(Process):
    def __init__(self,**kwargs):
        self.name = kwargs["name"]
        self.writer = None

    def process(self,frame,**kwargs):
        if self.writer is None:
            self.writer = cv2.VideoWriter(self.name, cv2.VideoWriter_fourcc(*'XVID'), 20, (frame.shape[1],frame.shape[0]))

        self.writer.write(frame)

    def finalize(self,**kwargs):
        if self.writer is not None:
            self.writer.release()

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
    def __init__(self, color, gradient, **kwargs):
        super().__init__(**kwargs)
        self.color = color
        self.gradient = gradient

    def process(self,frame,**kwargs):
        body = np.sign(frame[:,:,0]-self.color)
        grad_1 = np.sign((frame[:,:,1]-127)**2+(frame[:, :, 2]-127)**2 - self.gradient**2)

        return 127*np.stack([body, grad_1, grad_1], axis=-1)+127

class HueSaturation(Process):
    def __init__(self, hue, multi, **kwargs):
        super().__init__(**kwargs)
        self.hue = hue/2
        self.multi = multi


    def process(self,frame,**kwargs):
        hls = cv2.cvtColor(frame,cv2.COLOR_BGR2HLS).astype(np.float16)

        hue_distance = (1/180)*np.minimum(np.abs(hls[:,:,0]-self.hue),180-np.abs(hls[:,:,0]-self.hue))
        return hls[:,:,2]*np.maximum(0,1 -self.multi*hue_distance)

class Threshold(Process):
    def __init__(self, threshold, **kwargs):
        self.threshold = threshold

    def process(self,frame,**kwargs):
        return np.where(frame>self.threshold,255,0)

class Circle(Process):
    def process(self,frame,original = None,**kwargs):
        # Reduce noise
        frame = frame.astype(np.uint8)
        #gray = cv2.medianBlur(frame, 9)
        gray = cv2.erode(frame,np.ones((3,3),np.uint8),iterations=1)
        gray = cv2.dilate(gray,np.ones((3,3),np.uint8),iterations=1)
        gray = cv2.blur(frame,(7,7))
        #gray = np.clip(2*gray-16, 0, 255)
        if original is not None:
            output = original.copy()
        else:
            output = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
        # Detect circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=10,
            param2=10,
            minRadius=15,
            maxRadius=30
        )

        # Draw only the first detected circle
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # draw the outer circle
                cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

        return output

PROCESS = {
    "none":Process,
    "avg":AvgProcess,
    "fore":ForeGround,
    "pipe":PipeLine,
    "log":Logger,
    "color":ColorCut,
    "hmap":HeightMap,
    "cliff":CliffCut,
    "hs":HueSaturation,
    "circle":Circle,
    "threshold":Threshold,
    "video":VideoLogger,
}