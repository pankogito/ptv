import csv
import sqlite3

from process import Process


class LogSQCircles(Process):
    def __init__(self,name,**kwargs):
        self.name = name
        self.db = sqlite3.connect(name)
        self.db.execute("""CREATE TABLE IF NOT EXISTS circles (frame,cam_x,cam_y,radius)""")
        self.frame = 0

    def process(self,circles,**kwargs):
        self.db.executemany("INSERT INTO circles VALUES (?,?,?,?)",[[self.frame,float(circle[0]),float(circle[1]),float(circle[2])] for circle in circles])
        self.db.commit()
        self.frame += 1

    def finalize(self,**kwargs):
        self.db.close()

class LoadCircles(Process):
    def __init__(self,name,**kwargs):
        super().__init__()
        self.name = name
        self.db = sqlite3.connect(name)
        self.frame = 0
        self.frame_max = self.db.execute("SELECT max(frame) FROM circles").fetchone()[0]

    def process(self,**kwargs):
        circles = self.db.execute(f"SELECT cam_x,cam_y,radius FROM circles WHERE frame={self.frame}").fetchall()
        self.frame += 1
        print(circles)
        return circles



PROCESS = {
    "log circles": LogSQCircles,
    "load circles": LoadCircles,
}


