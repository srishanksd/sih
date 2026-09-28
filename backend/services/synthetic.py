import numpy as np


def field(t=0, size=96):
    y, x = np.mgrid[0:size, 0:size]
    cx, cy = 28 + t*.7, 48 + np.sin(t/8)*10
    core = np.exp(-(((x-cx)**2)/(2*10**2) + ((y-cy)**2)/(2*8**2)))
    cell2 = .65*np.exp(-(((x-(70-t*.35))**2)/(2*12**2) + ((y-62)**2)/(2*10**2)))
    return np.clip(.05 + .92*core + cell2, 0, 1)

def sensors():
    return {"DWR": .98, "INSAT": .94, "ILDN": .91, "AWS": .88, "GFS": .86}
