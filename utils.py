import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DIRECTION = {0:(-1,0),1:(0,1),2:(1,0),3:(0,-1)}

def visualize(maze, ax):
    """
    Parameters
    -----------------
    maze : ndarray
        迷路を表す配列
    
    Returns
    -----------------
    None
    """

    # Figureを用意
    ax.clear()
    ax.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)

    # カラーマップを生成
    List=[(0, "#FFFFFF"),(0.5,"#FF0000"),(1,"#000000")]
    cm=LinearSegmentedColormap.from_list("",List)
    ax.imshow(maze,cmap=cm)

def plot(y, minimum, ax, xmax=100, ymax=1000):
    
    ax.clear()
    x = np.arange(1,len(y)+1)
    ax.set_xlim(1,xmax)
    ax.set_ylim(0,ymax)
    ax.plot(x, y, color="#2200FF")
    ax.plot((0,xmax),(minimum,minimum),color="#FF0000",linewidth=5,alpha=0.2)