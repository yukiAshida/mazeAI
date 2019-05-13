import numpy as np
import matplotlib.pyplot as plt
from mazeGenarator import initilize
from mazeGenarator import generate
from mazeGenarator import visualize
from mazeGenarator import Solver
from utils import DIRECTION



def possibleAction(maze):

    w,h = maze.shape
    possibles = np.zeros((w,h,4),dtype=bool)
    
    base = maze[1:-1,1:-1].copy()
    
    # 現在位置と移動方向が両方とも道である場合True
    up = (maze[:-2,1:-1] == base)*(base == 0)
    down = (maze[2:,1:-1] == base)*(base == 0)
    left = (maze[1:-1,:-2] == base)*(base == 0)
    right = (maze[1:-1,2:] == base)*(base == 0)

    possibles[1:-1,1:-1] = np.array([up,right,down,left]).transpose(1,2,0)
    return possibles


def getNext(now_pos,action):
    """
    Parameters
    ---------------
    maze: ndarray
        迷路を表す配列

    action: int
        0:上 1:右 2:下 3:左
    
    Returns
    ---------------
    next_pos: tuple
        次の場所 (w,h)
    """

    dw,dh = DIRECTION[action]
    next_pos = (now_pos[0]+dw,now_pos[1]+dh)

    return next_pos


if __name__=="__main__":

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1)
    
    maze = initilize(10,10)
    maze = generate(maze)
    possibles = possibleAction(maze)

    q_value = 

    pos = (maze.shape[0]-2,1)
    
    action = None

    for i in range(100000):
        
        # maze_copy = maze.copy()
        # maze_copy[pos[0]][pos[1]] = 0.5
        # visualize(maze_copy,ax)
        # plt.pause(0.01)

        if pos == (maze.shape[0]-2, maze.shape[1]-2):    
            print("goal")

        possible_direction = np.where(possibles[pos[0]][pos[1]])[0]
        
        """
        if action != None:
            possible_direction = np.concatenate([possible_direction, np.tile(possible_direction[possible_direction!=(action-2)%4],3)])
        """

        action = np.random.choice(possible_direction)
        pos = getNext(pos,action)
        
