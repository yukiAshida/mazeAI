import numpy as np
import matplotlib.pyplot as plt
from mazeGenarator import initilize
from mazeGenarator import generate
from mazeGenarator import visualize
from mazeGenarator import Solver
from utils import DIRECTION
from utils import visualize, plot

import traceback


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

def ipsilonGreedy(q, possible_boolean, e=5):

    max_action = np.argmax( np.where(possible_boolean,q,-np.inf) )
    max_value = np.max( q[possible_boolean] )
    min_value = np.min( q[possible_boolean] )

    if max_value==min_value:
        return np.where(possible_boolean)[0]
    else:
        m = len( q[possible_boolean] )-1

        major = int(100-e)
        minor = 0 if m==0 else int(e//m)
        
        ipsilon_accommodated = np.concatenate([ [x]*major if x==max_action else [x]*minor for x in np.where(possible_boolean)[0] ]).astype(np.int64)
        #print(ipsilon_accommodated)
        return ipsilon_accommodated

        
def BVF():

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)    

    maze = initilize(15,15)
    maze = generate(maze)
    possibles = possibleAction(maze)
    
    solver = Solver(maze)
    _,best_length = solver.solve()
    
    Q = np.zeros((maze.shape[0], maze.shape[1], 4)) 

    length_list = []
    alpha = 0.1
    gamma = 10
    n_iter = 1000

    try:

        for i in range(n_iter):
            
            # 可視化用
            visual = maze.copy()
            length = 0

            # 初期化
            pos = (maze.shape[0]-2,1)
            action = None

            while True:
                
                visual[pos[0]][pos[1]] = min(visual[pos[0]][pos[1]]+0.1, 0.5)
                length += 1

                # 選択可能行動にイプシロン-グリーディーを適用
                possible_boolean = possibles[pos[0]][pos[1]]
                possible_action = ipsilonGreedy(Q[pos[0]][pos[1]], possible_boolean)
                
                
                # 行動を選択
                action = np.random.choice(possible_action)
                pre_pos = pos
                pos = getNext(pos,action)

                # 行動価値関数を更新
                goal = pos == (maze.shape[0]-2, maze.shape[1]-2)
                dQ = int(goal)*(i+1) + gamma*max(Q[pos[0]][pos[1]]) - Q[pre_pos[0]][pre_pos[1]][action]
                Q[pre_pos[0]][pre_pos[1]][action] = alpha*dQ
                
                if goal:
                    length_list.append(length)

                    visualize(visual,ax1)
                    plot(length_list,best_length,ax2,n_iter)

                    plt.pause(0.01)
                    break

        print(length_list)
        plt.show()

    except:
        traceback.print_exc()
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1)
        visualize(maze,ax)
        plt.show()

if __name__=="__main__":

    BVF()
