import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DIRECTION = {0:(-1,0),1:(0,1),2:(1,0),3:(0,-1)}

def initilize(w,h):
    """
    Parameters
    --------------------
    w,h : int
        迷路の縦，横のサイズ
    
    Returns
    --------------------
    maze : ndarray
        迷路を表す配列 (2w+1,2h+1)
        0が道，1が壁
    """

    assert w>1 and h>1, "迷路が小さすぎます"

    # 外縁が壁（1），内側はの配列を用意
    maze = np.ones((2*w+1,2*h+1))
    maze[1:-1,1:-1] = 0#np.zeros(2*w-1,2*h-1)    
    maze[2:-2:2,2:-2:2] =1

    return maze

def generate(maze):
    """
    Parameters
    Returns
    -----------------
    maze : ndarray
        迷路を表す配列
    """
    
    W,H = maze.shape
    for wi in range(2,W-1,2):
        for hi in range(2,H-1,2):
            
            # maze[wi][hi]の上下左右で壁でない方向を取得
            if wi == 2:
                possibles = possibleDirection(maze,wi,hi)
            else:    
                possibles = np.delete(possibleDirection(maze,wi,hi),0)
            
            # 壁でない方向から一つを選択
            selected = np.random.choice(possibles)
            
            # 選択した箇所を壁にする
            maze[wi+DIRECTION[selected][0]][hi+DIRECTION[selected][1]] = 1
                 
    return maze


def possibleDirection(maze,wi,hi):    
    return np.where([True if maze[wi+DIRECTION[i][0]][hi+DIRECTION[i][1]]==0 else False for i in range(4)])[0]


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


class Solver():
    
    def __init__(self, maze):
        
        self.maze = maze
        self.goal = (maze.shape[0]-2,maze.shape[1]-2)

        
    def find(self, w=1, h=1, to=None):
        
        if w==self.goal[0] and h==self.goal[1]:
            return [(w,h)]
        
        minimum_result = None
        
        
        length = np.inf
        for d in possibleDirection(self.maze,w,h):
            
            # 戻る方向でなければ
            if to==None or d!=(to-2)%4:
                
                wd, hd = DIRECTION[d]
                result = self.find(w+wd,h+hd,d)
                      
                #行き詰まりでなく，かつ現在見つけている経路よりも短い
                if result != None and len(result) < length:
                    minimum_result = [(w,h)] + result
                    length = len(result)
        
        return minimum_result
    
    def solve(self):
        
        result = self.find(w=self.maze.shape[0]-2)
        maze_copy= self.maze.copy()
        
        for cell in result:
            maze_copy[cell[0]][cell[1]] = 0.2
        
        return maze_copy


if __name__=="__main__":
    
    fig = plt.figure(figsize=(10,10))
    
    maze = initilize(20,40)
    maze = generate(maze)
    
    ax = fig.add_subplot(2,1,1)
    visualize(maze,ax)
    
    solver = Solver(maze)
    result = solver.solve()
    
    ax = fig.add_subplot(2,1,2)
    visualize(result,ax)
    
    plt.show()