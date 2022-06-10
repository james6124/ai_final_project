import numpy as np

class Hex():
    def __init__(self,broad_size=11,batch_size=32):
        self.broad_size = broad_size
        self.batch_size = batch_size
        self.board=self.init_board() # adjency list
    def reset(self):
        #ans=[]
        #for i in range(self.batch_size):
        temp=np.zeros(self.broad_size*self.broad_size)
            #ans.append(temp)
        return np.array(temp)

    def is_done(self,state):
        #print(self.board)
        boundary1=[0,1,2,3,4,5,6,7,8,9,10]
        boundary2=[10,11,22,33,44,55,66,77,88,99,110]
        for player in range(1,3):
            if(player==1):
                for node in boundary1:
                    #print(node)
                    visited=[]
                    done = self.dfs(player,state,visited,self.board,node)
                    if(done!=0):
                        return done
            if(player==2):
                for node in boundary2:
                    #print(node)
                    visited=[]
                    done = self.dfs(player,state,visited,self.board,node)
                    if(done!=0):
                        return done
        return 0

    def dfs(self,player,state, visited, board, node):  #function for dfs 
        #print(visited)
        if node not in visited:
            visited.append(node)
            if(state[node]!=player):
                return 0
            for neighbour in board[node]:
                #print(neighbour)
                done = self.dfs(player, state, visited, board, neighbour)
                if(done!=0):
                    return done
            if(state[node]==player and ((player==1 and node>=110 and node<=120)or(player==2 and node%11==10))):
                return player
            else:
                return 0
        else:
            return 0


    def step(self ,state ,action, who_is_playing): #一定要是對的action
        temp_state = state
        temp_state[action]=who_is_playing
        done=self.is_done(temp_state)
        reward=0
        # if(who_is_playing==done):
        #     reward=1
        # elif(done!=0 and who_is_playing!=done):
        #     reward=-1
        #print(reward)
        if(done==1):
            reward=1
        elif(done==2):
            reward==-1
        return temp_state, reward, done
    
    def init_board(self):
        board=[]
        up=[0,1,2,3,4,5,6,7,8,9,10]
        left=[0,11,22,33,44,55,66,77,88,99,110]
        right=[10,21,32,43,54,65,76,87,98,109,120]
        down=[110,111,112,113,114,115,116,117,118,119,120]
        for i in range(self.broad_size*self.broad_size):
            node=[]
            if(i==0):
                node.append(1)
                node.append(11)
                board.append(node)
            elif(i==10):
                node.append(9)
                node.append(20)
                node.append(21)
                board.append(node)
            elif(i==110):
                node.append(99)
                node.append(100)
                node.append(111)
                board.append(node)
            elif(i==120):
                node.append(109)
                node.append(119)
                board.append(node)
            elif(i in up):
                node.append(i-1)
                node.append(i+1)
                node.append(i+10)
                node.append(i+11)
                board.append(node)
            elif(i in left):
                node.append(i-11)
                node.append(i-10)
                node.append(i+1)
                node.append(i+11)
                board.append(node)
            elif(i in right):
                node.append(i-11)
                node.append(i-1)
                node.append(i+10)
                node.append(i+11)
                board.append(node)
            elif(i in down):
                node.append(i-1)
                node.append(i+1)
                node.append(i-10)
                node.append(i-11)
                board.append(node)
            else:
                node.append(i-1)
                node.append(i+1)
                node.append(i-10)
                node.append(i-11)
                node.append(i+10)
                node.append(i+11)
                board.append(node)
        return board
    


    
    


        

