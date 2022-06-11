import os
from  Hex import Hex
import numpy as np

i = 1
class MultiAgent():
    def __init__(self, board_size=11):
        self.board_size = board_size
    
    # ---------------------------------------------------------------------------- #
    #                                 find the path                                #
    # ---------------------------------------------------------------------------- #

    # count heuristic value
    def getHeuristicScore(self):
        # if the connection established 
        Agent_1 = self.getShortestPath(1)
        Agent_2 = self.getShortestPath(2)
        AgentScore_1 = self.getPathScore(Agent_1)
        AgentScore_2 = self.getPathScore(Agent_2) 
        if(AgentScore_1 == 0):
            winner = 1
        elif(AgentScore_2 == 0):
            winner = 2 
        else:
            winner = 0
        return AgentScore_1 - AgentScore_2, winner

    def getShortestPath(self, state, agentIndex):
        """
        input : state
        output : a list with path data (0.1.2)
        """
        # find one on the map
        # check if done
        if(agentIndex == 1):
            for i in state:
                if(i == 0):
                    return 0
        return 0

    def getPathScore(self, path, agentIndex):
        ExpectedSteps = 0
        for node in path:
            if(node == agentIndex):
                ExpectedSteps = ExpectedSteps + 1
        return ExpectedSteps

    def getNeighborPos(self, pos):
        up = pos - self.board_size
        down = pos + self.board_size
        right = pos + 1
        left = pos - 1
        upleft = pos - self.board_size -1 
        downright = pos + self.board_size +1
        # check boundary
        positions = [up, down, right, left, upleft, downright]
         
    # ---------------------------------------------------------------------------- #
    #                               minimax algorithm                              #
    # ---------------------------------------------------------------------------- #
    def getNextState(self, state, action, agentIndex):
        if(agentIndex == 1):
            state[action] = 1
        elif(agentIndex == 2):
            state[action] = 2
        else:
            print("not a player")
        return state

    def getMiniMaxAction(self, state):
        """
        return a postion that I should put
        """ 
        def mini_value(state, depth=0, agentIndex=2):
            global i
            i+=1
            print("mini")
            miniEval = float('inf')
            for action, val in enumerate(state):
                if(val == 0): # the position is empty
                    child = self.getNextState(state, action, 2)
                    v = max_value(child, depth+1, 1)
                    miniEval = min(v, miniEval)
                    
            print("mini", miniEval)
            return miniEval
        def max_value(state, depth=0, agentIndex=1):
            global i
            i+=1
            print("max")
            maxEval = -float('inf')
            for action, val in enumerate(state):
                if(val == 0):
                    child = self.getNextState(state, action, 1)
                    v = mini_value(child, depth+1, 2)
                    maxEval = max(v, maxEval)
            print("max",maxEval)
            return maxEval


        min_val = mini_value(state)
        print(min_val)
        print(i)
        # bestAction = min_val
        # maxValue = -float('inf')
        # i = 2
        # bestAction = 0
        # # randint = np.random.randint(120)
        # # for action, val in enumerate(state):
        #     # print(action, val)
        #     # agentIdx = (i + 1) % 2
        # agentIdx = 0
        # child = self.getNextState(state, 0, 2)
        # print(child)
        # v = mini_value(child, 0, agentIdx)
        # if v > maxValue:
        #     maxValue = v
        #     bestAction = action

        # yes
        # a = self.getNextState(state, 1, 2) 
        # print(a)
        bestAction = 2 
        return bestAction

    # def getAlphaBetaAction(self, state):
    #     #  AgentIndex == 2 : rival
    #     def mini_value(state, depth=0, agentIndex=2, alpha=-float('inf'), beta=float('inf')):
    #         # print("mini")
    #         miniEval = float('inf')
    #         for action, val in enumerate(state):
    #             if(val == 0): # the position is empty
    #                 child = self.getNextState(state, action, 2)
    #                 v = max_value(child, depth+1, 1)
    #                 miniEval = min(v, miniEval)
                    
    #         # print("mini", miniEval)
    #         return miniEval
    #     # AgentIndex == 1 : user
    #     def max_value(state, depth=0, agentIndex=1, alpha=-float('inf'), beta=float('inf')):
    #         # print("max")
    #         maxEval = -float('inf')
    #         for action, val in enumerate(state):
    #             if(val == 0):
    #                 child = self.getNextState(state, action, 1)
    #                 v = mini_value(child, depth+1, 2)
    #                 maxEval = max(v, maxEval)
    #         # print("max",maxEval)
    #         return maxEval

    #     min_val = mini_value(state)
    #     print(min_val)
    #     bestAction = 0
    #     return bestAction
        


# check value
env = Hex()
state = env.reset()
# check if the state is done need to do it outside
if env.is_done(state):
    print("done")
# print(state)
method = MultiAgent()
action = method.getMiniMaxAction(state)
print(action)

# while True:
    # print(state)
    # state[20]=1
    # print(state)

    # action = method.getAction(state)

    # print(action)

    # a = method.getNextState(state, 1, 1)

# class AlphaBetaAgent():
#     def getAction(self, state):
    

