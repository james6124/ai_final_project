import os
from  Hex import Hex
import numpy as np
import networkx as nx

class MultiAgent():
    def __init__(self, board_size=11, depth=2):
        self.board_size = board_size
        self.depth = depth
        self.winner = 0

    def init_graph(self, agentIndex):
        G = nx.Graph()
        # add node
        G.add_nodes_from(list(range(self.board_size * self.board_size)))
        G.add_nodes_from([1000, 1001])
        # add edges
        for i in range(self.board_size):
            for j in range(self.board_size):
                current = i * self.board_size + j
                if (j < self.board_size - 1):
                    G.add_edge(current, current + 1)
                if (i < self.board_size - 1):
                    G.add_edge(current, current + self.board_size)
                if (i < self.board_size - 1 and j > 0):
                    G.add_edge(current, current + self.board_size - 1)
        #  border edge
        if agentIndex == -1:
            for j in range(self.board_size):
                G.add_edge(1000, j)
                G.add_edge(1001, j + self.board_size * (self.board_size - 1))
        elif agentIndex == 1:
            for i in range(0, self.board_size * (self.board_size - 1) + 1, self.board_size):
                G.add_edge(1000, i)
                G.add_edge(1001, i + self.board_size - 1)
        return G
    # ---------------------------------------------------------------------------- #
    #                                 find the path                                #
    # ---------------------------------------------------------------------------- #

    # count heuristic value
    def getHeuristicScore(self, state):
        # if the connection established 
        G1 = self.state2graph(state, 1)
        G2 = self.state2graph(state, -1)
        agent_len1 = self.getShortestPathLength(G1) - 1
        agent_len2 = self.getShortestPathLength(G2) - 1
        if(agent_len1 == 0):
            self.winner = 1
        elif(agent_len2 == 0):
            self.winner = -1
        else:
            self.winner = 0
        return agent_len1 * 1.2 - agent_len2

    def state2graph(self, state, agentIndex):
        G = self.init_graph(agentIndex)
        for i in range(self.board_size * self.board_size):
            if (state[i] == agentIndex):
                adj_list = G.adj[i].items()
                for nbr, datadict in adj_list:
                    for other_nbr, other_datadict in adj_list:
                        if nbr == other_nbr or nbr == i or other_nbr == i:
                            continue
                        else:
                            G.add_edge(nbr, other_nbr)
            if (state[i] != 0):
                G.remove_node(i)
        return G

    # Return shortest path length. If path no exist, return -1.
    def getShortestPathLength(self, G):
        if nx.has_path(G, 1000, 1001):
            spl = nx.shortest_path_length(G, 1000, 1001)
            return spl
        else:
            return -1

    # ---------------------------------------------------------------------------- #
    #                               minimax algorithm                              #
    # ---------------------------------------------------------------------------- #
    def getNextState(self, state, action, agentIndex):
        if agentIndex:
            next_state = []
            for item in state:
                next_state.append(item)
            next_state[action] = agentIndex
        else:
            print("not a player")
        # print("new state evaluated")
        return next_state

    def getMiniMaxAction(self, state):
        """
        return a postion that I should put
        """ 
        def mini_value(state, depth):
            # print("mini")
            Done = False
            if (depth == self.depth):
                Done = True
            if (self.winner != 0):
                Done = True
            if (Done): 
                heuristicScore = self.getHeuristicScore(state)
                print(heuristicScore)
                return heuristicScore

            miniEval = float('inf')
            for action, v in enumerate(state):
                if (v == 0): # this position is empty
                    child = self.getNextState(state, action, 1)
                    val = max_value(child, depth + 1)
                    miniEval = min(val, miniEval)
            return miniEval
        def max_value(state, depth):
            # print("max")
            Done = False
            if (depth == self.depth):
                Done = True
            if (self.winner != 0):
                Done = True
            if (Done): 
                heuristicScore = self.getHeuristicScore(state)
                print(heuristicScore)
                return heuristicScore

            maxEval = -float('inf')
            for action, v in enumerate(state):
                if(v == 0):
                    child = self.getNextState(state, action, -1)
                    v = mini_value(child, depth + 1)
                    maxEval = max(v, maxEval)
            return maxEval

        # StateVal = self.getHeuristicScore(state)
        # if (StateVal == 0 and self.winner == 0):
        #     bestAction = 50
        #     # child = self.getNextState(state, bestAction, 1)
        #     return bestAction

        # from the top starting to travel
        maxValue = -float('inf')
        for action, v in enumerate(state):
            if v == 0:
                child = self.getNextState(state, action, -1)
                v = mini_value(child, 1)
                # print(action, v)
                if v > maxValue:
                    bestAction = action
                    maxValue = v
        return bestAction
        
    # def getAlphaBetaAction(self, state):
    #     #  AgentIndex == 2 : rival
    #     def mini_value(state, depth=0, agentIndex=2, alpha=-float('inf'), beta=float('inf')):
    #         # print("mini")
    #         miniEval = float('inf')
    #         for action, val in enumerate(state):
    #             if(val == 0): # the position is empty
    #                 child = self.getNextState(state, action, 2)
    #                 v = max_value(child, depth+1, 1)[0]
    #                 if v < miniEval:
    #                     miniEval = v
    #                     bestAction = action
    #                 if v < alpha :
    #                     return v, action
                    
    #         # print("mini", miniEval)
    #         return miniEval, bestAction
    #     # AgentIndex == 1 : user
    #     def max_value(state, depth=0, agentIndex=1, alpha=-float('inf'), beta=float('inf')):
    #         # print("max")
    #         maxEval = -float('inf')
    #         for action, val in enumerate(state):
    #             if(val == 0):
    #                 child = self.getNextState(state, action, 1)
    #                 v = mini_value(child, depth+1, 2)[0]
    #                 if b < maxEval:
    #                     maxEval = action
    #                     bestAction = action
    #                 if v > beta:
    #                     return v, bestAction
    #         # print("max",maxEval)
    #         return maxEval, bestAction


    #     min_val = mini_value(state)
    #     print(min_val)
    #     bestAction = 0
    #     return bestAction
        
# check value
# env = Hex()
# state = env.reset()
# # check if the state is done need to do it outside
# if env.is_done(state):
#     print("done")
# # print(state)
# method = MultiAgent()

# action = method.getMiniMaxAction(state)
# print(action)

# for i in range(0, 89, 11):
#     state[i] = 1

# action = method.getMiniMaxAction(state)
# print(action)