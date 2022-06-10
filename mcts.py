from reinforcement import Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Hex import Hex


class mcts():
    def __init__(self,board_size=11,eta=0.7,searching=100):
        self.n_actions = board_size*board_size
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.evaluate_state_net = Net(self.n_actions)  # the evaluate network
        self.eta = eta
        self.board_size = board_size
        self.searching = searching

    def standardize(self,data):
        return (data - min(data)) / (max(data) - min(data))

    def choose_best_action(self,state):
        player=1
        state_ = tuple(state)
        self.evaluate_net.load_state_dict(torch.load("./Tables/DQN.pt"))
        #self.evaluate_state_net.load_state_dict(torch.load("./Tables/policy.pt"))
        chosen_times = np.zeros(self.board_size*self.board_size)
        q_value=self.evaluate_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
        score = np.zeros(self.board_size*self.board_size)
        #print(state)
        for i in range(self.board_size*self.board_size):
            if(i%20==0):
                print(i)
            #print(state)
            score[i]=self.eta*(q_value[i]/(1+chosen_times[i]))
            #state = np.array(state_)
        
        
        for i in range(self.searching):
            if(i%20==0):
                print(i)
            action = np.argmax(score)
            chosen_times[action]=chosen_times[action]+1
            score[action]=self.value_after_thinking(state,action,player)+self.eta*(q_value[action]/(1+chosen_times[action]))
            state = np.array(state_)
        score_temp = score
        action=0
        #print(state)
        while True:
            action = np.argmax(score_temp)
            #print(action)
            #print(state[action])
            if(state[action]==0):
                break
            else:
                score_temp[action]=np.min(score_temp)

        return action


        
    def value_after_thinking(self,state,action,player):
        #print(state)
        # original_state = np.zeros(121)
        # for i in range(121):
        #     original_state[i] = state[i]
        original_state = state
        #state_ = tuple(state)
        #state.flags.writeable = False
        env = Hex()
        current_state,_,_ = env.step(original_state,action,player)
        rt = 0
        #print(state)
        while True:
            if(player==1):
                player=2
            else:
                player=1
            q_value = self.evaluate_net.forward(
                torch.FloatTensor(current_state)).squeeze(0).detach()
            weighted = self.standardize(q_value)
            #print(q_value)
            #print(np.array(weighted))
            current_action = np.random.choice(121,1,np.array(weighted).any())
            current_state, reward, done = env.step(current_state,current_action,player)
            if(done==1):
                rt = 1
                break
            if(done==2):
                rt = -1
                break
        #print(state)
        q_value = self.evaluate_net.forward(
                torch.FloatTensor(original_state)).squeeze(0).detach()
        weighted = self.standardize(q_value)
        q_value_state = 0
        for i in range(len(q_value)):
            q_value_state = q_value_state + weighted[i] * q_value[i]
        
        #state = np.array(state_)
        #print(state)
        
        return 1/2*(rt + q_value_state)
    

            

        


  