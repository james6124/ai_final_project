import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from tqdm import tqdm
from Hex import Hex
import csv
from game import game
#import tensorflow as tf
total_rewards = []
min=10000
min_state=10000


class replay_buffer():
    '''
    A deque storing trajectories
    '''

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.

        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish
        
        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.

        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''

    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 121  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        '''
        Forward the state to the neural network.
        
        Parameter:
            states: a batch size of states
        
        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000, valid=1):
        """
        The agent learning how to control the action of the cart pole.
        
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.valid=valid
        self.min=min
        self.env = env
        self.n_actions = 121  # the number of actions
        self.count = 0  # recording the number of iterations

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.target_net = Net(self.n_actions)  # the target network
        self.evaluate_state_net = Net(1)

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network
        self.optimizer_state = torch.optim.Adam(
            self.evaluate_state_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def standardize(self,data):
        return (data - min(data)) / (max(data) - min(data))

    def learn(self):
        '''
        - Implement the learning function.
        - Here are the hints to implement.
        
        Steps:
        -----
        1. Update target net by current net every 100 times. (we have done for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        -----
        
        Parameters:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        
        Returns:
            None (Don't need to return anything)
        '''
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        if len(self.buffer.memory) < self.batch_size:
            return
        b_state,b_action,b_reward,b_next_state,b_done=self.buffer.sample(self.batch_size)
        b_state = torch.tensor(np.array(b_state), dtype=torch.float32)
        b_action = torch.tensor(b_action, dtype=torch.long)  
        b_reward = torch.tensor(b_reward, dtype=torch.float32)
        b_next_state = torch.tensor(np.array(b_next_state), dtype=torch.float32)
        b_done = torch.tensor(b_done, dtype=torch.float32)
        batch_indices = np.arange(self.batch_size, dtype=np.int64) #np.arange(3) batch_indices=0,1,2,

        q_values = self.evaluate_net.forward(b_state)
        next_q_values = self.target_net.forward(b_next_state)
        predicted_value_of_now = q_values[batch_indices, b_action]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]
        q_target = b_reward + self.gamma * predicted_value_of_future * b_done
        criterion1 = nn.MSELoss()
        loss = criterion1(predicted_value_of_now, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        #if self.count % 100 == 0:
            #print(torch.max(q_values))
        

        
        # q_value_state = self.evaluate_state_net.forward(b_state)
        # #print(torch.max(q_value_state))
        # q_value_standardize = self.standardize(q_values)
        # q_target_state = np.zeros(self.batch_size)
        # q_target_state = np.reshape(q_target_state,(32,1))
        # q_target_state = torch.from_numpy(q_target_state)
        # torch.reshape(q_target_state,(32,1))
        # for i in range(self.batch_size):
        #     for j in range(len(q_value_state)):
        #         q_target_state[i] = q_target_state[i] + q_value_standardize[i][j] * q_values[i][j]
        # #torch.set_default_dtype(torch.float32)
        # q_value_state = q_value_state.float()
        # q_target_state = q_target_state.float()
        # criterion2 = nn.MSELoss()
        # #q_target_state = q_target_state.astype(np.float32)
        # loss_state = criterion2( q_value_state[batch_indices] , q_target_state[batch_indices] )
        # self.optimizer_state.zero_grad()
        # loss_state.backward()
        # self.optimizer_state.step()
        self.optimizer.step()


        #print(self.target_net.forward(b_state))
        # End your code

        # You can add some conditions to decide when to save your neural network model
        # if self.buffer.memory:
        #if len(total_rewards)>0 and np.mean(total_rewards)>200:
        #    torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")
        global min
        if loss < min:
            torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")
            min=loss

        # global min_state
        # if loss_state < min_state:
        #     torch.save(self.evaluate_state_net.state_dict(), "./Tables/policy.pt")
        #     min=loss

    def choose_action(self, state):
        """
        - Implement the action-choosing function.
        - Choose the best action with given state and epsilon
        
        Parameters:
            self: the agent itself.
            state: the current state of the enviornment.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        
        Returns:
            action: the chosen action.
        """
        with torch.no_grad():
            while True:
                if random.random() < self.epsilon:
                    action=0
                    while True:
                        action=np.random.rand(1)*121
                        action=int(action)
                        if(state[action]==0):
                            break
                    return action
                state1 = torch.tensor(state).float().detach()
                state1 = state1.to(torch.device("cpu"))
                state1 = state1.unsqueeze(0)
                q_values = self.target_net(state1)
                if(state[torch.argmax(q_values).item()]==0):
                    break
        return torch.argmax(q_values).item()
        # return action

    def check_max_Q(self):
        """
        - Implement the function calculating the max Q value of initial state(self.env.reset()).
        - Check the max Q value of initial state
        
        Parameter:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        
        Return:
            max_q: the max Q value of initial state(self.env.reset())
        """
        # Begin your code
        
        initial_state = torch.unsqueeze(torch.FloatTensor(self.env.reset()),0)
        return max(max(self.target_net(initial_state)))

        # End your code


def train(env):
    """
    Train the agent on the given environment.
    
    Paramenters:
        env: the given environment.
    
    Returns:
        None (Don't need to return anything)
    """
    global min
    min=10000
    agent = Agent(env)
    Game = game()
    episode = 1000
    rewards = []
    count=1
    for i in tqdm(range(episode)):
        all_action=[]
        state = env.reset()
        #count = 0
        player=2
        while True:
            
            #count += 1
            #print(count)
            agent.count += 1
            if(player==1):
                player=2
            else:
                player=1
            #all_state.append(list(state))
            action = agent.choose_action(state)

            sub_action=[]
            first,second=action_int(action)
            sub_action.append(first)
            sub_action.append(second)
            all_action.append(sub_action)
            #print(state)
            #print(all_state)
            next_state, reward, done = env.step(state,action,player)
            agent.buffer.insert(state, action, reward,
                                next_state, done)
            #if agent.count >= 1000:
            if(player==1):
                agent.learn()
            print(done)
            if done>0:
                #rewards.append(count)
                break
            state = next_state
            
        count+=1
        #print(count)
        #print(len(all_action))
        #if(count>300):
        Game.display(all_action)
        #f = open('state.csv', 'w')

        # create the csv writer
        #writer = csv.writer(f)

        # write a row to the csv file
        #writer.writerows(all_state)

        # close the file
        #f.close()
    #print(f"reward: {np.mean(rewards)}")
    #total_rewards.append(rewards)


def test(env):
    """
    Test the agent on the given environment.
    
    Paramenters:
        env: the given environment.
    
    Returns:
        None (Don't need to return anything)
    """
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    for _ in range(100):
        state = env.reset()
        count = 0
        while True:
            count += 1
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, _, done, _ = env.step(action)
            if done:
                rewards.append(count)
                break
            state = next_state
    #print(f"reward: {np.mean(rewards)}")
    #print(f"max Q:{testing_agent.check_max_Q()}")

def action_int(action):
    #table=['A','B','C','D','E','F','G','H','I','J','K']
    row=int(action/11)
    first=row
    second=action%11
    return first,second


if __name__ == "__main__":
    '''
    The main funtion
    '''
    env = Hex()

    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    # training section:
    for i in range(1):
        print(f"#{i + 1} training progress")
        train(env)
    # testing section:
    #test(env)
    
    if not os.path.exists("./Rewards"):
        os.mkdir("./Rewards")

    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))

    env.close()
