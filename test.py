from Hex import Hex
from mcts import mcts
import numpy as np

def action_int_english(action):
    table=['A','B','C','D','E','F','G','H','I','J','K']
    row=int(action/11)
    first=table[row]
    second=action%11+1
    return first,second

def action_english_int(action_array):
    table={'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10}
    first=table[action_array[0]]
    number = first * 11 + int(action_array[1]) - 1
    return number

env = Hex()
method = mcts()
state = env.reset()
while True:
    #print(state)
    action = method.choose_best_action(state)
    print(action_int_english(action))
    temp_state = state
    state = tuple(state)
    next_state, reward, done = env.step(state,action,1)
    state = next_state
    action_english = input("next action(english):")
    action_number = input("next action(number):")
    action_array = [action_english,action_number]
    action = action_english_int(action_array)
    next_state, reward, done = env.step(state,action,2)
    state = next_state
