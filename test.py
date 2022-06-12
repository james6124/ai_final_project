from distutils.command.build_scripts import first_line_re
from Hex import Hex
from mcts import mcts
import numpy as np
from internet_hex_bot import hex_bot
from baseline import MultiAgent

hexbot = hex_bot(1, 2, 0)

env = Hex()
# method = mcts()
method = MultiAgent()
state = env.reset()
first_flag = 1
while not hexbot.is_done():
    if first_flag:
        action = 60
        first_flag = 0
        botmove = hexbot.agent_put(int(action), 0)
        print("agent action: ", action)
        print("bot action: ", botmove)
    else:
        action = method.getMiniMaxAction(state)
        botmove = hexbot.agent_put(int(action), 0)
        print("agent action: ", action)
        print("bot action: ", botmove)
    
    next_state, reward, done = env.step(state,action,1)
    state = next_state

    next_state, reward, done = env.step(state,botmove,-1)
    state = next_state
