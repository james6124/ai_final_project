from Hex import Hex
from mcts import mcts
import numpy as np
from internet_hex_bot import hex_bot

hexbot = hex_bot()
hexbot.new_game(1, 3, 0)

env = Hex()
method = mcts()
state = env.reset()

action = method.choose_best_action(state)
print(int(action))

while not hexbot.is_done():
    action = method.choose_best_action(state)
    botmove = hexbot.agent_put(int(action), 0)
    
    next_state, reward, done = env.step(state,action,1)
    state = next_state

    next_state, reward, done = env.step(state,botmove,2)
    state = next_state
