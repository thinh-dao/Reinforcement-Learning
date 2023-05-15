# frozen-lake-ex1.py
import gym # loading the Gym library
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
 
env = gym.make("FrozenLake-v1", render_mode="human")
env.reset()                    
env.render()