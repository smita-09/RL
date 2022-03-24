import gym
import JSSEnv
import warnings
warnings.filterwarnings("ignore")

env = gym.make('jss-v1')
action_space_size = env.action_space.n
state_space_size = env.observation_space

#  print(action_space_size)
# print(state_space_size)
print(env.render())
# print(env.__init__(env = None))