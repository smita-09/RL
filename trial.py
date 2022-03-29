import gym
import JSSEnv
import warnings
warnings.filterwarnings("ignore")
import os
import ray 
from ray.tune.registry import register_env
from gym_example.envs.example_env import Example_v0

import shutil
chkpt_root = "tmp/exa"
shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)
ray_results = "{}/ray_results/".format(os.getenv("HOME"))
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)
ray.init(ignore_reinit_error=True)


env = gym.make('jss-v1')
action_space_size = env.action_space.n
state_space_size = env.observation_space

#  print(action_space_size)
# print(state_space_size)
print(env.render())
# print(env.__init__(env = None))  