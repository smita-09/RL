import gym
import JSSEnv
import warnings
warnings.filterwarnings("ignore")

def run_one_episode (env):
    env.reset()
    sum_reward = 0
    for i in range(30):
        # Just checking what happens if there is no op action is not there =?
        action = env.action_space.sample()
        print("What it is this action", action) # Taking a random action from the sample of actions
        # that is to say from the pool of actions
        state, reward, done, info = env.step(action)
        sum_reward += reward
        env.render()
        if done:
            break
    return sum_reward

env = gym.make('jss-v1')
sum_reward = run_one_episode(env)
print(sum_reward)
