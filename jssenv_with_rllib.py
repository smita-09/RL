import gym
import JSSEnv
import warnings
warnings.filterwarnings("ignore")

def run_one_episode (env, verbose=False):
    env.reset() 
    sum_reward = 0
    
    actions = [1, 2, 1, 0, 2, 0, 3, 0, 1, 2] # 0.79, 1, 1.4, -1.0004, -2.199
    actions = [1, 2, 1, 0, 2, 0, 2, 1, 0] # 2.39, 2.4, 2.4, -1.20, -2.6000
    # actions = [1, 2, 1, 0, 2, 0, 1, 0, 3, 2] # Around 1, .., .., -1.40
    # Not taking the no action in account do not know why 
    for i in range(10):
        action = actions[i]
        print('This is the action and the iteration',action, i)

        if verbose:
            print("action:", action)

        state, reward, done, info = env.step(action)
        sum_reward += reward

        if done:
            if verbose:
                print("done @ step {}".format(i))
                env.render()
            break
    

    # Letting the action picked randomly and not actually creating a pre defined list of actions to be picked.
    # for i in range(30):
    #     action = env.action_space.sample() # There are 3 jobs so 3 actions that could be alocated aas of now  
    #     print('This is the action and the iteration',action, i)

    #     if verbose:
    #         print("action:", action)

    #     state, reward, done, info = env.step(action)
    #     sum_reward += reward

    #     if done:
    #         if verbose:
    #             print("done @ step {}".format(i))
    #             env.render()
    #         break

    if verbose:
        print("cumulative reward", sum_reward)
    env.render()
    return sum_reward


def main ():
    # first, create the custom environment and run it for one episode
    env = gym.make('jss-v1')
    sum_reward = run_one_episode(env, verbose=True)

    # next, calculate a baseline of rewards based on random actions
    # (no policy)
    history = []

    # for _ in range(1):
    #     sum_reward = run_one_episode(env, verbose=False)
    #     history.append(sum_reward)

    # avg_sum_reward = sum(history) / len(history)
    # print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))


if __name__ == "__main__":
    main()



# 3 3 
# 0 7 2 8 1 10 25
# 1 6 0 4 2 12 30
# 0 8 1 8 2 7 35