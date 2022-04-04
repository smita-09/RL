import gym
import JSSEnv
import warnings
warnings.filterwarnings("ignore")

def run_one_episode (env, verbose=False):
    env.reset() 
    sum_reward = 0

    for i in range(30):
        action = env.action_space.sample() # There are 3 jobs so 3 actions that could be alocated aas of now  
        print('This is the action and the iteration',action, i)

        if verbose:
            print("action:", action)

        state, reward, done, info = env.step(action)
        sum_reward += reward

        # if verbose:
        #     env.render()

        if done:
            if verbose:
                print("done @ step {}".format(i))
                env.render()
            break
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