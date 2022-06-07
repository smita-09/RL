from tracemalloc import start
import gym
import sys
from  JSSEnv.envs.JssEnv import JssEnv
import warnings
warnings.filterwarnings("ignore")
from ray.rllib.agents import with_common_config
import os
import ray 
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents import with_common_config
import shutil
from ray.rllib.models import ModelCatalog
from models import *
from ray.rllib.utils.framework import try_import_tf
import time
import matplotlib.pyplot as plt # To plot the curve to see the progress
tf1, tf, tfv = try_import_tf()

def main ():
    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    select_env = "jss-v1"
    #select_env = "fail-v1"
    register_env(select_env, lambda config: JssEnv())
    #register_env(select_env, lambda config: Fail_v1())


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    #####################
    config['metrics_smoothing_episodes'] = 2000
    config['gamma'] = 1.0
    config['layer_nb'] = 2
    config['num_envs_per_worker'] = 4
    config['rollout_fragment_length'] =704
    config['sgd_minibatch_size'] = 33000
    config['lr'] = 0.0006861
    config['lr_start'] = 0.0006861
    config['lr_end'] = 0.00007783 
    config['clip_param'] = 0.541
    config['num_sgd_iter'] = 12
    config['vf_loss_coeff'] = 0.7918
    config['kl_coeff'] = 0.496
    config['kl_target'] = 0.05047
    config['entropy_start'] = 0.0002458
    config['shuffle_sequences'] = True
    config['entropy_end'] =  0.002042   
    config['use_gae'] = True

    config['train_batch_size'] = config['sgd_minibatch_size']

    config['entropy_coeff'] = config['entropy_start']
    config['entropy_coeff_schedule'] = [[0, config['entropy_start']], [15000000, config['entropy_end']]]

    config['framework'] = 'tf'
    config["log_level"] = "DEBUG"
    config["layer_nb"] = 2
    config["layer_size"] = 319
    ModelCatalog.register_custom_model("fc_masked_model_tf", FCMaskedActionsModelTF)
    config['model'] = {
        "fcnet_activation": "relu",
        "custom_model": "fc_masked_model_tf",
        'fcnet_hiddens': [config['layer_size'] for k in range(config['layer_nb'])],
        "vf_share_layers": False,
    }
    config = with_common_config(config)
    config.pop('layer_size', None)
    config.pop('layer_nb', None) 
    config.pop('lr_start', None)
    config.pop('lr_end', None)
    config.pop('entropy_start', None)
    config.pop('entropy_end', None)

    agent = ppo.PPOTrainer(config, env=select_env)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 50

    # train a policy with RLlib using PPO
    reward_means = list()
    for n in range(n_iter):
        print('This it the iteration', n)
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)
        reward_means.append(result['episode_reward_mean'])
        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))
    print(list(range(n_iter)), reward_means)
    plt.plot(list(range(n_iter)), reward_means)
    plt.title("Curve plotted using the given points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    # print(model.base_model.summary())


    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(select_env)

    state = env.reset()
    sum_reward = 0
    n_step = 250

    for step in range(n_step):
        action = agent.compute_action(state)
        print('This is the action', action)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        if done == 1:
            # report at the end of each episode
            env.render()
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0

    

if __name__ == "__main__":
    main()