from tracemalloc import start
from  JSSEnv.envs.JssEnv import JssEnv
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from models import *
from ray.rllib.utils.framework import try_import_tf
import gym
import argparse
import warnings
import ray 
import matplotlib.pyplot as plt # To plot the curve to see the progress
warnings.filterwarnings("ignore")
tf1, tf, tfv = try_import_tf()

res = []

def main (count):
    ray.init(ignore_reinit_error=True)

    # register the custom environment
    select_env = "jss-v1"

    
    # register_env(select_env, lambda config: JssEnv(config))

    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    #Adding env config 
    config['env_config'] = {
        'instance_path' : '/home/smita/Documents/RL/JSSEnv/JSSEnv/envs/instances/sh' + str(count)
    }
    register_env(select_env, lambda config: JssEnv(config))
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

    #register_env(select_env, lambda config: JssEnv(config))
    agent = ppo.PPOTrainer(config, env=select_env) # This initializes the environment

    # train a policy with RLlib using PPO
    
    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model

    # apply the trained policy in a rollout
    # Make sure to change this stuff before you make some changes
    #chkpt_file = 'tmp/exa/checkpoint_000060/checkpoint-60' 
    chkpt_file = 'tmp/exa/checkpoint_000056/checkpoint-56'
    agent.restore(chkpt_file)
    dict = { 'instance_path':'/home/smita/Documents/RL/JSSEnv/JSSEnv/envs/instances/sh' + str(count)}
    print('This is config file right here', dict['instance_path'])
    env = gym.make(select_env,env_config=dict) # This line was giving error when tried to pass the instance path in the arguments file


    state = env.reset()
    sum_reward = 0
    n_step = 250

    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward

        if done == 1:
            # report at the end of each episode
            (violation, processing_time) = env.count_processing_time_and_violations()
            res.append(tuple((violation, processing_time)))
            env.render()
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int,
                        help='an integer for the accumulator')

    args = parser.parse_args() #ge the arguments that are mentioned as params
    for i in range(1, args.integers):
        print('This is i',i)
        main(i)
        print('After the whole thing is done, this is the list of the processing time and count of violations that are happening', res)
    #main(1)