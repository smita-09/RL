"""Train and demo Reinforcement Learning agent.
To run:
    python main.py
"""


import argparse
import sys
import matplotlib.pyplot as plt
# import ujson as json
# sys.path.append('../')
# from game.tictactoe import TicTacToe
# from game.connectfour import ConnectFour
print("****"*20)
print(sys.path)
print("****"*20)
from game.chomp import Chomp
# from rl.agent import Agent


def process_args():
    """Process command line args."""
    message = 'Interactive Reinforcement Learning board game playing agent.'
    parser = argparse.ArgumentParser(description=message)

    default_game = 'tictactoe'
    default_mode = 'train'

    parser.add_argument('-g', '--game',
                        dest='game',
                        help='Board game choice.',
                        default=default_game)

    parser.add_argument('-m', '--mode',
                        dest='mode',
                        help='Mode for Agent can be train or demo.',
                        default=default_mode)

    options = parser.parse_args()
    return options



def play_chomp(mode):
    """Start Chomp game and training."""
    print('=====CHOMP=====')
    # Square board has optimal strategy to allow for easy sanity check that agent is learning.
    game = Chomp(rows=4, cols=4)
    if mode == 'train':
        # Train agent to go first
        agent = Agent(game, epsilon=9e-3, learning_rate=25e-2)
        n = 10000
        history = agent.train(n)
        print('After {} Episodes'.format(n))

        # Plot Reward Stats
        rfig, raxs = plt.subplots(nrows=3, ncols=1)
        rax_reward1 = raxs[0]
        rax_reward1.grid()
        rax_reward2 = raxs[1]
        rax_reward2.grid()
        rax_reward3 = raxs[2]
        rax_reward3.grid()

        rax_reward1.plot(history[0][:100], history[1][:100])
        rax_reward1.set(ylabel='Cumulative Reward', title='Chomp 4x4 Cumulative Reward')

        rax_reward2.plot(history[0][:1000], history[1][:1000], color='g')
        rax_reward2.set(ylabel='Cumulative Reward')

        rax_reward3.plot(history[0][:n], history[1][:n], color='r')
        rax_reward3.set(xlabel='Episode', ylabel='Cumulative Reward')

        rfig.savefig('chomp_reward.png')

        # Plot Qtable Memory Usage Stats
        memfig, memaxs = plt.subplots(nrows=3, ncols=1)
        memax_reward1 = memaxs[0]
        memax_reward1.grid()
        memax_reward2 = memaxs[1]
        memax_reward2.grid()
        memax_reward3 = memaxs[2]
        memax_reward3.grid()

        memax_reward1.plot(history[0][:100], history[2][:100])
        memax_reward1.set(ylabel='Size (KB)', title='Chomp 4x4 QTable Size')

        memax_reward2.plot(history[0][:1000], history[2][:1000], color='g')
        memax_reward2.set(ylabel='Size (KB)')

        memax_reward3.plot(history[0][:n], history[2][:n], color='r')
        memax_reward3.set(xlabel='Episode', ylabel='Size (KB)')
        plt.show()

        # agent.save_values(path='data/chomp_qtable.json')
        agent.demo()

    elif mode == 'hyper':
        # Hyper parameter optimization
        max_e = 0.0
        max_lr = 0.0
        max_reward = 0.0
        epsilons = [1e-1, 2e-1, 9e-2, 1e-2, 9e-3]
        learning_rates = [1e-1, 2e-1, 3e-1, 25e-2, 9e-2]
        for epsilon in epsilons:
            for learning_rate in learning_rates:
                agent = Agent(game, qtable={}, player='X', epsilon=epsilon, learning_rate=learning_rate)
                n = 10000
                history = agent.train(n, history=[])
                total = history[1][len(history[1]) - 1]
                print(total)
                if total > max_reward:
                    max_reward = total
                    max_e = epsilon
                    max_lr = learning_rate
        print('Max e: {}'.format(max_e))
        print('Max lr: {}'.format(max_lr))
        print('Max reward: {}'.format(max_reward))

    elif mode == 'demo':
        # qtable = json.load(open('data/chomp_qtable.json'))
        # agent = Agent(game, qtable=qtable)
        agent.demo()
    else:
        print('Mode {} is invalid.'.format(mode))


def main():
    """Entry point."""
    options = process_args()
    if options.game == 'chomp':
        play_chomp(options.mode)
    else:
        print('Game choice {} is current unsupported.'
              .format(options.game))
        sys.exit(1)


if __name__ == '__main__':
    main()