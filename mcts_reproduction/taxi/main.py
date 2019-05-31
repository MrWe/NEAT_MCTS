

from __future__ import print_function
import gym
import numpy as np
from gym import envs
import sys
sys.path.append('..')
import neat


env = gym.make("FrozenLake-v0")
observation = env.reset()

def eval_genomes(genomes, config):
    observation = env.reset()
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        for __ in range(20):
          for _ in range(200):
              action = net.activate([observation])
              high = action.index(max(action))
              observation, reward, done, info = env.step(high)
              genome.fitness += reward
              print(observation)

              if done:
                  observation = env.reset()
                  break


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.MctsReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run until a solution is found.
winner = p.run(eval_genomes)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
net = neat.nn.FeedForwardNetwork.create(winner, config)
observation = env.reset()
for __ in range(20):
  print('\n')
  for _ in range(200):
    env.render(mode='human')
    action = net.activate([observation])
    high = action.index(max(action))
    observation, reward, done, info = env.step(high)

    if done:
        observation = env.reset()
        break
env.close()
