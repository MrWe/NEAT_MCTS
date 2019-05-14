
"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import gym
import pickle
import torch
import numpy as np
from math import inf
import sys
sys.path.append('..')
import neat

env = gym.make("LunarLander-v2")


def eval_genomes(genomes, config):
    observation = env.reset()
    for genome_id, genome in genomes:
      worst_found_fitness = inf
      net = neat.nn.FeedForwardNetwork.create(genome, config)
      for __ in range(10):
        observation = env.reset()

        curr_fitness = 0
        
        for _ in range(1000):
            action = net.activate(observation)
            high = action.index(max(action))
            observation, reward, done, info = env.step(high)
            curr_fitness += reward

            if done:
              observation = env.reset()
              break
        worst_found_fitness = curr_fitness if curr_fitness < worst_found_fitness else worst_found_fitness
      genome.fitness = worst_found_fitness
      
      


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.mctsReproductionWeightEvolution,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run until a solution is found.
winner = p.run(eval_genomes, 200)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Save the winner.
with open('winner-feedforward', 'wb') as f:
  pickle.dump(winner, f)

for __ in range(10):
  # Show output of the most fit genome against training data.
  net = neat.nn.FeedForwardNetwork.create(winner, config)
  observation = env.reset()
  for _ in range(1000):
    env.render()
    action = net.activate(observation)
    high = action.index(max(action))
    observation, reward, done, info = env.step(high)

    if done:
      observation = env.reset()
      break
  env.reset()
env.close()
