
"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import gym
import sys
from math import inf
sys.path.append('..')
import neat


env = gym.make("MountainCar-v0")

def eval_genomes(genomes, config):
    observation = env.reset()
    for genome_id, genome in genomes:
      observation = env.reset()
      net = neat.nn.FeedForwardNetwork.create(genome, config)
      genome.fitness = 0
      max_height_reached = -inf
      max_vel_reached = -inf

      for _ in range(200):
        actions = net.activate(observation)
        action = actions.index(max(actions))
        observation, reward, done, info = env.step(action)
        if observation[0] > 0.5:
          genome.fitness = 1000
        if done:
          observation = env.reset()
          break
        genome.fitness += reward
        max_height_reached = observation[0] if observation[0] > max_height_reached else max_height_reached
        max_vel_reached = observation[1] if observation[1] > max_vel_reached else max_vel_reached
      genome.fitness +=  (1 + abs(max_height_reached)) ** 2
      genome.fitness +=  (1 + abs(max_vel_reached)) ** 2

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.mctsReproductionWeightEvolution,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(False))

# Run until a solution is found.
winner = p.run(eval_genomes, 100)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))

# Show output of the most fit genome against training data.
net = neat.nn.FeedForwardNetwork.create(winner, config)
for __ in range(10):
  observation = env.reset()
  for _ in range(200):
    env.render()
    actions = net.activate(observation)
    action = actions.index(max(actions))
    observation, reward, done, info = env.step(action)

    if done:
      observation = env.reset()
      print("Done")
      break
env.reset()
env.close()
