
"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import gym
import sys
sys.path.append('..')
import neat


env = gym.make("BipedalWalker-v2")

def eval_genomes(genomes, config):
    observation = env.reset()
    for genome_id, genome in genomes:
      observation = env.reset()
      net = neat.nn.FeedForwardNetwork.create(genome, config)
      genome.fitness = 0

      for _ in range(100):
          action = net.activate(observation)
          observation, reward, done, info = env.step(action)
          genome.fitness += reward

          if done:
            observation = env.reset()
            break
      
      


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.MctsReproductionWeightEvolution,
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

# Show output of the most fit genome against training data.
net = neat.nn.FeedForwardNetwork.create(winner, config)
for __ in range(10):
  observation = env.reset()
  for _ in range(100):
    env.render()
    action = net.activate(observation)
    observation, reward, done, info = env.step(action)

    if done:
      observation = env.reset()
      break
env.reset()
env.close()
