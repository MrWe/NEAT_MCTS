
"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function
import sys
import torch
sys.path.append('..')
import neat
from pytorch_neat.cppn import create_cppn


# 2-input XOR inputs and expected outputs.
xor_inputs = [(torch.Tensor([0.0]), torch.Tensor([0.0])), (torch.Tensor([0.0]), torch.Tensor([1.0])), (torch.Tensor([1.0]), torch.Tensor([0.0])), (torch.Tensor([1.0]), torch.Tensor([1.0]))]
xor_outputs = [(0.0,),     (1.0,),     (1.0,),     (0.0,)]


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = create_cppn(genome, config, ["In1","In2"], ["Out"])[0]
        #net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):

            output = net.activate([xi[0], xi[1]], shape=(1,1)).item()
            genome.fitness += (output - xo[0]) ** 2


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

# Show output of the most fit genome against training data.
print('\nOutput:')
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
for xi, xo in zip(xor_inputs, xor_outputs):
    output = winner_net.activate(xi)
    print("  input {!r}, expected output {!r}, got {!r}".format(
        xi, xo, output))
