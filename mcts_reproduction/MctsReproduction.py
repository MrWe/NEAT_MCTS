"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""
from __future__ import division, print_function

import math
import random
try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

from itertools import count
from sys import stderr, float_info

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean, NORM_EPSILON
from neat.six_util import iteritems, itervalues

# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate
# configuration. This scheme should be adaptive so that species do not evolve
# to become "cautious" and only make very slow progress.


class MctsReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter(
                                       'survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2),
                                   ConfigParameter('fitness_min_divisor', float, 1.0)])

    def __init__(self, config, reporters, stagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.stagnation = stagnation
        self.ancestors = {}

        if config.fitness_min_divisor < 0.0:
            raise RuntimeError(
                "Fitness_min_divisor cannot be negative ({0:n})".format(
                    config.fitness_min_divisor))
        elif config.fitness_min_divisor == 0.0:
            config.fitness_min_divisor = NORM_EPSILON
        elif config.fitness_min_divisor < float_info.epsilon:
            print("Fitness_min_divisor {0:n} is too low; increasing to {1:n}".format(
                config.fitness_min_divisor, float_info.epsilon), file=stderr)
            stderr.flush()
            config.fitness_min_divisor = float_info.epsilon

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for _ in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm)))
                         for n in spawn_amounts]

        return spawn_amounts

    # TODO
    def simulate(self, genome, config, fitness_function):
        g = config.genome_type(genome.key)
        g.configure_copy(genome, config.genome_config)
        g.parent = genome
        for _ in range(5):
            g.mutate(config.genome_config)
        for _ in range(5):
            g.mutate_connection_weights(config.genome_config)

        fitness_function(
            list(iteritems({g.key: g})), config)
        g.Q = g.fitness

        return g

    # propagates q value of a genome up the tree if the q value is better than already found paths.
    def backpropagate(self, genome):
        Q = genome.Q
        genome.N += 1
        while(genome.parent):
            genome = genome.parent
            genome.N += 1
            if Q < genome.Q:
                genome.Q = Q

    # Find possible add/delete_connection and find possible add/delete_node
    def get_possible_actions(self, genome, config):

        # Find possible connections to add
        possible_connections = genome.get_possible_new_connections(config)

        # Find possible nodes to add
        possible_nodes = genome.connections

        return possible_connections, possible_nodes

    def create_child(self, parent, action, action_type, config):
        gid = next(self.genome_indexer)
        child = config.genome_type(gid)
        child.configure_copy(parent, config.genome_config)
        child.parent = parent

        if action_type == 'connection':
            child.add_connection(config.genome_config,
                                 action[0], action[1], 1.0, True)

        elif action_type == 'node':
            child.add_node(config.genome_config, action)

        return child

    def beam_local_search(self, state, fitness_function, config):
        pqueue = Q.PriorityQueue()
        seen_fitnesses = set()

        seen_fitnesses.add(state.fitness)

        best_seen = state

        for _ in range(10):
            gid = next(self.genome_indexer)
            new_state = config.genome_type(gid)
            new_state.configure_copy(state, config.genome_config)
            new_state.mutate_connection_weights(config.genome_config)
            fitness_function(
                list(iteritems({new_state.key: new_state})), config)

            if new_state.fitness not in seen_fitnesses:
                seen_fitnesses.add(new_state.fitness)
                pqueue.put((new_state.fitness, new_state))

        for _ in range(50):
            current = pqueue.get()[1]
            neighbours = self._beam_get_neighbours(current, config)
            fitness_function(
                list(iteritems(neighbours)), config)
            for _, neighbour in neighbours.items():
                if neighbour.fitness < best_seen.fitness:
                    best_seen = neighbour
                if neighbour.fitness not in seen_fitnesses:
                    seen_fitnesses.add(neighbour.fitness)
                    pqueue.put((neighbour.fitness, neighbour))

        best_seen.children = state.children
        return best_seen

    def _beam_get_neighbours(self, state, config):
        neighbours = {}
        for _ in range(10):
            gid = next(self.genome_indexer)
            new_state = config.genome_type(gid)
            new_state.configure_copy(state, config.genome_config)
            for __ in range(math.floor(random.uniform(1, 4))):
                new_state.mutate_connection_weights(config.genome_config)
            neighbours[gid] = new_state
        return neighbours

    def expansion(self, state, fitness_function, config):
        if state.expanded:
            return
        state.expanded = True

        possible_connections, possible_nodes = self.get_possible_actions(
            state, config.genome_config)

        for connection in possible_connections:
            state.add_child(self.create_child(
                state, connection, 'connection', config))

        for node in list(possible_nodes.keys()):
            state.add_child(self.create_child(
                state, node, 'node', config))

    def _calc_uct(self, state, c):
        return (state.Q / state.N) + (2*c*(math.sqrt((2*math.log(state.N)) / state.N)))

    def selection(self, state, c):
        selected = state.children[0]
        selected_uct = self._calc_uct(state.children[0], c)
        for child in state.children:
            child_uct = self._calc_uct(child, c)
            if child_uct < selected_uct:
                selected = child
                selected_uct = child_uct
        return selected

    def reproduce(self, config, species, pop_size, generation, fitness_function):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        current_best_seen = None

        new_population = {}
        for s in species.species:
            for _, genome in species.species[s].members.items():

                while genome.expanded:
                    genome = self.beam_local_search(
                        genome, fitness_function, config)
                    if current_best_seen == None or current_best_seen.fitness > genome.fitness:
                        current_best_seen = genome
                    genome = self.selection(genome, 0.0)

                current_parent = genome

                if current_parent.parent == None or random.uniform(0, 1) < 0.1:
                    self.expansion(current_parent, fitness_function, config)
                else:
                    current_parent = current_parent.parent

                for _ in range(100):

                    selected_child = self.selection(current_parent, 0.5)
                    simulated_child = self.simulate(
                        selected_child, config, fitness_function)
                    if current_best_seen == None or current_best_seen.fitness > simulated_child.fitness:
                        current_best_seen = simulated_child
                    self.backpropagate(simulated_child)

                new_population[genome.key] = genome

        return current_best_seen, new_population
