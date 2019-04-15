"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""
from __future__ import division, print_function

import math
import random

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
        for i in range(num_genomes):
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
        for _ in range(20):
            g.mutate(config.genome_config)

        fitness_function(list(iteritems({genome.key: genome})), config)

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

    def expansion(self, state, config):
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
        return (state.Q / state.N) + c*math.sqrt((2*math.log(state.N)) / state.N)

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

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.

        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(
                    m.fitness for m in itervalues(stag_s.members))
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.
        if not remaining_species:
            species.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        fitness_range = max(
            self.reproduction_config.fitness_min_divisor, max_fitness - min_fitness)
        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in itervalues(afs.members)])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        self.reporters.info(
            "Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(
            min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)

        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # Do hillclimb

            # The species has at least one member for the next generation, so retain it.
            old_members = list(iteritems(s.members))
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            best_individual_id, best_individual = old_members[0]

            # temp
            current_parent = best_individual

            self.expansion(current_parent, config)

            for _ in range(1000):

                selected_child = self.selection(current_parent, 0.5)
                simulated_child = self.simulate(
                    selected_child, config, fitness_function)
                self.backpropagate(simulated_child)

            best_child = self.selection(current_parent, 0.)
            self.expansion(best_child, config)
            best_child = best_child.children[1]
            new_population[best_child.key] = best_child

            print(best_child.key)
            return new_population

            while spawn > 0:
                spawn -= 1
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_copy(best_individual, config.genome_config)
                for _ in range(math.floor(random.uniform(2, 10))):
                    child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (best_individual_id)

            return new_population
