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



#Evolution for weights only, not topology
class LocalSearchPopulation:
    def __init__(self, root, population_size, config, genome_indexer, fitness_function):
        self.root = root
        self.best_found_individual = root
        self.population_size = population_size
        self.config = config
        self.genome_indexer = genome_indexer
        self.fitness_function = fitness_function
        self.population = self.init_population()
        self.elitism = round(self.population_size/15)
        self.epochs = 5

    def init_population(self):
        population = {}
        for _ in range(self.population_size-1):
            gid = next(self.genome_indexer)
            new_state = self.config.genome_type(gid)
            new_state.configure_copy(self.root, self.config.genome_config)
            new_state.mutate_connection_weights(self.config.genome_config)

            population[gid] = new_state
        
        self.fitness_function(list(iteritems(population)), self.config)

        return population

    def run(self):
        for _ in range(self.epochs):
            next_population = {}
            selected = self.selection(self.population)

            #Transfer elites
            for n in range(self.elitism):
                next_population[selected[n].key] = selected[n]
            

            while len(next_population.items()) < self.population_size:
                p1 = self.weighted_choice(selected)
                p2 = self.weighted_choice(selected)
                
                gid = next(self.genome_indexer)
                child = self.config.genome_type(gid)
                child.configure_crossover(
                    p1, p2, self.config.genome_config)
                if random.uniform(0,1) < 0.5:
                    child.mutate_connection_weights(self.config.genome_config)

                next_population[gid] = child
            
            self.fitness_function(list(iteritems(next_population)), self.config)

            for key, item in next_population.items():
                if item is None or self.best_found_individual is None:
                    continue
                if self.config.fitness_criterion == "max":
                    if item.fitness > self.best_found_individual.fitness:
                        self.best_found_individual = item
                else:
                    if item.fitness < self.best_found_individual.fitness:
                        self.best_found_individual = item

            self.population = next_population
        


    def weighted_choice(self, choices):
        for choice in choices:
            if random.uniform(0,1) > 0.8:
                return choice
        return choices[len(choices)-1]

    def selection(self, population):
        selected = []

        pqueue = Q.PriorityQueue()
        seen_fitnesses = set()

        for _, value in population.items():
            if value.fitness not in seen_fitnesses:
                seen_fitnesses.add(value.fitness)

                if self.config.fitness_criterion == "max":
                    pqueue.put((-value.fitness, value))
                else:
                    pqueue.put((value.fitness, value))

        
        for _ in range(math.floor(len(population.items())/2)):
            try:
                selected.append(pqueue.get(False)[1])
            except:
                break

        return selected

        
    

        

class mctsReproductionWeightEvolution(DefaultClassConfig):
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

    def simulate(self, genome, config, fitness_function):
        g = config.genome_type(genome.key)
        g.configure_copy(genome, config.genome_config)
        g.parent = genome
        for _ in range(10):
            g.mutate(config.genome_config)
            g.mutate_connection_weights(config.genome_config)

        fitness_function(
            list(iteritems({g.key: g})), config)
        g.Q = g.fitness

        return g

    # propagates q value of a genome up the tree if the q value is better than already found paths.
    def backpropagate(self, genome, fitness_criterion):
        Q = genome.Q
        genome.N += 1
        while(genome.parent):
            genome = genome.parent
            genome.N += 1
            if fitness_criterion == "max":
                if Q > genome.Q:
                    genome.Q = Q
            else:
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

    def local_search(self, state, fitness_function, config):
        pop = LocalSearchPopulation(state, 20, config, self.genome_indexer, fitness_function)
        pop.run()
        best_found = pop.best_found_individual
        best_found.children = state.children
        best_found.expanded = state.expanded
        best_found.parent = state.parent
        best_found.N = state.N
        best_found.Q = state.Q
        best_found.key = state.key
        
        return best_found

    def expansion(self, state, fitness_function, config):

        state.expanded = True

        possible_connections, possible_nodes = self.get_possible_actions(
            state, config.genome_config)

        for connection in possible_connections:
            child = self.create_child(
                state, connection, 'connection', config)
            fitness_function(
                list(iteritems({child.key: child})), config)
            state.add_child(child)

        for node in list(possible_nodes.keys()):
            child = self.create_child(
                state, node, 'node', config)
            fitness_function(
                list(iteritems({child.key: child})), config)
            state.add_child(child)

    def _calc_uct(self, state, c):
        return (state.Q / state.N) + (2*c*(math.sqrt((2*math.log(state.N)) / state.N)))

    def selection(self, state, c, fitness_criterion):
        selected = state.children[0]
        selected_uct = self._calc_uct(state.children[0], c)
        for child in state.children:
            child_uct = self._calc_uct(child, c)
            if fitness_criterion == "max":
                if child_uct > selected_uct:
                    selected = child
                    selected_uct = child_uct
            else:
                if child_uct < selected_uct:
                    selected = child
                    selected_uct = child_uct
        return selected

    def reproduce(self, config, species, pop_size, generation, fitness_function):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.
        """


        index = 0
        current_best_seen = None
        new_population = {}
        for s in species.species:
                #for _, genome in species.species[s].members.items():
                first = list(species.species[s].members.keys())[0]
                genome = species.species[s].members[first]
                new_population[genome.key] = genome

                if genome.parent == None and len(genome.children) == 0:
                    self.expansion(genome, fitness_function, config)


                while genome.expanded:
                    print(len(genome.children))
                    index += 1
                    print(index)
                    if random.uniform(0,1) < 0.5:
                        genome = self.local_search(
                            genome, fitness_function, config)
                    
                    if len(genome.children) > 1 and random.uniform(0,1) < 1:
                        fitnesses = []
                        for child in genome.children:
                            fitnesses.append(child.Q)
                        for i in range(math.floor(len(fitnesses)/2)):
                            worst_child = fitnesses.index(max(fitnesses)) if config.fitness_criterion == "min" else fitnesses.index(min(fitnesses)) #Remove revsersed of fitness criterion
                            del genome.children[worst_child]
                            del fitnesses[worst_child]
                    
                    if config.fitness_criterion == "max":
                        if current_best_seen == None or current_best_seen.fitness < genome.fitness:
                            current_best_seen = genome
                    else:
                        if current_best_seen == None or current_best_seen.fitness > genome.fitness:
                            current_best_seen = genome

                    genome = self.selection(genome, 0.0, config.fitness_criterion)
                    
                    
                current_parent = genome

              

                if (current_parent.parent == None or random.uniform(0, 1) < 1) and len(current_parent.children) == 0:
                    self.expansion(current_parent, fitness_function, config)


                for _ in range(20):
                    selected_child = self.selection(current_parent, 0.5, config.fitness_criterion)
                    simulated_child = self.simulate(
                        selected_child, config, fitness_function)

                    if config.fitness_criterion == "max":
                        if current_best_seen == None or current_best_seen.fitness < simulated_child.fitness:
                            current_best_seen = simulated_child
                    else:
                        if current_best_seen == None or current_best_seen.fitness > simulated_child.fitness:
                            current_best_seen = simulated_child
                    self.backpropagate(simulated_child, config.fitness_criterion)

        return current_best_seen, new_population
