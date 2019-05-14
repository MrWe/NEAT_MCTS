import matplotlib.pyplot as plt
import pickle

class GlobalFitnesses:
    def __init__(self):
        self.fitnesses = []

    def add_fitness(self, fitness):
        self.fitnesses.append(fitness)
    
    def get_fitnesses(self):
        return self.fitnesses

    def plot_graph(self):
        plt.plot([i for i in range(len(self.fitnesses))], self.fitnesses)
        plt.show()
    
    def save(self):
        with open('fitnesses', 'wb') as f:
            pickle.dump(self.fitnesses, f)