import matplotlib.pyplot as plt
import pickle

class GlobalFitnesses:
    def __init__(self, num):
        self.num = num
        self.fitnesses = []

    def add_fitness(self, fitness):
        if len(self.fitnesses) < 1:
            self.fitnesses.append(fitness)
        elif fitness > self.fitnesses[-1]:
            self.fitnesses.append(fitness)
        else:
            self.fitnesses.append(self.fitnesses[-1])
    
    def get_fitnesses(self):
        return self.fitnesses

    def plot_graph(self):
        plt.plot([i for i in range(len(self.fitnesses))], self.fitnesses)
        plt.show()
    
    def save(self):
        with open('fitnesses', 'wb') as f:
            pickle.dump(self.fitnesses, f)
        
    def save_fig(self):
        fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
        ax.plot([i for i in range(len(self.fitnesses))], self.fitnesses)
        fig.savefig('fitnesses_figs/' + str(self.num)+'.png')   # save the figure to file
        plt.close(fig)    # close the figure