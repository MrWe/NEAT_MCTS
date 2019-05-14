import matplotlib.pyplot as plt
import pickle

with open(r"fitnesses", "rb") as input_file:
    e = pickle.load(input_file)
    plt.plot([i for i in range(len(e))], e)
    plt.show()