# Vizualization tools
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import networkx as nx

def graph(M):
    ''' Shows the graph corresponding to the biadjacency measurement matrix '''
    # Creates a basic graph from the biadjacency matrix
    SM = csc_matrix(M)
    SG = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(SM)
    nx.draw(SG)
    plt.show()
    return
