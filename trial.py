import networkx as nx
from dataIO import *
import matplotlib.pyplot as plt
import pickle as pkl
from utils import *

A  =  [[0.000000,  0.0000000,  0.0000000,  0.0000000,  0.05119703, 1.3431599],
       [0.000000,  0.0000000, -0.6088082,  0.4016954,  0.00000000, 0.6132168],
       [0.000000, -0.6088082,  0.0000000,  0.0000000, -0.63295415, 0.0000000],
       [0.000000,  0.4016954,  0.0000000,  0.0000000, -0.29831267, 0.0000000],
       [0.051197,  0.0000000, -0.6329541, -0.2983127,  0.00000000, 0.1562458],
       [1.343159,  0.6132168,  0.0000000,  0.0000000,  0.15624584, 0.0000000]]

#meshToAdjacencyMatrix('toydata.obj')
#G = nx.from_numpy_matrix(np.load('toydata_am.npy'))
#print(G)
#G = nx.from_numpy_matrix(np.array(A))
#print(G)
#nx.draw_networkx(G)

#plt.show()

#toydata_am = pkl.load(open('toydata_am.npy', 'rb'))
#print(toydata_am)

#lil_matrix, label = load_data()
#print('total:'+str(len(label)))
#pprint.pprint(lil_matrix[5])
#pprint.pprint(label[5])

#pprint.pprint(lil_matrix[199])
#pprint.pprint(label[199])
