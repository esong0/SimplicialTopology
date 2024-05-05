# Author: Euijun Song
# Python >= 3.8

import os
import numpy as np
import networkx as nx
import itertools
from scipy import sparse


class SimplicialTopology:
    def __init__(self, network):
        self.G = network

        self.maximal_cliques = [tuple(sorted(c)) for c in nx.find_cliques(self.G)]
        self.N = max([len(c) for c in self.maximal_cliques])

        # Considering all the cliques as simplices
        self.SC = []
        for k in range(self.N):
            _simplices_k = set()
            for c in self.maximal_cliques:
                for sub_c in itertools.combinations(c, k+1):
                    _simplices_k.add(sub_c)
            self.SC.append(sorted(list(_simplices_k)))
    

    def simplicial_complex(self): # Simplicial complex
        return self.SC
    

    def boundary_operator(self, k): # Boundary map
        if k < 0 or k >= self.N:
            raise Exception("Out of range")
        
        if k == 0:
            return sparse.lil_matrix((1, len(self.G.nodes())))
        
        B = sparse.lil_matrix((len(self.SC[k-1]), len(self.SC[k])))
        idx_simplices = {_s: _i for _i, _s in enumerate(self.SC[k-1])}

        for j, simplex in enumerate(self.SC[k]):
            for i, face in enumerate(itertools.combinations(simplex, k)):
                idx_face = idx_simplices[face]
                B[idx_face, j] = (-1)**i
        
        return B
    

    def hodge_laplacian(self, k): # Hodge Laplacian
        if k < 0 or k >= self.N:
            raise Exception("Out of range")
        
        B_k = self.boundary_operator(k)
        B_k1 = self.boundary_operator(k+1)

        L_up = np.dot(B_k1, B_k1.T)
        L_down = np.dot(B_k.T, B_k)
        L = L_up + L_down

        return L, L_up, L_down
    

    def simplicial_adjacency(self, k): # Higher-order adjacency
        if k < 0 or k >= self.N:
            raise Exception("Out of range")
        
        L, L_up, L_down = self.hodge_laplacian(k)

        A = abs(L)
        A_up = abs(L_up)
        A_down = abs(L_down)

        A.setdiag(0)
        A_up.setdiag(0)
        A_down.setdiag(0)

        return A, A_up, A_down
    

    def betti_number(self, k): # Betti number of homology group
        if k < 0 or k >= self.N:
            raise Exception("Out of range")
        
        B_k = self.boundary_operator(k)
        B_k1 = self.boundary_operator(k+1)

        beta_k = B_k.shape[1] - np.linalg.matrix_rank(B_k.todense()) - np.linalg.matrix_rank(B_k1.todense())

        return beta_k
