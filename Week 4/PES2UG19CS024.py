import numpy as np
from statistics import mode

class KNN:

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        matrix = np.zeros((x.shape[0], self.data.shape[0]))

        for i in range(x.shape[0]):
            for j in range(self.data.shape[0]):
                matrix[i][j] = minowski_distance(self.p, x[i], self.data[j])
        
        return matrix
        
    def k_neighbours(self, x):
        temp = []
        idx = []
        neigh = []
        matrix = KNN.find_distance(self, x)
        for i in matrix:
            temp = np.argpartition(i, self.k_neigh).tolist()
            idx.append(temp[:self.k_neigh])
            neigh.append([i[x] for x in temp[:self.k_neigh]])

        return [np.array(neigh), np.array(idx)]     

    def predict(self, x):
        predictions = []
        list_of_values = KNN.k_neighbours(self, x)
        for i in list_of_values[1]:
            predictions.append(mode([self.target[j] for j in i]))

        return np.array(predictions)

    def evaluate(self, x, y):
        true_values = 0
        z = KNN.predict(self, x)
        for i in range(len(z)):
            if z[i] == y[i]:
                true_values += 1
        
        return float((true_values/len(z))*100)

def minowski_distance(p, row1, row2):
    return float(np.sum(abs(row1 - row2)**p)**(1/p))