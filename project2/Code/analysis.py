import data
import pandas as pd
from math import sqrt
import random
import distance as distan

'''  Implement k-nearest neighbor and be prepared to find the best k value for 
your experiments. You must tune k and explain in your report how you did the 
tuning.'''
def knn(df):
    # (Num 3)
    # 
    # Input -
    # 
    # Output - 
    # 
    
    #TODO split df into test and train
    #Determing how to find best num_neighbors
    
    predictions = list()
    for row in test:
        #Grabbing the Nearest Neighbors by Euclidean Disstance
        neighbors = get_neighbors(train, row, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        
        #Calculating the Prediction based upon the Nearest Neighbors
        prediction = max(set(output_values), key=output_values.count)
        predictions.append(prediction)
        
    pass


''' Implement edited k-nearest neighbor. See above with respect to tuning k. 
On the regression problems, you should define an error threshold to determine 
if a prediction is correct or not. This will need to be tuned'''
def knn_edited(df):
    # (Num 4)
    # 
    # Input -
    # 
    # Output - 
    # 
    pass


''' Implement condensed k-nearest neighbor. See above with respect to tuning 
k and e'''
def knn_condenced(df):
    # (Num 5)
    # 
    # Input -
    # 
    # Output - 
    # 
    pass


''' Implement k-means clustering and use the cluster centroids as a reduced 
data set for k-NN.'''
def kmeans(df, k, dist_metric):
    # (Num 6)
    # 
    # Input -
    # 
    # Output - 
    # 
    
    # Get min and max values for each feature to get data space
    feature_min_max = []
    for f in df.columns.values:
        vals = df[f]
        feature_min_max.append([min(vals), max(vals)])
    
    # Pick k initial starting centroids from data space
    centroids = []
    for kk in range(k):
        centroid = []
        # populate the centroid with a value for each feature
        for f in feature_min_max:
            # TODO: DO THIS DIFFERENTLY WITH MORE MATH THINGS......
            centroid.append(random.randint(int(f[0]), int(f[1])))
            # Don't limit to only integers
            centroid[-1] *= random.random()
        centroids.append(centroid)

    
    while True:
        # make empty set for all clusters
        clusters = [[] for i in range(k)]
        for i,x in df.iterrows():
            x_vec = list(x)
            centroid_dists = []
            # calculate euclidean distance to all centroids
            for cent in centroids:
                centroid_dists.append(dist_metric(cent, x_vec))
                print(centroid_dists[-1])
        break


''' Implement Partitioning Around Medoids for k-medoids clustering and use the 
medoids as a reduced data set for k-NN. Note that the k for k-medoids is 
different than the k for k-NN.'''
def kmediods(df, k):
    # (Num 7)
    # 
    # Input -
    # 
    # Output - 
    # 
    pass

# Repetetive use functions for many of these?
def euclidean_distance(row1, row2):
    # A function for calculating the Euclidean Distance
    # Inputs: Two Rows to calculate the distance between
    # Output: The Euclidean Distance Between the Vectors
    
    distance = 0
    for i in range(1, len(row1)):
        distance += (row1.iloc[i] - row2.iloc[i])**2
    return sqrt(distance)

def get_neighbors(train, test_row, n):
    # A function for finding n nearest neighbors
    # Inputs: train is the training dataframe
    #     test_row is the row we are finding neighbors for
    #     n is the number of nearest neighbors
    # Outputs: The n nearest neighbors of the test_row
    
    distances = list()
    
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key = lambda tup: tup[1])
        
    neighbors = list()
    for i in range(n):
        neighbors.append(distances[i][0])
    
    return neighbors
    

# TODO: Other steps

if __name__ == '__main__':
    # put main logic here
    pass
