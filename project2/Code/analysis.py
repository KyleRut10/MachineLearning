import data
import pandas as pd
from math import sqrt
import random
import distance as distan
import numpy as np

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
def kmeans(df, k, catigorical):
    # (Num 6)
    # 
    # Input -
    # 
    # Output - 
    # 
   
    # get atribute distance matrices
    cat_dists = {}
    for cat in catigorical:
        cats,mat = distan.VDM(df[cat], df['class']) 
        cat_dists[cat] = distan.vdm_df(cats,mat)

    # drop the class label for clustering
    if 'class' in df.columns.values:
        df = df.drop(columns=['class'])

    # Get min and max values for each feature to get data space
    feature_min_max = []
    for f in df.columns.values:
        vals = df[f]
        # this will work even for string data, it'll just get ignored later
        feature_min_max.append([min(vals), max(vals)])
    
    # Pick k initial starting centroids from data space
    centroids = []
    for kk in range(k):
        centroid = []
        # populate the centroid with a value for each feature
        for i,col in enumerate(df.columns.values):
            f = feature_min_max[i]
            if col in catigorical:
                # Add random attribute value from catigorical values
               centroid.append(random.choice(df[col].unique())) 
            else:
                centroid.append(random.randint(int(f[0]), int(f[1])))
                # Don't limit to only integers
                centroid[-1] += random.random()
        centroids.append(centroid) 

    
    looping = True
    times_looped = 0
    while looping:
        '''
        print()
        print('init centroids')
        for centroid in centroids:
            #print(', '.join([str(round(x,3)) for x in centroid]))
            print(', '.join([str(x) for x in centroid]))
        '''
        if times_looped % 20 == 0:
            print('times looped: {}'.format(times_looped))
        times_looped += 1

        old_centroids = centroids.copy()
        # make empty set for all clusters
        clusters = [[] for i in range(k)]
        # go through each datapoint and assign it to a cluster
        for i,x in df.iterrows():
            x_vec = list(x)
            centroid_dists = []
            # calculate distance to all centroids
            for cent in centroids:
                cent_dist = 0
                # Calculate the distance of each point in the centroid
                for p,point in enumerate(cent):
                    col = df.columns.values[p]
                    # check if the column is catigorical
                    if col in catigorical:
                        cent_dist += cat_dists[col].loc[point,x[p]]
                    else:
                        cent_dist += distan.euclidean(point, x[p])
                centroid_dists.append(cent_dist)
            # Pick the minimum distance
            c_val = min(centroid_dists)
            #print(min(centroid_dists),  ' ', max(centroid_dists))
            # find the index and assign it to the cluaster
            c_index = centroid_dists.index(c_val)
            clusters[c_index].append(x_vec)
            #print('cluster index: ', c_index)
        # make new centroids for each cluster
        centroids = []
        for c,cluster in enumerate(clusters):
            # make empty mean holder
            means = {}
            for col in df.columns.values:
                if col in catigorical:
                    means[col] = {}
                    for uni in df[col].unique():
                        means[col][uni] = 0
                else:
                    means[col] = 0
            # for each data point
            for data in cluster:
                # Go through each attribute in the row
                for p,col in enumerate(df.columns.values):
                    # if the column is catigor, use the catigorical dist matrix
                    if col in catigorical:
                        # add one to the count
                       means[col][data[p]] += 1
                    else:
                        means[col] += data[p]
            #print(means)
            # assign mean to centroid
            centroid = []
            for m,mean_key in enumerate(list(means.keys())):
                mean = means[mean_key]
                # real featured value, just add and divide by length clust c
                if not isinstance(mean, dict):
                    if len(clusters[c]) != 0:
                        centroid.append(mean/len(clusters[c]))
                    else:
                        centroid.append(0)
                else:
                    # initilize maximum vlaue
                    max_val = 0
                    max_key = ''
                    for key in mean.keys():
                        if mean[key] > max_val:
                            max_val = mean[key]
                            max_key = key
                    centroid.append(max_key)
            centroids.append(centroid) 
        '''
        print('final centroids')
        for centroid in centroids:
            #print(', '.join([str(round(x,3)) for x in centroid]))
            print(', '.join([str(x) for x in centroid]))
        '''
        #print('Checking convergance')
        # sum over all the clusters
        centroid_score = 0
        for c,centroid in enumerate(centroids):
            #for x in centroid:
            for p,point in enumerate(centroid):
                col = df.columns.values[p]
                # check if the column is catigorical
                if col in catigorical:
                    centroid_score += cat_dists[col].loc[point,
                                                         old_centroids[c][p]]
                else:
                    centroid_score += distan.euclidean(point, x[p])
           
        #print(centroid_score)
        if centroid_score <= 0.05*k:
            looping = False
        # exit if looped too many times
        if times_looped > 10:
            looping = False


    # number in each cluster...
    for i,l in enumerate(clusters):
        if len(l) != 0:
            print('lenght cluster {}: {}'.format(i, len(l)))
    
    # calculate distortion
    distort = 0
    # outer j=0 to k sum
    for kk,cluster in enumerate(clusters):
        inner_sum = 0
        # for each data point in the cluster
        for val in cluster:
            # for each attribute in that data point
            val_distort = 0
            for i,col in enumerate(df.columns.values):
                # check if catigorical
                if col in catigorical:
                    val_distort = cat_dists[col].loc[val[i], centroids[kk][i]]
                else:
                    val_distort = distan.euclidean(val[i], centroids[kk][i])
            inner_sum += val_distort**2
        distort += inner_sum

    print('distortion: ', distort)

    # return distortion and centroids
    return distortion, centroids


''' Implement Partitioning Around Medoids for k-medoids clustering and use the 
medoids as a reduced data set for k-NN. Note that the k for k-medoids is 
different than the k for k-NN.'''
def kmedoids(df, k):
    # (Num 7)
    # 
    # Input -
    # 
    # Output - 
    # 

    # randomly select centroids
    medoids = random.choices(range(len(df)), k=k)

    clusters = [[] for i in range(k)]
    looping = True
    times_looped = 0
    while looping:
        if times_looped % 1 == 0:
            print('looped: {}'.format(times_looped))
        times_looped += 1
        
        # Keep track of old centroid values
        old_medoids = medoids.copy()
        
        # loop through every data point
        for point_loc in range(len(df)):
            point = df.iloc[point_loc]
            # assign values to clusters
            dists = []
            for med_loc in medoids:
                med = df.iloc[med_loc, :]
                # calculate distance to each centroid
                dists.append(distan.euclidean(med, point))
                #dists.append(euclidean_distance(cent, point))
            # pick minimum distance and put point in correct cluster
            clusters[dists.index(min(dists))].append(point_loc)
        
        # calculate distortion
        distort = distortion(df, clusters, medoids)
        
        # swaping
        for i,med_loc in enumerate(medoids):
            for xi in range(len(df)):
                # don't compare if medoid, break from loop
                if xi == med_loc:
                    break
                swap_medoids = medoids.copy()
                swap_medoids[i] = xi
                swap_distort = distortion(df, clusters, swap_medoids, 
                                          distan.euclidean)
                if swap_distort <= distort:
                    pass # this is where they'd be swaped back
                else: 
                    medoids = swap_medoids

        # Check if the centroids are the same
        looping = False
        for i,m in enumerate(medoids):
            if old_medoids[i] != m:
                looping = True


    print('Number in each cluster')
    for c in clusters:
        print(len(c))
    
    # return medoids
    #return medoids

# distortion method for k-medoids
def distortion(df, clusters, medoids):
    distort = 0
    k = len(medoids)
    for j in range(k):
        for v in clusters[j]:
            distort += distan.euclidean(df.iloc[j,:], df.iloc[v,:])
    return distort


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
