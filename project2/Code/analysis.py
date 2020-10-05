import data
import pandas as pd
from math import sqrt
import random
import distance as distan
import numpy as np

'''  Implement k-nearest neighbor and be prepared to find the best k value for 
your experiments. You must tune k and explain in your report how you did the 
tuning.'''
def knn(df, cat_flag_array, classify=True, cat_func='ham', num_func='euclidean'):
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
            centroid[-1] += random.random()
        centroids.append(centroid) 

    
    looping = True
    times_looped = 0
    while looping:
        #'''
        print()
        print('init centroids')
        for centroid in centroids:
            print(', '.join([str(round(x,3)) for x in centroid]))
        #'''
        if times_looped % 10 == 0:
            print('times looped: {}'.format(times_looped))
        times_looped += 1

        old_centroids = centroids.copy()
        # make empty set for all clusters
        clusters = [[] for i in range(k)]
        for i,x in df.iterrows():
            x_vec = list(x)
            centroid_dists = []
            # calculate euclidean distance to all centroids
            for cent in centroids:
                centroid_dists.append(dist_metric(cent, x_vec))
            c_val = min(centroid_dists)
            #print(min(centroid_dists),  ' ', max(centroid_dists))
            c_index = centroid_dists.index(c_val)
            clusters[c_index].append(x_vec)
            #print('cluster index: ', c_index)
        # make new centroids for each cluster
        centroids = []
        for i,cluster in enumerate(clusters):
            means = [0 for x in range(len(df.columns.values))]
            for data in cluster:
                for d,point in enumerate(data):
                    means[d] += point
            for j,m in enumerate(means):
                if len(cluster) != 0:
                    means[j] = means[j]/len(cluster)
                else:
                    means[j] = 0
            #print('means\n', means)
            centroids.append(means)
        #for i,c in enumerate(centroids):
        #    for ii,cc in enumerate(c):
        #        print(cc, old_centroids[i][ii])
        #'''
        print('final centroids')
        for centroid in centroids:
            print(', '.join([str(round(x,3)) for x in centroid]))
        #'''
        #looping = False
        print('Checking convergance')
        conv_sum = 0
        for i,c in enumerate(centroids):
            cent_diffs = []
            # for each of the values in a centroid
            '''
            diff_dist = dist_metric(c, old_centroids[i])
            if diff_dist >= 0.05:
                looping = True
            print(diff_dist)
            '''
            # subtract vectors
            vec_diff = 0
            for ii,cc in enumerate(c):
                vec_diff += (cc-old_centroids[i][ii])**2

            conv_sum += sqrt(vec_diff)
                cent_diffs.append(abs(cc-old_centroids[i][ii])**2)
                if abs(cc - old_centroids[i][ii])**2 >= 0.05:
                    looping = True
            print(', '.join([str(round(x, 3)) for x in cent_diffs]))
        if conv_sum <= 0.05*k:
            looping = False
        if times_looped > 10:
            print('Looped over 200 times, ending iterations')
            looping = False

    # number in each cluster...
    for l in clusters:
        print(len(l))



''' Implement Partitioning Around Medoids for k-medoids clustering and use the 
medoids as a reduced data set for k-NN. Note that the k for k-medoids is 
different than the k for k-NN.'''
def kmedoids(df, k, dist_metric):
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
                dists.append(dist_metric(med, point))
                #dists.append(euclidean_distance(cent, point))
            # pick minimum distance and put point in correct cluster
            clusters[dists.index(min(dists))].append(point_loc)
        
        # calculate distortion
        distort = distortion(df, clusters, medoids, dist_metric)
        
        # swaping
        for i,med_loc in enumerate(medoids):
            for xi in range(len(df)):
                # don't compare if medoid, break from loop
                if xi == med_loc:
                    break
                swap_medoids = medoids.copy()
                swap_medoids[i] = xi
                swap_distort = distortion(df, clusters, swap_medoids, 
                                          dist_metric)
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
def distortion(df, clusters, medoids,dist_metric):
    distort = 0
    k = len(medoids)
    for j in range(k):
        for v in clusters[j]:
            distort += dist_metric(df.iloc[j,:], df.iloc[v,:])
    return distort

def mixed_distance(vect1, vect2, cat_flag_array, num_func='euclidean', cat_func='ham',
                                                                            cat_dict=None):
    if (cat_func = 'VDM' and 1 in cat_flag_array and cat_dict==None):
        raise TypeError('Must pass in VDM dictionary if categorical columns exist')
    vect1_cats = []
    vect2_cats = []
    vect1_nums = []
    vect2_nums = []
    for i, cat_flag in enumerate(cat_flag_array):
        if cat_flag:
            vect1_cats.append(vect1[i])
            vect2_cats.append(vect2[i])
        else:
            vect1_nums.append(vect1[i])
            vect2_nums.append(vect2[i])
    if (cat_func == 'ham'):
        cat_dist = Hamming(vect1_cats, vect2_cats)
    elif (cat_func == 'VDM'):
        cat_dist = VDM_dist(vect1_cats, vect2_cats, cat_dict)
    else:
        raise TypeError('cat_func not supported')
    if (num_func == 'euclidean'):
        num_dist = euclidean_distance(vect1_nums, vect2_nums)
    else:
        raise TypeError('num_func not supported')

    return len(vect1_cats)*cat_dist + len(vect1_nums)*num_dist




# A function for calculating the Euclidean Distance
# Inputs: Two Rows to calculate the distance between
# Output: The Euclidean Distance Between the Vectors
def euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

# A function for finding k nearest neighbors
# Inputs: train is the training dataframe
#     test_row is the row we are finding neighbors for
#     k is the number of nearest neighbors
# Outputs: The k nearest neighbors of the test_row
def get_neighbors(train, test_row, k):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key = lambda tup: tup[1])
    neighbors = list()
    for i in range(n):
        neighbors.append(distances[i][0])
    return neighbors
