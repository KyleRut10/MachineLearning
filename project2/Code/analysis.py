import data
import pandas as pd
from math import sqrt
import random
import distance as distan
import numpy as np
import statistics

'''  Implement k-nearest neighbor and be prepared to find the best k value for 
your experiments. You must tune k and explain in your report how you did the 
tuning.'''
def knn(df, cat_flag_array, k, classify=True, cat_func='ham', num_func='euclidean'):
    col_names = df.iloc[:, :-1].columns.values # dont want class column label
    data = df.values
    train_data = df.iloc[:, :-1].values # remove class column and get raw array
    classes = df.iloc[:, -1]
    cat_dict = None
    if (cat_func == 'VDM'):
        cat_dict = {}
        for i, flag in enumerate(cat_flag_array):
            if flag:
                cat_dict[col_names[i]] = distan.VDM(train_data.T[i], classes)
    neighbors = get_neighbors(data, col_names, data[0], k, cat_flag_array,
                        cat_func, c_dict=cat_dict)
    print(neighbors)
    if classify:
        return find_max_mode(neighbors[:,-1])
    else:
        return sum(neighbors[:,-1])/len(neighbors)

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
    # (Num 6) k-means clustering
    # 
    # Input - df: dataframe with all data points k: number of clusters
    # catigorical: List of the catigorical attributes
    # Output - distort: the distortion for the final clustering
    # centroids: the resulting centroids for the k clusters
   
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
            #print('times looped: {}'.format(times_looped))
            pass
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
                    # Git the value with the highest number of hits
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
            print('IT CONVERGED!!!')
            looping = False
        # exit if looped too many times
        if times_looped > 100:
            looping = False

    '''
    # number in each cluster...
    for i,l in enumerate(clusters):
        if len(l) != 0:
            print('lenght cluster {}: {}'.format(i, len(l)))
    '''

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

    #print('distortion: ', distort)

    # return distortion and centroids
    return distort, centroids


''' Implement Partitioning Around Medoids for k-medoids clustering and use the 
medoids as a reduced data set for k-NN. Note that the k for k-medoids is 
different than the k for k-NN.'''
def kmedoids(df, k):
    # (Num 7) k-medoids clustering
    # 
    # Input - df: dataframe with all the data points in it k: number mediods
    # 
    # Output - medoids: the resulting medoids
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
    return medoids

# distortion method for k-medoids
def distortion(df, clusters, medoids):
    distort = 0
    k = len(medoids)
    for j in range(k):
        for v in clusters[j]:
            distort += distan.euclidean(df.iloc[j,:], df.iloc[v,:])
    return distort

def get_neighbors(train, col_names, instance, k, cat_flag_array, cat_func, c_dict=None):
    distances = list()
    for train_row in train:
        dist = distan.mixed_distance(
            instance, train_row, col_names, cat_flag_array, cat_func=cat_func, cat_dict=c_dict)
        distances.append((train_row, dist))
    distances.sort(key = lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return np.array(neighbors)



# found this online to handle mode ties
def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode
