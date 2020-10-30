import random
import analysis as a
import data as d
import math
import pandas as pd


def tune_kmeans(df, k_vals_list, cat):
    # remove potential negative values
    k_vals_list = [k for k in k_vals_list if k > 0 and k < len(df)]
    print(k_vals_list)
    # make empty distortion
    distortions = [math.inf for x in range(len(k_vals_list))]
    # Run k-means for all options
    for i,k in enumerate(k_vals_list):
        dist,centroids = a.kmeans(df, k, cat)
        print('Ran k-means for k = {}, dist = {}'.format(k, round(dist,5)))
        distortions[i] = dist

    # find the minimum distortion
    min_dist = min(distortions)
    k_loc = distortions.index(min_dist)
    # return the k value with the lowest distortion
    return k_vals_list[k_loc]


def stratified_sample(data):
    #Produced Random Samples by Proportion
    sort = data.sort_values(by = 'class')
  
    # Getting out the Tuning Data
    # NOTE: This is an estimate of the 10% but it is not very precise
    totals = list()
    tuning_data = pd.DataFrame()
    itr = 0
    for i in sorted(data['class'].unique()):
        totals = totals.append(sum(data['class'] == i))
        random_row = random.sample(range(totals[itr], totals[itr + 1]),
        max(floor(totals(itr + 1) * 0.1), 1))
      
        for j in random_row:
            tuning_data = tuning_data.append(sort.iloc[i])
            sort = sort.drop(j)
        itr += 1
  
    
    # Repeating this process for Training and Testing Data
    totals = list(0)
    TenGroups = {}
    itr = 0
    for i in sorted(data['class'].unique()):
        totals = totals.append(sum(sort['class'] == i))
        for n in range(9):
            TenGroups[n] = pd.Dataframe()
            rows = totals[itr + 1] - totals[itr]
            base = floor((rows)*0.1)
            timesadd = mod(rows*0.1)
            lister = list()
            for x in range(10):
                if (x < timesadd):
                    lister = lister.append(base + 1)
                else:
                    lister = lister.append(base)
            for m in lister:
                TenGroups[n] = TenGroups[n].append(sort.iloc[totals[itr] + m])
        itr += 1

    return (tuning_data, TenGroups)


def nonrandom_sample(data):
    tuning = pd.DataFrame()
    sort = data.sort_values('response')
    removable1 = sort
    for i in range(floor(sort.shape[0]/10)):
        tuning = tuning.append(sort[floor(sort.shape[0]/10)*i])
        removable = removable.drop(floor(sort.shape[0]/10)*i)
    
    TenGroups = {}
    
    for i in range(10):
        TenGroups[i] = pd.DataFrame
    
    removable2 = removable1
    for i in range(floor(removable.shape[0]/10)):
        for j in range(9):
            TenGroups[j] = TenGroups[j].append(removable1[floor(removable1.shape[0]/10)*i + j])
            removable2 = removable2.drop(floor(removable1.shape[0]/10)*i + j)
    TenGroups[9] = removable2


def tune_run_kmeans(type_funct, data_funct):
    cat = type_funct()[0]
    df = data_funct()
    ss = int(math.sqrt(len(df)))
    k = tune_kmeans(df, [ss+i for i in range(-20, 20, 4)], cat)
    print('Best k is {}'.format(k))
    dist, centroids = a.kmeans(df, k, cat)
    return dist, centroids


def run_kmeans():
    print('********************')
    print('*******K-MEANS******')
    print('********************')
    print('--------------------')
    # glass dataset
    print('***GLASS DATASET****')
    tune_run_kmeans(d.type_glass, d.data_glass)

    # House-votes dataset
    print('\n**HOUSE VOTES**')
    tune_run_kmeans(d.type_glass, d.data_glass)

    # Segmentation dataset
    print('\n***SEGMENTATION***')
    tune_run_kmeans(d.type_segmentation, d.data_segmentation)
    
    '''
    # abalone dataset
    print('\n***ABALONE***')
    tune_run_kmeans(d.type_abalone, d.data_abalone)
    '''

def demo():
    print('Get k-means dataset')
    print('Through tuning using destortion functions, k = 30 for glass')
    dist, centroids = a.kmeans(d.data_glass(), 30, d.type_glass()[0])
    print('distortion: ', dist)
    print()



if __name__ == '__main__':
    #demo()
    run_kmeans()
    #run_knn()
