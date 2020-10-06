# Make distance metrics in terms of vector x1 and vector x2 so we can use them
# interchangabely
import pandas as pd
import numpy as np


# A function for calculating the Euclidean Distance
# Inputs: Two Rows to calculate the distance between
# Output: The Euclidean Distance Between the Vectors
def euclidean(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += abs(x1[i] - x2[i])**2
    dist = dist**(1/2)
    return dist

# 1D vector + 1D vector -> (1D vector, 2D matrix)
# takes a categorical attribute vector and a class vector and computes the value difference metric
# matrix between the possible values of that attribute vector (with p set to 1)
# returns a tuple (unique_value_list, VDM matrix)
def VDM(vect, classes):
    if (len(vect) != len(classes)):
        raise IndexError("Your vectors are different length")
    unique_cats = list(set(vect)) # unique categorical values in vect
    unique_classes = list(set(classes))

    sum_mat = [[0]*len(unique_cats) for cat in unique_cats]
    for a in unique_classes: #i is the index, a is the class
        a_counts = [0]*len(unique_cats) # saving the counts so only have to iterate once
        val_counts = [0]*len(unique_cats)
        for val_num, val in enumerate(unique_cats):
            for i, entry in enumerate(vect):
                if (entry == val):
                    val_counts[val_num] += 1
                    if (classes[i] == a):
                        a_counts[val_num] += 1
        # add the running total to the matrix for this class
        for val1_i, val1 in enumerate(unique_cats):
            for val2_i, val2 in enumerate(unique_cats):
                if (val1 != val2):
                    prob1 = a_counts[val1_i]/val_counts[val1_i]
                    prob2 = a_counts[val2_i]/val_counts[val2_i]
                    sum_mat[val1_i][val2_i] += abs(prob1-prob2) # add the diff to each mat value
    return (unique_cats, sum_mat)

def Hamming(c1, c2):
    #Calculates the Hamming Distance
    sum = 0
    for i in range(len(c1)):
        if (c1[i] != c2[i]):
            sum += 1
    return sum

def vdm_df(unique_cats, sum_mat):
    # initilize a dictionary to store the lists in
    df_dict = {'index': unique_cats}
    for uc in unique_cats:
        df_dict[uc] = []

    for row in sum_mat:
        for i,uc in enumerate(unique_cats):
            df_dict[uc].append(row[i])

    df = pd.DataFrame.from_dict(df_dict)
    df = df.set_index('index')
    #print(df)
    return df

def mixed_distance(vect1, vect2, col_labels, cat_flag_array, num_func='euclidean', cat_func='ham',
                                                                            cat_dict=None):
    if (cat_func == 'VDM' and 1 in cat_flag_array and cat_dict==None):
        raise TypeError('Must pass in VDM dictionary if categorical columns exist')
    vect1 = vect1[:-1] #removing class col
    vect2 = vect2[:-1] #removing class col
    vect1_cats = []
    vect2_cats = []
    vect1_nums = []
    vect2_nums = []
    cat_labels = []
    for i, cat_flag in enumerate(cat_flag_array):
        if cat_flag:
            vect1_cats.append(vect1[i])
            vect2_cats.append(vect2[i])
            cat_labels.append(col_labels[i])
        else:
            vect1_nums.append(vect1[i])
            vect2_nums.append(vect2[i])
    if (cat_func == 'ham'):
        cat_dist = Hamming(vect1_cats, vect2_cats)
    elif (cat_func == 'VDM'):
        cat_dist = VDM_dist(vect1_cats, vect2_cats, cat_labels, cat_dict)
    else:
        raise TypeError('cat_func not supported')
    if (num_func == 'euclidean'):
        num_dist = euclidean(vect1_nums, vect2_nums)
    else:
        raise TypeError('num_func not supported')

    return len(vect1_cats)*cat_dist + len(vect1_nums)*num_dist

def VDM_dist(vect1, vect2, cat_labels, cat_dict):
    dist = 0
    for i, col in enumerate(cat_labels):
        val_list = cat_dict[col][0]
        val1_ind = val_list.index(vect1[i])
        val2_ind = val_list.index(vect2[i])
        dist += cat_dict[col][1][val1_ind][val2_ind]
    return dist







