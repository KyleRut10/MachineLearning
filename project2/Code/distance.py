# Make distance metrics in terms of vector x1 and vector x2 so we can use them
# interchangabely


def euclidean(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += abs(x1[i] + x2[i])**2
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










