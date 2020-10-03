# Make distance metrics in terms of vector x1 and vector x2 so we can use them
# interchangabely 


def euclidean(x1, x2):
    dist = 0
    for i in range(len(x1)):
        dist += abs(x1[i] + x2[i])**2
    dist = dist**(1/2)
    return dist
