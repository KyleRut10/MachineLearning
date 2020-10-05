import analysis as a


def tune_kmeans(k_vals_list):
    
    pass


if __name__ == '__main__':
    # TODO: Put main logic here
    pass

def stratified_sample(data) {
    #Produced Random Samples by Proportion
    sort = data.sort_values(by = 'class')
  
    # Getting out the Tuning Data
    # NOTE: This is an estimate of the 10% but it is not very precise
    totals = list(0)
    tuning_data = pd.DataFrame()
    itr = 0
    for i in sorted(data['class'].unique()):
        totals = totals.append(sum(df['class'] == i))
        random_row = random.sample(range(totals[itr], totals[itr + 1]),\\
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
            timesadd = mod((rows)*0.1))
            lister = list()
            for x in range(10)
                if (x < timesadd):
                    lister = lister.append(base + 1)
                else:
                    lister = lister.append(base)
            for m in lister:
                TenGroups[n] = TenGroups[n].append(sort.iloc[totals[itr] + m])
        itr += 1
    
    # Could store tuning_data as TenGroups[10] to output
    pass
}

def nonrandom_sample(data) {
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
            removable2 = removable2.drop(floor(removable1.shape[0]/10)*i + j]))
    TenGroups[9] = removable2
}
