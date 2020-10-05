import analysis as a


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
    TenGroups[9] = sort
    for i in sorted(data['class'].unique()):
        totals = totals.append(sum(sort['class'] == i))
        for n in range(9):
          TenGroups[n] = pd.Dataframe()
          for m in range(floor((totals[itr + 1] + totals[itr])*0.1)):
            TenGroups[n] = TenGroups[n].append(sort.iloc[totals[itr] + m])
            TenGroups[9] = TenGroups[9].drop(totals[itr] + m)
        itr += 1
    
    # Could store tuning_data as TenGroups[10] to output
    pass
}
