

def standardize(column):
    # Standardizes a column by z-score normalization
    
    return column - mean(column)/column.std
