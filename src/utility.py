# Given numpy element return row vector
def vrow(arr):
    return arr.reshape(1, arr.size)


# Given numpy element return column vector
def vcol(arr):
    return arr.reshape(arr.size, 1)
