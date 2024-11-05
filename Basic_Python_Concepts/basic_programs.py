def fibonnaci_series(n):
    """Function for printing fibonnaci series of n numbers.

    Args:
        n (int): number of terms in fibonnaci series.
    """
    a = 0
    b = 1
    print(a, b, end=' ')
    for i in range(n-2):
        c = a + b
        print(c, end=' ')
        a = b
        b = c
    print()


def max_min_array(arr):
    """Find maximum and minimum elements of an array.

    Args:
        arr (list): array of numbers.
    """
    print("Maximum element: %d\nMinimum element: %d" % (max(arr), min(arr)))


def pos_neg_array(arr):
    """Find positive and negative elements of an array.

    Args:
        arr (list): array of numbers.
    """
    for i in arr:
        if i >= 0:
            print("Positive", i)
        elif i < 0:
            print("Negative", i)


def add_matrices(mat1, mat2):
    final = mat1
    for i in range(len(mat1)):
        for j in range(len(mat1[i])):
            final[i][j] += mat2[i][j]
    
    for i in range(len(mat1)):
        for j in range(len(mat1[i])):
            print(f"%2d + %2d = %2d" %
                  (mat1[i][j], mat2[i][j], final[i][j]), end=" | ")
        print()

    # print(final)


# fibb = int(input("Enter the number of terms in fibonnaci series: "))
fibonnaci_series(10)

print("\n")
# arr = list(map(int, input("Enter the elements of array: ").split()))
max_min_array([1, 2, 3, -1, -248, 1948, 1, 0, 2, -3])

print("\n")
pos_neg_array([1, 2, 3, -1, -248, 1948, 1, 0, 2, -3])


print("\n")
mat1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
mat2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
add_matrices(mat1, mat2)
