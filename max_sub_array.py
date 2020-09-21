def findMaxSubArray(A):
    """Jay Kadan's algorithm"""
    ln = len(A)

    if ln:
        res, summ, l, r, min_pos = A[0], 0, 0, 0, -1

        for i in range(ln):
            summ += A[i]

            if summ > res:
                res = summ
                l = min_pos + 1
                r = i

            if summ < 0:
                summ = 0
                min_pos = i

        return A[l:r+1]
