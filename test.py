def f(x, bestx=[0]):
    if bestx[0] < x:
        bestx[0] = x
    print(bestx[0])
    return bestx[0]


f(1)
f(2)
f(1.5)
