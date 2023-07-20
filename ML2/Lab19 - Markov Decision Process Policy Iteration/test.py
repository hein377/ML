x = [[(1,2,3,4), (5,6,7,8)],
     [(9,1,2,3), (4,5,6,7)]
     ]

for row in x: print(row)
x[0][0] = 100
x[0][1] = (0, 0, 0, 0)

for row in x: print(row)

y = (1.23, 1.23, 1.23, 1.23)
y = tuple(round(i,1) for i in y)
print(y)