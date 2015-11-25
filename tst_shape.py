from sp_map import MapArray

ma = MapArray()

print "a new array: "
print "ndim, shape, count_nonzero = ", ma.ndim, ma.shape, ma.count_nonzero()
print "get an element: ", ma[3, 4], "\n"

print "************** set one element"
ma[3, 4] = -101.
print ma.ndim, ma.shape, ma.count_nonzero(), ma[3, 4]

ma[3, 4] = -202
print ma.ndim, ma.shape, ma.count_nonzero(), ma[3, 4]

print "************** set another one"
ma[5, 6] = 111
print ma.ndim, ma.shape, ma.count_nonzero(), ma[3, 4]


print "\n\n ******************* inplace add scalar"
print ">>>>>>>>> id: ", id(ma)

print "Was: ", ma.ndim, ma.shape, ma.count_nonzero(), ma[3, 4]

ma += 88
print "Is:: ", ma.ndim, ma.shape, ma.count_nonzero(), ma[3, 4]

ma -= 88.
print "Subtract:: ", ma.ndim, ma.shape, ma.count_nonzero(), ma[3, 4]

print ">>>>>>>>> id: ", id(ma)

print "\n\n ******************* inplace add smth else"
try:
    ma += 'lalala'
except TypeError:
    pass

print "\n\n ******************* __add__ smth else"

xxx = ma + 1.
print "xxx: ", xxx.ndim, xxx.shape, xxx.count_nonzero(), xxx[3, 4]
print "ma: ", ma.ndim, ma.shape, ma.count_nonzero(), ma[3, 4]
print "ids: ma, xxx", id(ma), id(xxx)

print "\n\n ******************* todense"

value = 0.
ma = MapArray()
#for i in (0, 2, 1):
#    for j in (0, 3, 4, 2, 1):
for i in (0, 1, 2):
    for j in (0, 1, 2, 3, 4):
        ma[i, j] = value
       # print(i, j, ' : ', value)
        value += 1


print ma.todense()


print "\n\n ******************* todense 2"

ma = MapArray()
ma[1, 1] = 1.
print ma.todense(), '\n'

ma[2, 3] = 2.
print ma.todense(), '\n'

ma[3, 4] = 3.
ma[0, 4] = -8
print ma.todense()


## multiply
#other = ma * 3
#print other.shape, other.count_nonzero()

## in-place mul
#o2 = ma + 3
