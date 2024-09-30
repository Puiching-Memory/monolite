from annoy import AnnoyIndex
import random

f = 40  # Length of item vector that will be indexed

t = AnnoyIndex(f, 'euclidean')
for index,i in enumerate(range(1000)):
    print(index)
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

# ...

u = AnnoyIndex(f, 'euclidean')
u.load('test.ann') # super fast, will just mmap the file
print(u.get_nns_by_item(0, 2)) # will find the 1000 nearest neighbors