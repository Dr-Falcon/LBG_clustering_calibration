import numpy as np
import matplotlib.pyplot as plt

path = '../data/radiosky_new.dat'
#path = '../data/vlass_radio.dat'
#path = '../data/racs_radio.dat'
#path = '../data/radiosky_vlass_racs.dat'
#path = '../data/radiosky_vlass_racs_VLASSQL_compo1.dat'
#path = '../data/first_14dec17.dat'
# reading coordinates of radio galaxies
with open(path) as f:
    line = [s.strip() for s in f.readlines()]

print(len(line))
    
ra = np.zeros(len(line))
dec = np.zeros(len(line))
f = open(path)
for i in range (len(line)):
    list = f.readline()
    column = list.split()
    ra[i] = float(column[0])
    dec[i] = float(column[1])

n_rand = 10000000

print(max(ra),min(ra))
print(max(dec),min(dec))
    
ra_random = min(ra) + (max(ra)-min(ra))*np.random.rand(n_rand)
dec_random = min(dec) + (max(dec)-min(dec))*np.random.rand(n_rand)

#sort only ra_rand
#ra_random = sorted(ra_random)
fname = '../data/radiosky_random_v1.dat'
#fname = '../data/vlass_random_10M.dat'
#fname = '../data/radiosky_vlass_racs_random.dat'
#fname = '../data/radiosky_vlass_racs_VLASSQL_compo1_random.dat'
#fname = '../data/first_14dec17_random.dat'
f = open(fname,'w')
for i in range(n_rand):
    output = '{} {}\n'.format(ra_random[i],dec_random[i])
    f.write(output)
f.close()
