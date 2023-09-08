"""
The Evolution Strategy can be summarized as the following term:
{mu/rho +, lambda}-ES

Here we use following term to find a maximum point.
{n_pop/n_pop + n_kid}-ES

Visit my tutorial website for more: https://mofanpy.com/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1             # DNA (real number)
DNA_BOUND = [0, 5]       # solution upper and lower bounds
N_GENERATIONS = 200
POP_SIZE = 100           # population size
N_KID = 50               # n kids per generation


def F(x): return np.sin(10*x)*x + np.cos(2*x)*x     # to find the maximum of this function


# find non-zero fitness for selection
def get_fitness(pred): return pred.flatten()


def make_kid(pop, n_kid):
    # store two DNA sequence with a dictionary
    kids = {'DNA': np.empty((n_kid, DNA_SIZE))}    # kids['DNA']:(50,1)
    kids['mut_strength'] = np.empty_like(kids['DNA'])  #kids['mut_strength']:(50,1)
    print(zip(kids['DNA'], kids['mut_strength']))
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        # crossover. initial kv=[0.], ks=[0.]
        p1, p2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False) # two different random numbers from 0~99
        cp = np.random.randint(0, 2, DNA_SIZE, dtype=np.bool_)  # crossover points    # cp1=[true] or cp2=[false]
        #attention: if x is an np.array x=[...], then x[False]=[], x[True]=[[...]], also : x[~cp1]=[], x[~cp2]=[...]
        #x[m,True]=[x[m]],x[m,~True]=[], x=[m,False]=[], x[m, ~False]=[x[m]]
        #So, when cp=[True]: line2 and line4 are useless since the left and the right
        kv[cp] = pop['DNA'][p1, cp]            #line1
        kv[~cp] = pop['DNA'][p2, ~cp]          # line2
        ks[cp] = pop['mut_strength'][p1, cp]   # line3
        ks[~cp] = pop['mut_strength'][p2, ~cp] # line4

        # mutate (change DNA based on normal distribution)
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape)-0.5), 0.)    # must > 0
        kv += ks * np.random.randn(*kv.shape)
        kv[:] = np.clip(kv, *DNA_BOUND)    # equals to kv[:]=np.clip(kv,0,5)   clip the mutated value
    return kids


def kill_bad(pop, kids):
    # put pop and kids together， now there are 101 individuals
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = get_fitness(F(pop['DNA']))            # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][-POP_SIZE:]   # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop


pop = dict(DNA=5 * np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),   # initialize the pop DNA values
           mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values

plt.ion()       # something about plotting
x = np.linspace(*DNA_BOUND, 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    # something about plotting
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(pop['DNA'], F(pop['DNA']), s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # ES part
    kids = make_kid(pop, N_KID)
    pop = kill_bad(pop, kids)   # keep some good parent for elitism

plt.ioff(); plt.show()