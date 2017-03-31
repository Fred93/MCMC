import numpy as np
import matplotlib.pyplot as plt

#Parameters for proposal distribution
startMean = 0
proposalVariance = 5

iterations = 1000000
burnin = 1000


def target(x):
    return (0.3*np.exp(-0.2*x*x)+0.7*np.exp(-0.2*(x-10)*(x-10)))

def norm(z, mean, var):
    return np.exp(-(z-mean)*(z-mean)/2*var)/np.sqrt(2*np.pi*var)

x = np.array(range(-5,15))
y = target(x)
y = y/sum(y)
plt.plot(x,y)

x = startMean
sample=[]
counter = 0
for i in range(iterations + burnin):
    cand = np.random.normal(x, proposalVariance)
    u = np.random.uniform(0,1)
    prob = (target(cand)*norm(x, cand, proposalVariance))/(target(x)*norm(cand, x, proposalVariance))
    if (prob>u):
        ##Accept candidate
        x = cand

        if i >= burnin:
            counter += 1

    #Burn in phase
    if (i>=burnin):
        sample.append(x)

plt.hist(sample, 100, alpha=0.6, label='Iterations:' + str(iterations)+'\nVariance: ' + str(proposalVariance)+'\nacceptance rate: ' + str((1.0*counter)/iterations), normed=True, color=(1,0,0))
plt.legend(loc='upper left',prop={'size':12})
plt.show()