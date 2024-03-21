import NNClasses

mutRates = {
    "connections": 0.5,
    "link": 2.0,
    "bias": 0.3,
    "node": 0.5,
    "enable": 0.2,
    "disable": 0.4,
    "step": 0.1,
    "pu"
    "perturb": 0.9
}

newGenome = NNClasses.genome(mutRates)
print(NNClasses.network(newGenome).neurons)