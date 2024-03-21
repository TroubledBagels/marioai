import math
import random

# CONSTANTS
INPUTNUM = 182  # (13 * 13) screen pixels + 12 status inputs (grounded, direction, etc.) + 1 bias
OUTPUTNUM = 4  # A, B, LEFT, RIGHT
MAXNODES = 1000000  # Big number


# CLASSES
class pool:  # The pool of the different species - the main class
    def __init__(self, config):  # Creates a new pool with no species
        self.species = []
        self.generation = 0
        self.innovation = 4
        self.currentSpecies = 0
        self.currentGenome = 0
        self.currentFrame = 0
        self.maxFitness = 0
        self.population = config["population"]
        self.baseMutRates = config["mutRates"]
        self.deltaDisjoint = config["deltaDisjoint"]
        self.deltaWeights = config["deltaWeights"]
        self.deltaThreshold = config["deltaThreshold"]
        self.staleSpeciesLimit = config["staleSpecies"]
        self.mutChance = config["mutChance"]
        self.crossoverChance = config["crossoverChance"]
        self.stepSize = config["stepSize"]
        self.timeout = config["timeout"]
        self.maxNodes = config["maxNodes"]

    def newInnovation(self):  # Increments innovation and returns - used for creating new genes
        self.innovation += 1  # Sort of like an iteration number
        return self.innovation

    def rankGlobally(self):  # Ranks all the species globally, used for comparisons
        globalRanks = []
        for i in range(len(self.species)):
            tempSpecies = self.species[i]  # Takes each species
            for j in range(len(tempSpecies.genomes)):
                globalRanks.append(tempSpecies.genomes[j])  # Adds each genome to the global ranks

        globalRanks.sort(key=lambda x: x.fitness)  # Sorts the global ranks by fitness

        for i in range(len(globalRanks)):
            globalRanks[i].globalRank = i  # Assigns a global rank to each genome

    def totalAverageFitness(self):  # Sums the average fitness of all species
        total = 0
        for i in range(len(self.species)):
            tempSpecies = self.species[i]
            total += tempSpecies.averageFitness

        return total

    def cullSpecies(self, cutToOne):  # Removes the bottom half of each species (or all but the top genome)
        for i in range(len(self.species)):
            tempSpecies = self.species[i]
            tempSpecies.genomes.sort(key=lambda x: x.fitness)  # Sorts the genomes by fitness
            tempSpecies.genomes = tempSpecies.genomes[:math.ceil(len(tempSpecies.genomes)/2)]  # Removes the bottom half
            if cutToOne:
                tempSpecies.genomes = tempSpecies.genomes[:1]  # Removes all but the top genome

    # Note: Staleness is the number of generations a species has gone without improvement
    def removeStaleSpecies(self):  # Removes the stale species
        survived = []

        for i in range(len(self.species)):  # For each species
            tempSpecies = self.species[i]  # Takes the species

            tempSpecies.genomes.sort(key=lambda x: x.fitness)  # Sorts by fitness

            if tempSpecies.genomes[0].fitness > tempSpecies.topFitness:  # If the top genome is better than the species' top
                tempSpecies.topFitness = tempSpecies.genomes[0].fitness  # Sets the species' top to the genome's fitness
                tempSpecies.staleness = 0  # Resets the staleness
            else:
                tempSpecies.staleness += 1  # Otherwise, increments the staleness

            if tempSpecies.staleness < self.staleSpeciesLimit or tempSpecies.topFitness >= self.maxFitness:  # If the species is not stale
                survived.append(i)  # Adds the species to the list of survivors

        self.species = [self.species[i] for i in survived]  # Removes the stale species

    # A species that is deemed weak is one that has a fitness less than the sum * population - may need to change
    def removeWeakSpecies(self):
        survived = []

        sum = self.totalAverageFitness()  # Sums the average fitness of all species

        for i in range(len(self.species)):  # For each species
            tempSpecies = self.species[i]  # Takes the species
            breed = math.floor(tempSpecies.averageFitness / sum * self.population)  # Calculates the number of children the species can have
            if breed >= 1:
                survived.append(i)  # Adds the species to the list of survivors

        self.species = [self.species[i] for i in survived]  # Removes the weak species

    def addToSpecies(self, child):  # Adds a genome to a species
        foundSpecies = False  # Boolean to check if the species has been found
        for i in range(len(self.species)):  # For each species
            tempSpecies = self.species[i]  # Takes the species
            if not foundSpecies and sameSpecies(child, tempSpecies.genomes[0], self.deltaDisjoint, self.deltaWeights, self.deltaThreshold):
                tempSpecies.genomes.append(child)  # Adds the genome to the species if it is the same species
                foundSpecies = True  # Sets the boolean to true

        if not foundSpecies:  # If the species has not been found
            childSpecies = species()  # Creates a new species
            childSpecies.genomes.append(child)  # Adds the genome to the species
            self.species.append(childSpecies)  # Adds the species to the pool

    def newGeneration(self):  # Creates a new generation
        self.cullSpecies(False)  # First removes the bottom half of each species
        self.rankGlobally()  # Ranks all the remaining species globally
        self.removeStaleSpecies()  # Takes those and removes the stale species
        self.rankGlobally()  # Ranks the remaining globally again

        for i in range(len(self.species)):  # For each remaining species
            self.species[i].calculateAverageFitness()  # Calculates the average fitness of the species

        self.removeWeakSpecies()  # Removes the weak species

        sum = self.totalAverageFitness()  # Sums the average fitness of all leftover species

        children = []
        for i in range(len(self.species)):  # For each leftover species
            tempSpecies = self.species[i]
            breed = math.floor(tempSpecies.averageFitness / sum * self.population) - 1  # Calculates the number of children the species can have
            for j in range(breed):  # For each possible child
                children.append(tempSpecies.breedChild(self.crossoverChance))  # Create and add the child

        self.cullSpecies(True)  # Removes all but the top genome of each species

        while len(children) + len(self.species) < self.population:  # While the number of children and species is less than the population
            tempSpecies = self.species[random.randint(0, len(self.species) - 1)]  # Randomly selects a species
            children.append(tempSpecies.breedChild(self.crossoverChance))  # Creates and adds a child for said species

        for i in range(len(children)):  # For each child
            self.addToSpecies(children[i])  # Adds the child to a species

        self.generation += 1  # Increments the generation

    def evaluateCurrent(self, inputs):  # Calculates the output of the current genome
        print("Evaluating species: " + str(self.currentSpecies) + ", genome:", self.currentGenome)
        tempSpecies = self.species[self.currentSpecies]  # Takes the current species
        tempGenome = tempSpecies.genomes[self.currentGenome]  # Takes the current genome

        controller = tempGenome.network.evaluate(inputs)  # Evaluates the genome with the inputs

        if controller[2] and controller[3]:  # If both left and right are pressed
            controller[2] = 0  # Set left to 0
            controller[3] = 0  # Set right to 0

        return controller  # Returns the controller

    def nextGenome(self):  # Moves to the next genome
        self.currentGenome += 1  # Increments the current genome
        if self.currentGenome > len(self.species[self.currentSpecies].genomes):  # Checks for overflow
            self.currentGenome = 0
            self.currentSpecies += 1  # Increments the current species
            if self.currentSpecies > len(self.species):  # Checks for overflow in the species
                self.newGeneration()  # If at the end, creates a new generation
                self.currentSpecies = 0

    def fitnessAlreadyMeasured(self):  # Checks if the fitness has already been measured
        tempSpecies = self.species[self.currentSpecies]  # Takes the current species
        tempGenome = tempSpecies.genomes[self.currentGenome]  # Takes the current genome
        return tempGenome.fitness != 0  # Returns if the fitness has been measured

    def writeToFile(self, filename):  # Writes the pool to a file - long and complicated
        file = open(filename, "w")
        file.write(str(self.generation) + "\n")
        file.write(str(self.maxFitness) + "\n")
        file.write(str(len(self.species)) + "\n")
        for i in range(len(self.species)):
            file.write(str(self.species[i].topFitness) + "\n")
            file.write(str(self.species[i].staleness) + "\n")
            file.write(str(len(self.species[i].genomes)) + "\n")
            for j in range(len(self.species[i].genomes)):
                file.write(str(self.species[i].genomes[j].fitness) + "\n")
                file.write(str(self.species[i].genomes[j].maxneuron) + "\n")
                for mut in self.species[i].genomes[j].mutationRates:
                    file.write(mut + "\n")
                    file.write(str(self.species[i].genomes[j].mutationRates[mut]) + "\n")
                file.write("done\n")

                file.write(str(len(self.species[i].genomes[j].genes)) + "\n")
                for k in range(len(self.species[i].genomes[j].genes)):
                    file.write(str(self.species[i].genomes[j].genes[k].into) + "\n")
                    file.write(str(self.species[i].genomes[j].genes[k].out) + "\n")
                    file.write(str(self.species[i].genomes[j].genes[k].weight) + "\n")
                    file.write(str(self.species[i].genomes[j].genes[k].innovation) + "\n")
                    file.write(str(int(self.species[i].genomes[j].genes[k].enabled)) + "\n")
        file.close()


class species:  # The species class
    def __init__(self):  # Creates a new species with no genomes
        self.topFitness = 0
        self.staleness = 0
        self.genomes = []
        self.averageFitness = 0

    def calculateAverageFitness(self):  # Calculates the average fitness of the species
        total = 0
        for i in range(len(self.genomes)):
            total += self.genomes[i].fitness  # Takes each genome's fitness and adds it to the total

        div = len(self.genomes)  # Divides the total by the number of genomes
        if div == 0:
            div = math.inf  # If there are no genomes, sets the division to infinity

        self.averageFitness = total / div  # Divides the total by the number of genomes

    def breedChild(self, xOverChance):  # Breeds a child based on the crossover chance
        if random.random() < xOverChance:  # If the random number is less than the crossover chance, breeds a child from 2 parents
            g1 = self.genomes[random.randint(0, len(self.genomes) - 1)]  # Randomly selects the first parent
            g2 = self.genomes[random.randint(0, len(self.genomes) - 1)]  # Randomly selects the second parent
            child = crossover(g1, g2)  # Creates a child from the 2 parents
        else:  # Otherwise
            g = self.genomes[random.randint(0, len(self.genomes) - 1)]
            child = g.createCopy()  # Creates a copy of a random genome

        child.mutate()  # Mutates the child
        return child  # Returns the child

class genome:  # The genome class
    def __init__(self, mutRates):  # Creates a new genome with no genes, and the given mutation rates
        self.genes = []
        self.fitness = 0
        self.adjustedFitness = 0
        self.network = None
        self.maxneuron = 0
        self.globalRank = 0
        self.mutationRates = mutRates

    def createCopy(self):  # Creates a copy of the genome
        newGenome = genome(self.mutationRates)  # Creates a new genome with the same mutation rates

        for i in range(len(self.genes)):
            newGenome.genes.append(self.genes[i].createCopy())  # Adds a copy of each gene to the new genome

        newGenome.maxneuron = self.maxneuron  # Sets the max neuron to the same as the original genome

        return newGenome  # Returns the new genome

    def mutate(self, curPool):  # Mutates the genome
        for mutation in self.mutationRates:  # For each mutation rate
            if random.random() < 0.5:  # If the random number is less than 0.5
                self.mutationRates[mutation] *= 0.95  # Decreases the mutation rate by 5%
            else:  # Otherwise
                self.mutationRates[mutation] *= 1.05263 # Increases the mutation rate by 5.263% (i.e. decrease by -5%)

        if random.random() < self.mutationRates["connections"]:  # If a generated random number is less than the connection mutation rate
            self.pointMutate()  # Mutates the genome by changing the weight of a gene

        p = self.mutationRates["link"]  # Sets p to the link mutation rate
        while p > 0:  # While p is greater than 0
            if random.random() < p:  # If a generated random number is less than p
                self.linkMutate(False, curPool)  # Mutates the genome by adding a new gene
            p -= 1  # Decrements p

        p = self.mutationRates["bias"]  # Sets p to the bias mutation rate
        while p > 0:  # While p is greater than 0
            if random.random() < p:  # If a generated random number is less than p
                self.linkMutate(True, curPool)  # Mutates the genome by adding a new gene with a bias
            p -= 1  # Decrements p

        p = self.mutationRates["node"]  # Sets p to the node mutation rate
        while p > 0:  # While p is greater than 0
            if random.random() < p:  # If a generated random number is less than p
                self.nodeMutate(curPool)  # Mutates the genome by adding a new node
            p -= 1  # Decrements p

        p = self.mutationRates["enable"]  # Sets p to the enable mutation rate
        while p > 0:  # While p is greater than 0
            if random.random() < p:  # If a generated random number is less than p
                self.enableDisableMutate(True)  # Mutates the genome by enabling a gene
            p -= 1  # Decrements p

        p = self.mutationRates["disable"]  # Sets p to the disable mutation rate
        while p > 0:  # While p is greater than 0
            if random.random() < p:  # If a generated random number is less than p
                self.enableDisableMutate(False)  # Mutates the genome by disabling a gene
            p -= 1  # Decrements p

    def sortByOut(self):
        self.genes.sort(key=lambda x: x.out)  # Sorts the genes by their out value

    '''
        In the following section, gaussian distributions are used to change the weights of the genes
        Gaussian distributions are used as they are a good way to model the distribution of weights, and avoid very
        large changes, which could be detrimental to the genome (i.e. going from never pressing A to always pressing A)
    '''
    def pointMutate(self):  # Mutates the genome by changing the weight of a gene
        step = self.mutationRates["step"]  # Sets step to the step mutation rate

        for i in range(len(self.genes)):  # For each gene
            if random.random() < self.mutationRates["perturb"]:  # If a generated random number is less than the perturb mutation rate
                self.genes[i].weight += random.gauss(0, 1) * step*2 - step  # Perturbs the weight of the gene
                '''
                    The gaussian value is multiplied by 2 and the step as the step size determines the gradient of the
                    change of the weight, and the gaussian value determines the size of the change
                '''
            else:
                self.genes[i].weight = random.gauss(0, 1)*4 - 2 # Otherwise, sets the weight to a new random value

    def linkMutate(self, forceBias, curPool):  # Mutates the genome by adding a new gene
        neuron1 = RandomNeuron(self.genes, False)  # Randomly selects a neuron
        neuron2 = RandomNeuron(self.genes, True)  # Randomly selects an output neuron (if possible)

        print("Neurons selected: ", neuron1, neuron2)  # Prints the selected neurons

        newLink = gene()  # Creates a new gene

        # Note: output neurons always have a value > MAXNODES
        if neuron1 <= INPUTNUM and neuron2 <= INPUTNUM:  # If both neurons are input neurons, prints and returns
            print("Both neurons are input neurons")
            return

        if neuron2 <= INPUTNUM:  # If the second neuron is an input neuron
            temp = neuron1  # Swaps the neurons
            neuron1 = neuron2
            neuron2 = temp

        newLink.into = neuron1  # Sets the into value of the new gene to the first neuron
        newLink.out = neuron2  # Sets the out value of the new gene to the second neuron
        if forceBias:  # If the gene is a bias gene
            newLink.into = INPUTNUM  # Sets the into value to the bias neuron

        if containsLink(self.genes, newLink):  # If the gene already exists, no need to add it
            return

        newLink.innovation = curPool.newInnovation()  # Sets the innovation of the new gene to the next innovation
        newLink.weight = random.gauss(0, 1)*4 - 2  # Sets the weight of the new gene to a random value

        self.genes.append(newLink)  # Adds the new gene to the genome

    def nodeMutate(self, curPool):  # Mutates the genome by adding a new node
        if len(self.genes) == 0:  # If there are no genes, prints and returns
            return

        self.maxneuron += 1  # Increments the max neuron

        tempGene = self.genes[random.randint(0, len(self.genes) - 1)]  # Randomly selects a gene
        if not tempGene.enabled:  # If the gene is not enabled, no need to add a new node
            return
        tempGene.enabled = False  # Disables the gene

        gene1 = tempGene.createCopy()  # Creates a copy of the gene
        gene1.out = self.maxneuron  # Sets the out value of the gene to the max neuron
        gene1.weight = 1.0  # Sets the weight of the gene to 1.0
        gene1.innovation = curPool.newInnovation()  # Sets the innovation of the gene to the next innovation
        gene1.enabled = True  # Enables the gene
        self.genes.append(gene1)  # Adds the gene to the genome

        gene2 = tempGene.createCopy()  # Creates a copy of the gene
        gene2.into = self.maxneuron  # Sets the into value of the gene to the max neuron
        gene2.innovation = curPool.newInnovation()  # Sets the innovation of the gene to the next innovation
        gene2.enabled = True  # Enables the gene
        self.genes.append(gene2)  # Adds the gene to the genome
        # Note: the gene's weight is not changed at all from the copy

    def enableDisableMutate(self, enable):  # Mutates the genome by essentially toggling a gene
        candidates = []  # Creates a list of candidates
        for i in range(len(self.genes)):  # For each gene
            if self.genes[i].enabled != enable:  # If the gene does not have the same enabled value as the parameter
                candidates.append(self.genes[i])  # Adds the gene to the list of candidates
        # This means that the list of candidates will all be possible to enable or disable, depending on the parameter

        if len(candidates) == 0:  # If there are no candidates, returns as there is nothing to enable or disable
            return

        tempGene = candidates[random.randint(0, len(candidates) - 1)]  # Randomly selects a candidate
        tempGene.enabled = not tempGene.enabled  # Toggles the enabled value of the gene

    '''
        Input format:
        index - meaning
            0 - Grounded, 0: grounded, 1: in air from jump, 2: in air from fall, 3: in air from flagpole
            1 - Direction, 0: not on screen, 1: left, 2: right
            2 - HSpeed, float
            3 - Move Direction: 1: right, 2: left
            4 - Player X, int
            5 - Player Y, int
            6 - Swimming, 0: swimming, 1: not swimming
            7 - Powerup State, 0: small, 1: big, 2: fire
            8 - Frame, int (0-255)
            9 - Player State, 0: leftmost, 1: climbing vine, 2: entering pipe, 3: going down pipe, 4 and 5: autowalk, 6: dies, 7: entering area, 8: normal, 9: turning big, 10: turning small, 11: dying, 12: turning fire 
           10 - Fallen off screen, 0: not fallen, >1: fallen
           11 - Dead: 0: not dead, 1: dead - value derived from 10 and 9
    '''
    def updateFitness(self, inputs, curPool):
        if inputs[11] == 1:
            self.fitness = 0
            return

        curFitness = inputs[4] - inputs[9]

        if inputs[0] == 3:
            curFitness += 1000


        self.fitness = curFitness
        if self.fitness > curPool.maxFitness:
            curPool.maxFitness = self.fitness
        pass


class gene:  # The gene class
    def __init__(self):  # Creates a new gene with no values, but is enabled
        self.into = 0
        self.out = 0
        self.weight = 0
        self.enabled = True
        self.innovation = 0

    def createCopy(self):  # Creates a copy of the gene
        newGene = gene()  # Creates a new gene
        newGene.into = self.into  # Copies all the values of the original gene to the new gene
        newGene.out = self.out
        newGene.weight = self.weight
        newGene.enabled = self.enabled
        newGene.innovation = self.innovation

        return newGene # Returns the new gene


class neuron:  # The neuron class - not much to it
    def __init__(self):
        self.incoming = []  # The incoming genes
        self.value = 0  # The value of the neuron


class network:  # The network class
    def __init__(self, inputGenome):  # Creates a new network of neurons within a genome
        self.neurons = {}  # The dict of neurons

        for i in range(INPUTNUM):  # For each input value, creates a new neuron and adds it to the dict
            self.neurons[i] = neuron()

        for i in range(OUTPUTNUM):  # For each output value, creates a new neuron and adds it to the dict
            self.neurons[MAXNODES + i] = neuron()  # Note: the value of the neuron is not set

        inputGenome.sortByOut()  # Sorts the genes by their out value

        for i in range(len(inputGenome.genes)):  # For each gene
            tempGene = inputGenome.genes[i]  # Takes the gene
            if tempGene.enabled:  # If the gene is enabled
                if self.neurons[tempGene.out] is None:  # If the neuron's out value is not in the dict
                    self.neurons[tempGene.out] = neuron()  # Adds a new neuron at that out value to the dict
                self.neurons[tempGene.out].incoming.append(tempGene)  # Adds the gene to the incoming genes of the neuron
                if self.neurons[tempGene.into] is None:  # If the neuron's into value is not in the dict
                    self.neurons[tempGene.into] = neuron()  # Adds a new neuron at that into value to the dict

        inputGenome.network = self  # Sets the genome's network to the new network

    def evaluate(self, inputs):  # Evaluates the network with the given inputs
        inputs.append(1)  # Adds a bias input
        if len(inputs) != INPUTNUM:  # If the number of inputs is not the same as the number of input neurons
            print("Error: Incorrect number of inputs")  # Prints an error message
            return  # Returns

        for i in range(INPUTNUM):  # For each input neuron
            self.neurons[i].value = inputs[i]  # Sets the value of the respective neuron to the input value

        for iNeuron in self.neurons:  # For each neuron
            sum = 0  # Sets the sum to 0
            for iGene in iNeuron.incoming:  # For each incoming gene
                tempIncoming = self.neurons[iGene.into].incoming[iGene]  # Takes the incoming gene
                other = self.neurons[tempIncoming.into]  # Takes the incoming neuron
                sum += tempIncoming.weight * other.value  # Adds the weight of the gene multiplied by the value of the neuron to the sum

            if len(iNeuron.incoming) > 0:  # If the neuron has incoming genes
                iNeuron.value = sigmoid(sum)  # Sets the value of the neuron to the sigmoid of the sum

        outputs = []
        for i in range(OUTPUTNUM):  # For each output neuron
            if self.neurons[INPUTNUM + i].value > 0:  # If the value of the output neuron is greater than 0
                outputs.append(True)  # Adds True to the outputs
            else:  # Otherwise
                outputs.append(False)  # Adds False to the outputs

        return outputs  # Returns the outputs - should be a list of 4 bools, representing A, B, LEFT, RIGHT


'''
    The following functions are global, in that they do not directly relate to a class
    
    Some of these functions could be placed into a class, however it would make the code far harder to read and 
    understand
    
    These functions normally create, modify, compare or do something else with species, genomes, genes, etc.
'''


def sigmoid(x):  # Simple sigmoid function
    return 2 / (1 + math.exp(-4.9 * x)) - 1


def CreateBasicGenome(mutRates, curPool):  # Creates a basic genome with the given mutation rates
    basicGenome = genome(mutRates)  # Creates a new genome with the given mutation rates
    basicGenome.innovation = 1  # Sets the innovation to 1

    basicGenome.maxneuron = INPUTNUM  # Sets the max neuron to the number of input neurons
    basicGenome.mutate(curPool)  # Mutates the genome

    return basicGenome  # Returns the genome


def crossover(g1, g2):  # Crosses over 2 genomes to create a child
    if g2.fitness > g1.fitness:  # Puts the genomes in fitness order if they are not already
        temp = g1
        g1 = g2
        g2 = temp

    child = genome(g1.mutationRates)  # Creates a new genome with the mutation rates of the first genome

    innovations2 = {}  # Creates a dict of innovations
    for i in range(len(g2.genes)):  # For each gene in the second genome
        tempGene = g2.genes[i]
        innovations2[tempGene.innovation] = tempGene  # Adds the gene to the dict

    for i in range(len(g1.genes)):  # For each gene in the first genome
        tempGene = g1.genes[i]  # Takes the gene

        if tempGene.innovation not in innovations2:  # If the value is not in the dict
            child.genes.append(tempGene.createCopy())  # Adds a copy of the first gene to the child

        tempGene2 = innovations2[tempGene.innovation]  # Takes the gene with the same innovation from the second genome
        if random.randint(0, 1) == 0 and tempGene2.enabled:  # If a random number is 0 and the gene is enabled
            child.genes.append(tempGene2.createCopy())  # Adds a copy of the second gene to the child
        else:
            child.genes.append(tempGene.createCopy())  # Otherwise, adds a copy of the first gene to the child

    child.maxneuron = max(g1.maxneuron, g2.maxneuron)  # Sets the max neuron to the maximum of the 2 genomes (i.e. the larger genome)

    for mutation in g1.mutationRates:  # For each mutation
        child.mutationRates[mutation] = g1.mutationRates[mutation]  # Sets the mutation rate of the child to the mutation rate of the first genome

    return child  # Returns the child


def RandomNeuron(genes, nonInput):  # Randomly selects a neuron, with the option to exclude input neurons
    neurons = {}  # Creates a dict of neurons
    if not nonInput:  # If input neurons are not to be excluded
        for i in range(INPUTNUM):
            neurons[i] = True  # Adds the input neurons to the dict

    for i in range(OUTPUTNUM):  # For each output neuron
        neurons[MAXNODES + i] = True  # Adds the output neurons to the dict

    print("Neuron length: ", len(neurons))  # Prints the length of the dict (debugging)

    for i in range(len(genes)):  # For each gene
        if (not nonInput) or genes[i].into > INPUTNUM:  # If input neurons are not to be excluded, or the gene's into value is greater than the number of input neurons
            neurons[genes[i].into] = True  # Adds the gene's into value to the dict
        if (not nonInput) or genes[i].out > INPUTNUM:  # If input neurons are not to be excluded, or the gene's out value is greater than the number of input neurons
            neurons[genes[i].out] = True  # Adds the gene's out value to the dict

    neurons = list(neurons.keys())  # Converts the dict to a list of only the numbers - i.e. the positions of the neurons

    count = len(neurons)  # Takes the length of the list

    n = random.randint(0, count - 1)  # Randomly selects a neuron
    '''for i in range(len(neurons)):  # For each neuron
        n -= 1
        if n == 0:
            return neurons[i]'''

    return neurons[n]  # Returns the selected neuron


def containsLink(genes, link):  # Checks if a gene is already in the genome
    for i in range(len(genes)):  # For each gene
        if genes[i].into == link.into and genes[i].out == link.out:  # If the gene's into and out values are the same as the link's
            return True  # Returns True
    return False  # Otherwise, returns False


def disjoint(genes1, genes2):  # Calculates the number of disjoint genes between 2 genomes, i.e. those they do not share
    i1 = {}  # Creates a dict of innovations for the first genome
    for i in range(len(genes1)):  # For each gene in the first genome
        gene = genes1[i]  # Takes the gene
        i1[gene.innovation] = True  # Adds the gene's innovation to the dict

    i2 = {}  # Creates a dict of innovations for the second genome
    for i in range(len(genes2)):  # Does the same for the second genome
        gene = genes2[i]
        i2[gene.innovation] = True

    disjointGenes = 0  # Sets the number of disjoint genes to 0
    for i in range(len(genes1)):  # For each gene in the first genome
        gene = genes1[i]  # Takes the gene
        if gene.innovation not in i2:  # If the gene's innovation is not in the second genome
            disjointGenes += 1  # Increments the number of disjoint genes

    for i in range(len(genes2)):  # Does the same for the second genome
        gene = genes2[i]
        if gene.innovation not in i1:
            disjointGenes += 1

    div = max(len(genes1), len(genes2))  # Selects the largest genome to be the divisor
    if div == 0:
        return math.inf  # If there are no genes, returns infinity to avoid division by 0

    return disjointGenes / div  # Returns the number of disjoint genes divided by the divisor


def weights(genes1, genes2):  # Calculates the average weight difference between 2 genomes
    i2 = {}  # Creates a dict of innovations for the second genome
    for i in range(len(genes2)):  # For each gene in the second genome
        tempGene = genes2[i]  # Takes the gene
        i2[tempGene.innovation] = tempGene  # Adds the gene to the dict

    sum = 0  # Sets the sum to 0
    coincident = 0  # Sets the number of coincident genes to 0
    for i in range(len(genes1)):  # For each gene in the first genome
        tempGene = genes1[i]  # Takes the gene

        if tempGene.innovation in i2:  # If the gene's innovation is in the second genome
            gene2 = i2[tempGene.innovation]  # Takes the gene from the second genome
            sum += abs(tempGene.weight - gene2.weight)  # Adds the absolute difference of the weights to the sum
            coincident += 1  # Increments the number of coincident genes

    if coincident == 0:
        return math.inf  # If there are no coincident genes, returns infinity to avoid division by 0

    return sum / coincident  # Returns the sum divided by the number of coincident genes


def sameSpecies(genome1, genome2, deltaDisjoint, deltaWeights, deltaThreshold):  # Checks if 2 genomes are the same species
    dd = deltaDisjoint * disjoint(genome1.genes, genome2.genes)  # Calculates the number of disjoint genes multiplied by the delta disjoint
    dw = deltaWeights * weights(genome1.genes, genome2.genes)  # Calculates the average weight difference multiplied by the delta weights
    return dd + dw < deltaThreshold  # Returns if the sum of the 2 values is less than the delta threshold


def initialisePool(config):  # Initialises the pool with the given config
    newPool = pool(config)  # Creates a new pool with the given config
    pop = config["population"]  # Takes the population from the config
    for i in range(pop):  # For each genome
        print("Initialising genome", i)  # Prints the genome number (debugging)
        basic = CreateBasicGenome(config["mutRates"], newPool)  # Creates a basic genome with the given mutation rates
        print("Genome created with,", len(basic.genes), "genes")  # Prints the number of genes in the genome (debugging)
        newPool.addToSpecies(basic)  # Adds the genome to a species
    return newPool  # Returns the pool


def loadFromFile(filename):  # Long load from txt file function - do not worry - treat as black box as is disgusting - can be understood if thoroughly read through
    file = open(filename, "r")
    newPool = pool()
    newPool.generation = int(file.readline())
    newPool.maxFitness = int(file.readline())
    numSpecies = int(file.readline())
    for i in range(numSpecies):
        tempSpecies = species()
        tempSpecies.topFitness = int(file.readline())
        tempSpecies.staleness = int(file.readline())
        numGenomes = int(file.readline())
        for j in range(numGenomes):
            tempGenome = genome()
            tempGenome.fitness = int(file.readline())
            tempGenome.maxneuron = int(file.readline())
            line = file.readline()
            while line != "done":
                tempGenome.mutationRates[line] = float(file.readline())
                line = file.readline()
            numGenes = int(file.readline())
            for k in range(numGenes):
                tempGene = gene()
                tempGene.into = int(file.readline())
                tempGene.out = int(file.readline())
                tempGene.weight = float(file.readline())
                tempGene.innovation = int(file.readline())
                tempGene.enabled = bool(int(file.readline()))
                tempGenome.genes.append(tempGene)
            tempSpecies.genomes.append(tempGenome)
        newPool.species.append(tempSpecies)

    file.close()

    while newPool.fitnessAlreadyMeasured():
        newPool.nextGenome()

    return newPool