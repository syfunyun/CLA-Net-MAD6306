# MAD 6306 Complex Networks - Dr. Jia Liu: jliu@uwf.edu
# Written by Sy Fontenot: jdf52@students.uwf.edu

# Paper: "A cellular learning automata based algorithm for detecting community structure in complex networks"
# Authors: Yuxin Zhao, Wen Jiang, Shenghong Li, Yinghua Ma, Guiyang Su, Xiang Lin
# dio: https://doi.org/10.1016/j.neucom.2014.04.087

# Pseudo code #

# Input
    # Anxn : the adjacency matrix of the network G = (V, E), where V is the set of the nodes, E is the set of the edges
    #        connecting the nodes and n = |V| is the number of nodes in the network. Aij = 1 if node i and node j are
    #        directly connected; otherwise, Aij = 0.

# Parameters
    # a : the reward parameter for the update of the action probability vector, where 0 < a < 1.

# Variables
    # ri : the number of the actions for learning automaton Li, which is equal to the degree of node i in the network.
    # Wij(t) : the number of times the jth action of learning automaton Li has been rewarded up to cycle t,
    #          with 1 <= i <= n and 1 <= j <= ri.
    # Zij(t) : the number of times the jth action of learning automaton Li has been chosen up to cycle t,
    #          with 1 <= i <= n and 1 <= j <= ri.
    # Qbest : The largest modularity obtained in the past cycles.

# Method

# Initialization
    # Step 1: pij = 1 / ri, for 1 <= i <= n and 1 <= j <= ri.
    # Step 2: Initialize Wij(t), Zij(t) and Qbest by randomly choosing the action for each learning automaton Li a
    #         small number of times, with 1 <= i <= n and 1 <= j <= ri.
    # repeat:
        # Step 3: Each learning automaton Li chooses an action αi(t) according to its action probability vector pi(t),
        #         for 1 <= i <= n.
        # Step 4: The solution vector S(t) = (α1(t), α2(t), …, αn(t)) is transferred into the  membership vector
        #         C(t) = (c1(t), c2(t), …, cn(t)) to represent the obtained community structure through the decoding
        #         process.
        # Step 5: The global environment calculates the modularity Q(t) of the community structure represented by the
        #         membership vector C(t).
        # Step 6:
        #   for each learning automaton Li(1 <= i <= n) do
        #       if Q(t) >= Qbest and ki(ci(t)) >= ki(c'), ALLc' != ci(t) then
        #           The response from the environments βi(t) = 0
        #       else
        #           The response from the environments βi(t) = 1
        #       end if
        #   end for
        # Step 7: update Qbest = max(Q(t), Qbest).
        # Step 8: For each learning automaton Li(1 <= i <= n), given the chosen action αi(t) = αiq, update Wiq(t) and
        #         Ziq(t) according to the following equations:
        #             Wiq(t) = Wiq(t - 1) + (1 - βi(t))
        #             Ziq(t) = Ziq(t - 1) + 1
        # Step 9: The current optimal action of each learning automaton Li is estimated according to Eq. (7), for
        #         1 <= i <= n.
        # Step 10: Update the action probability vector pi of each learning automaton Li according to Eq. (8), for
        #          1 <= i <= n.
    # until: The obtained community structure remains fixed in some consecutive cycles.

# Output
    # The solution vector S(t), the membership vector C(t) and the corresponding community structure in the network.

# Equaitons
    # Eq. (7) : Di(t) = Wi(t) / Zi(t) for i = 1, 2, …, r
    # Eq. (8) : pj(t + 1) = { pj(t) + a(1 - pj(t))  if j = m
    #                       { (1 - a)pj(t)          if j != m

import numpy as np
import networkx as nx
import scipy as sp
import random
import time

# TODO: Comment and explain code
# TODO: Terminate after consecutive results
# TODO [Future]: Determine best reward (quickest average convergence)

def findModularity(membershipVector, adjacencyMatrix):
    size = adjacencyMatrix.shape[0]
    degreeCount = adjacencyMatrix.sum() # Twice the edges (2m)
    rowSum = adjacencyMatrix.sum(axis = 0)
    summation = 0

    for col in range(size):
        for row in range(size):
            if (membershipVector[row] - membershipVector[col]) == 0:
                summation += adjacencyMatrix.item((row, col)) - rowSum.item(row) * rowSum.item(col) / (degreeCount)

    return summation / (degreeCount)

def solutionToMembershipVector(solutionVector):
    size = len(solutionVector)
    membershipVector = np.full(size, size)

    for iteration in range(size):
        newCount = 0
        for index in range(size):
            nextIndex = index
            while membershipVector[solutionVector[nextIndex]] - membershipVector[nextIndex] != 0:
                minValue = min(membershipVector[nextIndex], membershipVector[solutionVector[nextIndex]])
                membershipVector[nextIndex] = minValue
                membershipVector[solutionVector[nextIndex]] = minValue
                nextIndex = solutionVector[nextIndex]
            if membershipVector[solutionVector[nextIndex]] == size:
                membershipVector[nextIndex] = newCount
                newCount += 1

    return membershipVector

def chooseActionVector(actionProbabilityMatrix):
    size = actionProbabilityMatrix.shape[0]
    actionVector = np.zeros(size, dtype = int)

    for node in range(size):
        colProbabilityVector = actionProbabilityMatrix[:, node]
        randomNumber = random.random()
        selectedAction = 0

        while colProbabilityVector[selectedAction] <= randomNumber:
            randomNumber -= colProbabilityVector[selectedAction]
            selectedAction += 1
        actionVector[node] = selectedAction

    return actionVector

def getCommunityStructure(membershipVector):
    communityCount = len(set(membershipVector))
    communityStruture = []

    for index in range(communityCount):
        communityStruture.append(np.where(np.array(membershipVector) == sorted(set(membershipVector))[index])[0].tolist())

    return communityStruture

# def compareCommunityStructure(membershipVector1, membershipVector2):
#     if len(membershipVector1) != len(membershipVector2):
#         print("Membership vectors of different length!")
#         return False

#     if len(set(membershipVector1)) != len(set(membershipVector2)):
#         print("Different number of communities!")
#         return False

#     communityCount = len(set(membershipVector1))
#     communityStruture = getCommunityStructure(membershipVector1)

#     for index in range(communityCount):
#         if np.where(np.array(membershipVector2) == sorted(set(membershipVector2))[index])[0].tolist() not in communityStruture:
#             print("Different community structure!")
#             return False
        
#     return True

def satisfiesRaghavan(node, membershipVector, adjacencyMatrix):
    if len(membershipVector) != adjacencyMatrix.shape[0]:
        print("Error: Sizes do not match!")
        exit(1)

    communityStruture = getCommunityStructure(membershipVector)
    inCommunity = []
    outCommunity = []
    for community in range(len(communityStruture)):
        if (node in communityStruture[community]):
            inCommunity = communityStruture[community]
        else:
            outCommunity.append(communityStruture[community])

    # Flatten
    inCommunity = np.ravel(inCommunity)
    if outCommunity:
        outCommunity = np.hstack(outCommunity)

    inSum = 0
    outSum = 0

    for inNode in inCommunity:
        inSum += adjacencyMatrix.item((node, inNode))
    for outNode in outCommunity:
        outSum += adjacencyMatrix.item((node, outNode))

    return inSum >= outSum

def CLAnet(adjacencyMatrix, reward, initialRuns, consecutiveRepeats):

    # Check Square Matrix
    matrixDim = adjacencyMatrix.shape
    if matrixDim[0] != matrixDim[1]:
        print("Error: Matrix not square!")
        exit(1)

    # Variables
    size = matrixDim[0]
    actionCountArray = adjacencyMatrix.sum(axis = 0)
    # actionRewardArray = np.zeros(size, dtype = int)
    actionRewardCountMatrix = np.zeros((size, size), dtype = int)
    actionChosenCountMatrix = np.zeros((size, size), dtype = int)
    bestModularity = 0
    # bestCommunityStructure = []
    # prevCommunityStructure = []

    # Step 1 #

    # Action Probability Vectors (as matrix)
    actionProbabilityMatrix = np.zeros((size, size))

    # Calculate initial action probabilities
    for col in range(size):
        for row in range(size):
            if adjacencyMatrix.item((row, col)) == 1:
                actionProbabilityMatrix[row][col] = 1 / actionCountArray[col]

    # Step 2 #
                
    for run in range(initialRuns):
        solutionVector = chooseActionVector(actionProbabilityMatrix)
        membershipVector = solutionToMembershipVector(solutionVector)
        modularity = findModularity(membershipVector, adjacencyMatrix)

        for node in range(size):
            actionChosenCountMatrix[node][solutionVector[node]] += 1
            if modularity >= bestModularity and satisfiesRaghavan(node, membershipVector, adjacencyMatrix):
                actionRewardCountMatrix[node][solutionVector[node]] += 1

        if modularity >= bestModularity:
            bestCommunityStructure = membershipVector
            bestModularity = modularity

    communityRepeatCount = 0

    while communityRepeatCount < consecutiveRepeats:
        
        # Step 3 #

        solutionVector = chooseActionVector(actionProbabilityMatrix)
        
        # Step 4 #

        membershipVector = solutionToMembershipVector(solutionVector)

        # Step 5 #
        
        modularity = findModularity(membershipVector, adjacencyMatrix)

        # Steps 6 & 8 #

        for node in range(size):
            actionChosenCountMatrix[solutionVector[node]][node] += 1
            if modularity >= bestModularity and satisfiesRaghavan(node, membershipVector, adjacencyMatrix):
                actionRewardCountMatrix[solutionVector[node]][node] += 1

        # Step 7 #

        if modularity >= bestModularity:
            bestCommunityStructure = membershipVector
            bestModularity = modularity

        # Step 9 #
            
        optimalEstimateMatrix = np.zeros((size, size))
        optimalActionVector = np.zeros(size, dtype = int)

        for col in range(size):
            for row in range(size):
                if adjacencyMatrix.item((row, col)) > 0:
                    if actionChosenCountMatrix[row][col] > 0:
                        optimalEstimateMatrix[row][col] = actionRewardCountMatrix[row][col] / actionChosenCountMatrix[row][col]
                    if optimalEstimateMatrix[row][col] >= optimalEstimateMatrix[optimalActionVector[col]][col]:
                        optimalActionVector[col] = row
            
        # Step 10 #

        for col in range(size):
            optimalAction = optimalActionVector[col]
            for row in range(size):
                probability = actionProbabilityMatrix[row][col]
                if row == optimalAction:
                    actionProbabilityMatrix[row][col] = probability + reward * (1 - probability)
                else:
                    actionProbabilityMatrix[row][col] = (1 -  reward) * probability

            
        # if prevCommunityStructure == membershipVector:
        #     communityRepeatCount += 1
        # else:
        #     communityRepeatCount = 0
        # print(communityRepeatCount, bestModularity)
        communityRepeatCount += 1
        # print(communityRepeatCount)
        # prevCommunityStructure = membershipVector
        # print(prevCommunityStructure)
        # print(modularity)

        # print(solutionToMembershipVector(optimalActionVector))

    # return solutionToMembershipVector(bestCommunityStructure)
    return bestModularity



aMatrix = np.matrix([[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]])

bMatrix = np.matrix([[0, 1, 0, 1],
                     [1, 0, 1, 1],
                     [0, 1, 0, 0],
                     [1, 1, 0, 0]])

cMatrix = np.matrix([[0, 1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1, 1],
                     [0, 0, 0, 1, 1, 0],
                     [0, 0, 1, 0, 1, 0],
                     [0, 1, 1, 1, 0, 0],
                     [1, 1, 0, 0, 0, 0]])

# network = nx.read_gml('karate.gml', label = 'id')
# adjacencyMatrix = nx.adjacency_matrix(network, nodelist = None, weight = 'weight').todense()

# for i in range(10):
#     start = time.time()
#     # communityVector = CLAnet(adjacencyMatrix, .001, 5, 10000)
#     # print(findModularity(communityVector, adjacencyMatrix), "\t", time.time() - start, "seconds")
#     bestModularity = CLAnet(adjacencyMatrix, .0005, 5, 10000)
#     print(bestModularity, "\t", time.time() - start, "seconds")

# network = nx.read_gml('dolphins.gml', label = 'id')
# adjacencyMatrix = nx.adjacency_matrix(network, nodelist = None, weight = 'weight').todense()

# for i in range(10):
#     start = time.time()
#     # communityVector = CLAnet(adjacencyMatrix, .001, 5, 10000)
#     # print(findModularity(communityVector, adjacencyMatrix), "\t", time.time() - start, "seconds")
#     bestModularity = CLAnet(adjacencyMatrix, .0005, 5, 10000)
#     print(bestModularity, "\t", time.time() - start, "seconds")

# network = nx.read_gml('football.gml', label = 'id')
# adjacencyMatrix = nx.adjacency_matrix(network, nodelist = None, weight = 'weight').todense()

# for i in range(10):
#     start = time.time()
#     # communityVector = CLAnet(adjacencyMatrix, .001, 5, 10000)
#     # print(findModularity(communityVector, adjacencyMatrix), "\t", time.time() - start, "seconds")
#     bestModularity = CLAnet(adjacencyMatrix, .0005, 5, 10000)
#     print(bestModularity, "\t", time.time() - start, "seconds")

## network = nx.read_gml('netscience.gml', label = 'id')
## adjacencyMatrix = nx.adjacency_matrix(network, nodelist = None, weight = 'weight').todense()

## for i in range(10):
##     start = time.time()
##     # communityVector = CLAnet(adjacencyMatrix, .001, 5, 10000)
##     # print(findModularity(communityVector, adjacencyMatrix), "\t", time.time() - start, "seconds")
##     bestModularity = CLAnet(adjacencyMatrix, .0005, 5, 10000)
##     print(bestModularity, "\t", time.time() - start, "seconds")

network = nx.read_gml('power.gml', label = 'id')
adjacencyMatrix = nx.adjacency_matrix(network, nodelist = None, weight = 'weight').todense()

for i in range(10):
    start = time.time()
    # communityVector = CLAnet(adjacencyMatrix, .001, 5, 10000)
    # print(findModularity(communityVector, adjacencyMatrix), "\t", time.time() - start, "seconds")
    bestModularity = CLAnet(adjacencyMatrix, .0005, 5, 10000)
    print(bestModularity, "\t", time.time() - start, "seconds")
