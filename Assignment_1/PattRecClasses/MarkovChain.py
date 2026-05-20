import numpy as np
from .DiscreteD import DiscreteD
from .GaussD import GaussD
from .HMM import HMM
import matplotlib.pyplot as plt

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[0]

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]:
            self.is_finite = True


    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        #*** Insert your own code here and remove the following error message 
        S = []

        s = np.random.choice(self.nStates, p=self.q)

        for i in range(1, tmax):
            s_next = np.random.choice(self.A.shape[1], p=self.A[s])

            if self.is_finite and s_next == self.nStates:
                break
            S.append(s_next)
            s = s_next
        return np.array(S)

    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, pX):
        a_temp = np.zeros((self.nStates, pX.shape[1]))
        c = np.zeros(pX.shape[1]+1)
        a_hat = np.zeros((self.nStates, pX.shape[1]))

        a_temp[:, 0] = self.q * pX[:, 0]
        c[0] = np.sum(a_temp[:, 0])
        a_hat[:, 0] = a_temp[:, 0] / c[0]
        for t in range(1, pX.shape[1]):
            a_temp[:, t] = (a_hat[:, t-1] @ self.A[:, :self.nStates]) * pX[:, t]
            c[t] = np.sum(a_temp[:, t])
            a_hat[:, t] = a_temp[:, t] / c[t]
        c[-1] = np.sum(a_hat[:, -1] @ self.A[:, self.nStates]) if self.is_finite else 0
        return a_hat, c

    def finiteDuration(self):
        pass
    
    def backward(self, c, pX):
        b = np.zeros((self.nStates, pX.shape[1]))
        b_hat = np.zeros((self.nStates, pX.shape[1]))
        if self.is_finite:
            b[:, -1] = self.A[:, self.nStates]
            b_hat[:, -1] = b[:, -1] / (c[-1] * c[-2])
        else:
            b[:, -1] = 1
            b_hat[:, -1] = b[:, -1] / c[-2]
        for t in range(pX.shape[1]-1, 0, -1):
            b_hat[:, t-1] = (self.A[:, :self.nStates] @ (pX[:, t] * b_hat[:, t])) / c[t-1]
        return b_hat

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
