"""
Modified version based on the code by mengxiaomao for the paper
https://arxiv.org/abs/1901.07159
"""
import scipy
from scipy import special
import numpy as np
from scipy.special import lambertw
import math
dtype = np.float32
class Env_cellular():
    def __init__(self, MAX_EP_STEPS, s_dim, location_vector, location_GF, Emax, K, T, eta, Pn, Pmax, w_csk, fading_n, fading_0):
        self.emax = Emax  # battery capacity
        self.K = K
        self.T = T
        self.eta = eta
        self.Pn = Pn  # grant-based user's power
        self.Pmax = Pmax
        self.w_csk = w_csk
        BW = 10 ** 6  # 10MHz
        sigma2_dbm = -170 + 10 * np.log10(BW)  # Thermal noise in dBm
        self.noise = 10 ** ((sigma2_dbm - 30) / 10)
        self.s_dim = s_dim
        self.MAX_EP_STEPS = MAX_EP_STEPS
        distance_GF = np.sqrt(np.sum((location_vector - location_GF) ** 2, axis=1))
        distance_GB = np.sqrt(np.sum((location_vector) ** 2, axis=1))
        distance = np.matrix.transpose(np.array([distance_GF, distance_GB]))
        distance = np.maximum(distance, np.ones((self.K, 2)))
        PL_alpha = 3
        PL = 1 / distance ** PL_alpha / (10 ** 3.17)
        self.hn = np.multiply(PL, fading_n)
        distance_GF0 = np.sqrt(np.sum(location_GF ** 2, axis=1))
        distance0 = np.maximum(distance_GF0, 1)
        PL0 = 1 / (distance0 ** PL_alpha) / (10 ** 3.17)
        self.h0 = fading_0 * PL0
        self.channel_sequence = np.zeros((self.MAX_EP_STEPS, 2))
        for i in range(self.MAX_EP_STEPS):
            id_index = i % self.K
            self.channel_sequence[i, :] = self.hn[id_index, :]

    def step(self, action, state, j):
        hn = state[0, 0] / self.noise
        hn0 = state[0, 1]
        h0 = self.h0 / self.noise  # state[0,2]/self.noise
        En = state[0, -1]
        En_bar = action * min(self.emax - En, self.T * self.eta * self.Pn * hn0) - (1 - action) * min(En, self.T * self.Pmax + self.w_csk)
        mu1 = self.eta * self.Pn * hn0 * h0 / (1 + self.Pn * hn)
        mu2 = En_bar * h0 / self.T / (1 + self.Pn * hn)
        mu3 = self.w_csk * h0 / (1 + self.Pn * hn)
        wx0 = np.real(lambertw(math.exp(-1) * (mu1 - 1), k=0))
        alphaxx = (mu1 - mu2) / (math.exp(wx0 + 1) - 1 + mu1 + mu3)
        alpha01 = 1 - (En + En_bar) / (self.T * self.eta * self.Pn * hn0)
        alpha02 = (self.T * self.eta * self.Pn * hn0 - En_bar) \
                  / (self.T * self.eta * self.Pn * hn0 + self.T * self.Pmax + self.T * self.w_csk)
        alphax2 = max(alpha01, alpha02)
        alphan = min(1, max(alphaxx, alphax2))
        reward = [0]
        if En_bar == self.T * self.eta * self.Pn * hn0:
            P0n=0
            reward = [0]  # remark in the paper
        elif alphan == 0: #<= 0.00000001:
            P0n=0
            reward = [0]
        else:
            P0n = (1 - alphan) * self.eta * self.Pn * hn0 / alphan - En_bar / alphan / self.T - self.w_csk
            reward = alphan * np.log(1 + P0n * h0 / (1 + self.Pn * hn))

        if math.isnan(reward[0]):
            reward = [0]

        batter_new = min(self.emax, En + En_bar)
        state_next = self.channel_sequence[(j+1) % self.K, :].tolist()
        state_next.append(float(batter_new))
        state_next = np.reshape(state_next, (1, self.s_dim))
        EHD = (1 - alphan) * self.eta * self.T * self.Pn * hn0
        done=False
        return reward, state_next, EHD, done, alphan

    def step_greedy(self,   state, j):
        hn = state[0,0]/self.noise
        hn0 = state[0,1]
        h0 = self.h0/self.noise
        En = state[0,-1]
        alphan = min(1, En/self.T/self.Pmax)
        if alphan==0:
            reward = 0
        else:
            reward = alphan*np.log(1 + self.Pmax*h0/(1 + self.Pn*hn))

        if math.isnan(reward):
            print(f"alpha is {alphan}  ")
            print(f"{En}")
        batter_new = min(self.emax, En - alphan*self.T*self.Pmax + (1-alphan) * self.T*self.eta*self.Pn*hn0)
        EHG = (1-alphan)*self.T*self.eta*self.Pn*hn0
        state_next = self.channel_sequence[(j+1) % self.K, :].tolist()
        state_next.append(batter_new)
        state_next = np.reshape(state_next, (1, self.s_dim))
        done = False
        return reward, state_next, EHG, done, alphan

    def step_random(self,   state, j):
        hn = state[0,0]/self.noise
        hn0 = state[0,1]
        h0 = self.h0/self.noise
        En = state[0,-1]
        alphan = np.random.uniform(0, min(1, En/self.T/self.Pmax))
        if alphan==0:
            reward = 0
        else:
            reward = alphan*np.log(1 + self.Pmax*h0/(1 +self.Pn*hn))

        if math.isnan(reward):
            print(f"alpha is {alphan}  ")
            print(f"{En}")
        batter_new = min(self.emax, En -alphan*self.T*self.Pmax +(1-alphan)*self.T*self.eta*self.Pn*hn0)
        EHR = (1-alphan)*self.T*self.eta*self.Pn*hn0
        state_next = self.channel_sequence[(j+1) % self.K, :].tolist()
        #state_next.append(self.h0)
        state_next.append(batter_new)
        state_next = np.reshape(state_next, (1, self.s_dim))
        done=False
        return reward, state_next, EHR, done, alphan

    def reset(self):
        batter_ini = self.emax
        return batter_ini
print("Successfully Compiled")