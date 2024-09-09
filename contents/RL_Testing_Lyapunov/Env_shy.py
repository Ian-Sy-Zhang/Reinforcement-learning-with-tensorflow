import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy import linalg as la, optimize
import control

class CustomEnv(gym.Env):
    def __init__(self, A, B, n, m, reward_type, test, testing_mode=False, random_seed=1):
        super(CustomEnv, self).__init__()
        np.random.seed(random_seed)
        self.n = n
        self.m = m
        self.Q = np.eye(n)
        self.R = np.eye(m)
        self.X_desired = np.zeros(n)

        self.stable_counter = 0
        self.stable_counter_threshold = 7
        self.step_counter = 0

        self.state_distance_boundary = 2
        self.out_of_boundary_punish = 100

        self.A = A
        self.B = B
        self.K, self.P = self.findPK(self.A, self.B, self.Q, self.R)
        self.Acl = self.A - self.B @ self.K
        self.action_inf = abs(self.K).sum(axis=1).max()

        self.reward_type = reward_type
        
        self.action_space = spaces.Box(low=-20, high=20, shape=(m,1), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(n,1), dtype=np.float32)

        self.reward_distance_threshold = 0.2

        #for test
        self.perturbation_vectors = self.random_perturbation_vector_generation(shape = (m, 1), num_samples=10, noise_scale=0.001) 
        self.test = test

        self.testing_mode = testing_mode

    def set_model(self, model):
        self.model = model

    def findPK(self, A, B, Q, R):
        K = control.dlqr(A, B, Q, R)[0]
        Acl = A - B @ K
        P = la.solve_discrete_lyapunov(Acl.T, Q)
        is_positive_definite = np.all(np.linalg.eigvals(P) > 0)
        if is_positive_definite:
            print("System can be stable")
            return K, P
        else:
            print("System can't be stable")
            return None, None
    
    def step(self, action):
        self.step_counter += 1
        X_star_k_plus_1 = (self.Acl @ self.state).astype(np.float32)
        X_k_plus_1 = (self.A @ self.state + self.B @ action).astype(np.float32)

        terminated = False
        truncated = False

        if self.reward_type == "Shy":
            reward = self.calculateReward(X_star_k_plus_1, X_k_plus_1)
        elif self.reward_type == "potential":
            reward = self.calculateRewardPotential(X_k_plus_1)
        elif self.reward_type == "steps":
            reward = self.calculateRewardPotentialMultiStep(X_k_plus_1)
        self.state = X_k_plus_1

        distance = np.linalg.norm(self.state - self.X_desired)

        if not self.testing_mode:
            if distance >= self.state_distance_boundary:
                terminated = True
                reward -= self.out_of_boundary_punish

        info = {}
        return self.state, float(reward), terminated, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.stable_counter = 0
        self.step_counter = 0
        self.state = np.random.uniform(-1, 1, size=(self.m, 1)).astype(np.float32)
        return self.state, {}
    
    def calculateReward(self, X_star_k_plus_1, X_k_plus_1):
        punish = 0
        for i in range(self.n):
            for j in range(i, self.n):
                if i == j:
                    diff = pow(X_k_plus_1[i], 2) - pow(X_star_k_plus_1[i], 2)
                    punish += abs(self.P[i, j] * diff)
                else:
                    diff = X_k_plus_1[i] * X_k_plus_1[j] - X_star_k_plus_1[i] * X_star_k_plus_1[j]
                    punish += abs(2 * self.P[i, j] * diff)
            reward = 1 - punish
        return reward if reward > 0 else 0
    
    def random_perturbation_vector_generation(self, shape, num_samples, noise_scale):
        perturbed_vectors = []
        for _ in range(num_samples):
            noise = np.random.normal(scale=noise_scale, size=shape)
            perturbed_vector = noise
            perturbed_vectors.append(perturbed_vector)
        return perturbed_vectors

    def perturbated_neighbor_force_comparison_model(self, model, vector, perturbation_vectors):
        total_drift = 0
        vector_drift = self.A @ vector + self.B @ (model.predict(vector, deterministic=True)[0])
        for perturbation_vector in perturbation_vectors:
            perturbated_neighbor = vector + perturbation_vector
            perturbated_neighbor_drift = self.A @ perturbated_neighbor + self.B @ (model.predict(perturbated_neighbor, deterministic=True)[0])
            diff_vector = vector_drift - perturbated_neighbor_drift
            total_drift += np.linalg.norm(diff_vector)
        return total_drift

    def calculateRewardPotential(self,  X_k_plus_1):
        potential = X_k_plus_1.T @ self.P @ X_k_plus_1
        potential_reward = np.exp(-1 * potential)

        # total_drift = self.perturbated_neighbor_force_comparison_model(model = self.model, vector=self.state, perturbation_vectors=self.perturbation_vectors)
        # continous_reward = np.exp(-1 * total_drift)

        return potential_reward

    def calculateRewardPotentialMultiStep(self, X_k_plus_1):
        distance = np.linalg.norm(self.state - self.X_desired)
        reward = 0

        X_k_potential = self.state.T @ self.P @ self.state
        potential_reward = np.exp(-1 * X_k_potential)
        # X_k_plus_1_potential = X_k_plus_1.T @ self.P @ X_k_plus_1
        # potential_diff = X_k_plus_1_potential - X_k_potential

        # if potential_diff[0][0] < 0:
        #     reward += 1
        # else:
        #     reward -= 1
        
        convergence_reward = np.exp(-1 * distance)
        reward = potential_reward + convergence_reward * 0.7

        # if distance <= self.reward_distance_threshold:
        #     # convergence_reward = np.exp(-1 * distance)
        #     # reward += convergence_reward
        #     self.stable_counter += 1
        # else:
        #     self.stable_counter = 0
        
        # if self.stable_counter >= self.stable_counter_threshold:
        #     reward = 15
        
        return reward
