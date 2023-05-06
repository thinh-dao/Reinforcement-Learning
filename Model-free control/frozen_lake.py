# frozen-lake-ex1.py
import gymnasium as gym # loading the Gym library
import numpy as np
import time
import copy   

class Frozen_lake:
    def __init__(self, gamma=0.9, seed=0, config=None, stochastic=True):
        self.env = gym.make('FrozenLake-v1', render_mode='ansi', desc=config, is_slippery=stochastic).unwrapped
        np.random.seed(seed)   
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)
        self.P = self.env.P
        self.state_space = tuple(range(self.env.observation_space.n))
        self.action_space = tuple(range(self.env.action_space.n))
        self.gamma = gamma
        self.epsilon = epsilon
        #self.policy = np.array([self.env.action_space.sample() for i in range(len(self.state_space))])
        self.policy = np.zeros(len(self.state_space), dtype=int)
        self.Q = np.zeros((len(self.state_space), len(self.action_space)))
        
    def one_step_lookahead(self, s, V):
        Q = np.zeros(len(self.action_space))
        for a in self.action_space:
            for prob, next_state, reward, done in self.P[s][a]:
                Q[a] += prob * (reward + self.gamma * V[next_state])
                
        best_action = np.argmax(Q)
        return np.max(Q), best_action

    def policy_evaluation(self, threshold = 0.001):
        V = np.zeros(len(self.state_space))
        
        while True:
            delta = 0
            for s in self.state_space:
                v = 0
                a = self.policy[s]
                for prob, next_state, reward, done in self.P[s][a]:
                    v += prob * (reward + self.gamma * V[next_state])
                delta = max(delta, np.abs(V[s] - v))
                V[s] = v
                
            if delta < threshold:
                break
            
        return V

    def policy_improvement(self, values):
        new_policy = copy.deepcopy(self.policy)
        policy_stable = False
        for s in self.state_space:
            best_Q, best_action = self.one_step_lookahead(s, values)
            new_policy[s] = best_action
        if np.array_equal(self.policy, new_policy): policy_stable = True
        self.policy = new_policy
        return policy_stable
        
    def policy_iteration(self): 
        start = time.time()
        cnt = 0
        while True:
            cnt += 1
            V = self.policy_evaluation()
            policy_stable = self.policy_improvement(V)
            if policy_stable:
                break
        
        end = time.time()
        print(f"Policy converges after {cnt} iterations")
        print(f"Runtime: {end-start}")
        self.render_single()
        
    
    def value_iteration(self, threshold = 0.001):
        start = time.time()
        V = np.zeros(len(self.state_space))
        cnt = 0
        
        while True:
            cnt += 1
            delta = 0
            for s in self.state_space:
                best_Q, best_action = self.one_step_lookahead(s, V)
                temp = V[s]
                V[s] = best_Q
                delta = max(delta, abs(temp - V[s]))
            if delta < threshold:
                break 
        
        print(f"Values converge after {cnt} iterations")
        print(V)
        self.policy_improvement(V)
        end = time.time()
        print(f"Runtime: {end-start}")
        self.render_single()
        
    def epsilon_greedy(self, epsilon):
        for s in self.state_space:
            if np.random.random() >= epsilon:
                self.policy[s] = np.argmax(self.Q[s])
            else:
                self.policy[s] = np.random.randint(len(self.action_space))
                
    def sarsa(self, alpha, epsilon, n_episodes):
        start = time.time()
        self.epsilon_greedy(epsilon)
        for episode in range(n_episodes):
            state,_ = self.env.reset()
            action = self.policy[state]
            done = False
            while done == False:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action = self.policy[next_state]
                self.Q[state][action] = self.Q[state][action] + alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
                self.epsilon_greedy(epsilon)
                action = next_action
                state = next_state
                
        end = time.time()
        print(f"Runtime: {end-start}")
        self.render_single()

    def render_single(self, max_steps=100):
        for a in self.policy:
            if a == 0:
                print('left', end=',')
            elif a == 1:
                print('down', end=',')
            elif a == 2:
                print('right', end=',')
            elif a == 3:
                print('up', end=',')
        print()
        episode_reward = 0
        ob, _ = self.env.reset()
        self.env.render_mode = 'human'
        for t in range(max_steps):
            self.env.render()
            time.sleep(0.25)
            a = self.policy[ob]
            ob, rew, done, _, _ = self.env.step(a)
            episode_reward += rew
            if done:
                break
        self.env.render()
        if not done:
            print(
                "The agent didn't reach a terminal state in {} steps.".format(
                    max_steps
                )
            )
        else:
            print("Episode reward: %f" % episode_reward)
            

if __name__ == "__main__":
    config = ['SFFFF','HFFFF','FFFFF','FFFFF', 'FFFFG']
    gamma = 1.0
    alpha = 0.25 
    epsilon = 0.29
    n_episodes = 14697
    seed = 741684
    game = Frozen_lake(gamma=gamma,seed=seed,config=config)
    game.sarsa(alpha, epsilon, n_episodes)