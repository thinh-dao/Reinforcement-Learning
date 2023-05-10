import gymnasium as gym
import numpy as np
import time

class Taxi_driver:
    def __init__(self):
        self.env = gym.make('Taxi-v3', render_mode='ansi')
        self.gamma = 0.9
        self.epsilon = 0.25
        # self.decay_rate = 0.01
        self.nA = 6
        self.nS = 500
        self.policy = np.zeros(shape=500)
        self.Q = np.zeros(shape=(500, 6))
        
    def Q_learning(self, alpha, n_episodes):
        samp_reward = 0
        for episode in range(n_episodes):
            print('Episode', episode+1)
            if ((episode + 1) % 100 == 0):
                avg_reward = samp_reward/100
                print(f'Episode {episode-99} to episode {episode+1}: {avg_reward}')
                samp_reward = 0
                if avg_reward >= 9.7: break
                
            state, info = self.env.reset()
            done = False
            while done == False:
                action = self.epsilon_greedy(state)
                next_state, reward, done, _, info = self.env.step(action)
                samp_reward += reward
                self.Q[state][action] = self.Q[state][action] + alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
                state = next_state

        self.policy_extraction()
        self.render_single()
        
    def policy_extraction(self):
        for s in range(len(self.Q)):
            self.policy[s] = np.argmax(self.Q[s])
        
    def epsilon_greedy(self, state):
        if np.random.random() >= self.epsilon:
            return np.argmax(self.Q[state])
        else:
            return self.env.action_space.sample()
        # self.epsilon -= self.epsilon * self.decay_rate
        
                
    def render_single(self, max_steps=100):
        episode_reward = 0
        ob, _ = self.env.reset()
        for t in range(max_steps):
            time.sleep(0.25)
            a = self.policy[ob]
            ob, rew, done, _, _ = self.env.step(a)
            episode_reward += rew
            if done:
                break
        # self.env.render()
        if not done:
            print(
                "The agent didn't reach a terminal state in {} steps.".format(
                    max_steps
                )
            )
        else:
            print("Episode reward: %f" % episode_reward)

game = Taxi_driver()
alpha = 0.25
n_episodes = 1000000
game.Q_learning(alpha, n_episodes)
print(game.Q[462][4])
print(game.Q[398][3])
print(game.Q[253][0])
print(game.Q[377][1])
print(game.Q[83][5])



