import numpy as np
import gym

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=False)  # Set is_slippery=True for stochastic environment

# Initialize Q-table
state_size = env.observation_space.n
action_size = env.action_space.n
q_table = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.8        # Learning rate
gamma = 0.95       # Discount factor
epsilon = 1.0      # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 2000
max_steps = 100

# Training loop
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    for step in range(max_steps):
        # Choose action (epsilon-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])
        
        # Take action and observe result
        next_state, reward, done, _, _ = env.step(action)
        
        # Update Q-value
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])
        
        state = next_state
        
        if done:
            break
    
    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training finished.\n")

# Test the agent
state = env.reset()[0]
env.render()
done = False
while not done:
    action = np.argmax(q_table[state, :])
    state, reward, done, _, _ = env.step(action)
    env.render()
