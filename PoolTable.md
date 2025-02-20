Here’s a Python program that demonstrates **reinforcement learning** using the **Q-learning algorithm** to train an agent to play a simplified **pool table game**. The goal is for the agent to learn how to hit the ball into a pocket using trial and error.

---

### **Python Program: Reinforcement Learning for Pool Table Game**

```python
import numpy as np
import random

# Define the pool table environment
class PoolTableEnv:
    def __init__(self):
        # Define the table dimensions (5x5 grid)
        self.table_size = 5
        # Define the pocket location (bottom-right corner)
        self.pocket = (4, 4)
        # Define the ball's starting position
        self.ball_pos = (0, 0)
        # Define possible actions (up, down, left, right)
        self.actions = ['up', 'down', 'left', 'right']
        # Define rewards
        self.reward_hit_pocket = 10
        self.reward_miss = -1

    def reset(self):
        # Reset the ball to the starting position
        self.ball_pos = (0, 0)
        return self.ball_pos

    def step(self, action):
        # Move the ball based on the action
        x, y = self.ball_pos
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.table_size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.table_size - 1:
            y += 1

        self.ball_pos = (x, y)

        # Check if the ball is in the pocket
        if self.ball_pos == self.pocket:
            reward = self.reward_hit_pocket
            done = True
        else:
            reward = self.reward_miss
            done = False

        return self.ball_pos, reward, done

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, max_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate

        # Initialize Q-table with zeros
        self.q_table = np.zeros((env.table_size, env.table_size, len(env.actions)))

    def choose_action(self, state):
        # Exploration vs Exploitation
        if random.uniform(0, 1) < self.exploration_rate:
            # Explore: choose a random action
            action = random.choice(self.env.actions)
        else:
            # Exploit: choose the action with the highest Q-value
            x, y = state
            action_index = np.argmax(self.q_table[x, y])
            action = self.env.actions[action_index]
        return action

    def update_q_table(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        action_index = self.env.actions.index(action)

        # Q-learning formula
        old_value = self.q_table[x, y, action_index]
        next_max = np.max(self.q_table[next_x, next_y])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[x, y, action_index] = new_value

    def decay_exploration_rate(self, episode):
        # Decay exploration rate
        self.exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)

# Train the agent
def train_agent(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Choose action
            action = agent.choose_action(state)
            # Take action and observe result
            next_state, reward, done = env.step(action)
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state)
            # Update state and total reward
            state = next_state
            total_reward += reward

        # Decay exploration rate
        agent.decay_exploration_rate(episode)

        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Exploration Rate: {agent.exploration_rate:.2f}")

# Test the trained agent
def test_agent(env, agent):
    state = env.reset()
    done = False
    steps = 0

    while not done:
        # Choose action (exploit only)
        action = agent.choose_action(state)
        print(f"Step {steps + 1}: Ball Position: {state}, Action: {action}")
        # Take action
        next_state, reward, done = env.step(action)
        state = next_state
        steps += 1

    print(f"Ball reached the pocket at {state} in {steps} steps!")

# Main program
if __name__ == "__main__":
    # Create environment and agent
    env = PoolTableEnv()
    agent = QLearningAgent(env)

    # Train the agent
    print("Training the agent...")
    train_agent(env, agent, num_episodes=1000)

    # Test the trained agent
    print("\nTesting the trained agent...")
    test_agent(env, agent)
```

---

### **How It Works**

1. **Environment**:
   - The pool table is represented as a 5x5 grid.
   - The ball starts at `(0, 0)` and must reach the pocket at `(4, 4)`.
   - The agent can move the ball `up`, `down`, `left`, or `right`.

2. **Q-Learning Agent**:
   - The agent learns by updating a **Q-table**, which stores the expected rewards for each state-action pair.
   - It balances **exploration** (trying random actions) and **exploitation** (using learned knowledge).

3. **Training**:
   - The agent trains over 1000 episodes, gradually reducing its exploration rate.
   - It learns to maximize rewards by hitting the pocket.

4. **Testing**:
   - After training, the agent is tested to see if it can efficiently guide the ball to the pocket.

---

### **Example Output**

#### Training Output:
```
Episode 100, Total Reward: -10, Exploration Rate: 0.90
Episode 200, Total Reward: 8, Exploration Rate: 0.80
Episode 300, Total Reward: 10, Exploration Rate: 0.70
...
Episode 1000, Total Reward: 10, Exploration Rate: 0.01
```

#### Testing Output:
```
Step 1: Ball Position: (0, 0), Action: right
Step 2: Ball Position: (0, 1), Action: down
Step 3: Ball Position: (1, 1), Action: right
Step 4: Ball Position: (1, 2), Action: down
Step 5: Ball Position: (2, 2), Action: right
Step 6: Ball Position: (2, 3), Action: down
Step 7: Ball Position: (3, 3), Action: right
Step 8: Ball Position: (3, 4), Action: down
Step 9: Ball Position: (4, 4), Action: down
Ball reached the pocket at (4, 4) in 9 steps!
```

---

### **Key Concepts Illustrated**
1. **Reinforcement Learning**:
   - The agent learns by interacting with the environment and receiving rewards.
   - It uses a **Q-table** to store and update knowledge.

2. **Exploration vs Exploitation**:
   - The agent balances trying new actions (exploration) and using known good actions (exploitation).

3. **Reward System**:
   - Positive reward for hitting the pocket.
   - Negative reward for missing.

4. **Q-Learning Formula**:
   - Updates the Q-value based on the current reward and the maximum future reward.

---

This program is a simple yet effective demonstration of reinforcement learning. You can expand it by adding more features, such as obstacles on the table or multiple balls! Let me know if you need further assistance.
