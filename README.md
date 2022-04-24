# Machine-Learning_basic_HW5
Machine-Learning_basic_HW5

Q-Learning
# Goal
- Applying Q-learning & DQN algorithm
- Understand why we need to use neural  network to efficiently improve the performance of the agent
- Approach the world of reinforcement learning

Q-Learning

1. From the current state, we choose the maximum value 
of state-action pair value from the current Q table and 
apply the corresponding action, or randomly select one.
2. After applying the action, the agent will transit to next 
state and get the reward, then we can update the Q table 
by the following equation.

# Basic-part 
- Applying Q learning with tabular implementation.
- You just need to fill the missing segments in template
- Record the reward throughout the training process and plot the reward record in the report
- You can set any number as your episode length and episode
- Remember to discretize the state space so that you can build a table to finish the task
- We will evaluate your q table to make sure you can balance the pole on the car more than 200 timestam

# Advanced 
- Apply Deep Q-Network
- Using the same environment “CartPole-v0”
- You don’t need to build the network from scratch, you just need to fill the missing segments in template
- Do not modify the code that are not between the comments
- Record the reward throughout the training process and plot the reward record in the report 
- We will evaluate your model by checking whether the pole can keep balance over 200 timestamps

# Bonus
- Please use the deep q-network to train an agent in the environment “Breakout-v4”
- Please describe how you implementthe DQN to the environment
- Record the reward throughout the training process and plot the reward record in the report
- Please save the model, and we will evaluate the model based on the number of bricks that you can diminish
