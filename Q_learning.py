import numpy as np
from collections import defaultdict
import plotting
import itertools
import pandas as pd
import sys
from ObstacleAvoidanceENV import UAVEnv
from my_env import MyEnv
# import seaborn as sns

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation(state) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        # print(nA,epsilon,observation)
        best_action = np.argmax(Q[observation]) #argmax- index of the highest value in the array
        A[best_action] += (1.0 - epsilon) 
        return A

    return policy_fn


def q_learning(env, num_episodes=1000, discount_factor=1.0, alpha=0.5, epsilon=0.1, wind_direction=None, wind_strength=None):
# def q_learning(env, num_episodes=1000, discount_factor=0.6, alpha=0.5, epsilon=0.1,max_iterations = 500):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.

    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    origin = (0,0)
    des = (50,50)
    env = UAVEnv(origin,des)
    params = (env.nS,env.nA, env.P, env.isd, env.wind_direction, env.wind_strength)
    my_env = MyEnv(*params)
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n)) #defaultdict- when a key doesn't exist for a value then returns 0

    # Keeps track of useful statistics from plotting.py
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    episode_lengths = []

    margin = 1
    w = 10
    trajectories = []
    origin1 = env.origin
    # timesteps = []
    for i_episode in range(num_episodes):
        epsilon = 1 / (i_episode + 1)
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        # Print out which episode
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment and pick the first action
        state = env.reset() 

        # One step in the environment
        pp = [] #next_state
        aa = [] #action
        rr = [] #reward

        pp.append(int(np.ravel_multi_index(origin1, (50, 50)))) #append-add list to end of other list
        rr.append(-4 * np.sqrt(2))
        for t in itertools.count():
            # if t>=max_iterations:
            #     break
            # Take a step
            action_probs = policy(state) #?? line 25
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs) #random.choice-Generates a random sample from a given 1-D array
            aa.append(action)
            wind_direction = None
            wind_strength = None
            next_state, reward, done, prob, wind_direction,wind_strength = env.step(action, wind_direction, wind_strength)
            # print(action)
            rr.append(reward)
            pp.append(int(next_state))
            # timesteps.append(t)
            
            # next_position = np.unravel_index(next_state, (100, 100))
            # if next_position[0] < margin or next_position[0] > 100 - margin or next_position[1] < margin or next_position[1] > 100 - margin:
            #     Q[state][action] = Q[state][action] + alpha * (reward + discount_factor * Q[next_state][np.argmax(Q[next_state])] - Q[state][action])
            #     break
            # else:
            #     Q[state][action] = Q[state][action] + alpha * (reward + discount_factor * Q[next_state][np.argmax(Q[next_state])] - Q[state][action])
            #     state = next_state
            #     if done:
            #         # print(t,len(aa),len(rr),len(pp))
            #         stats.episode_lengths[i_episode] = t
            #         stats.episode_rewards[i_episode] = sum(rr)
            #         break
            
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            # stats.episode_rewards[i_episode] = sum(rr)
            stats.episode_lengths[i_episode] = t

            # Off policy Temporal Difference(TD) Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta  # Q function
            # timesteps = q_learning(env)
            # Q_df = pd.DataFrame(Q)
            # print(Q_df)
            # print(timesteps)
            
            # Plot the dataframe as a heatmap
            # sns.heatmap(Q_df, annot=True, fmt='.2f')
            
            if done:
                aa.append(int(0))
                # episode_lengths.append(t+1)
                break

            state = next_state
        trajectory = []
        trajectory.append([[pp[i], aa[i], rr[i]] for i in range(len(aa))])
        trajectory = np.array(trajectory)
        trajectory = trajectory.reshape(trajectory.shape[1], trajectory.shape[2])
        if i_episode > 2998:
            trajectories.append(trajectory)

    return Q, stats, np.array(trajectories)

"""
    
In this particular implementation, the function q_learning takes as input an environment env, along with several parameters such as the number of episodes to run for, the discount factor, the learning rate, and the probability of taking a random action (epsilon). The function returns a tuple containing the optimal action-value function (a dictionary mapping states to action values) and an EpisodeStats object with arrays for episode lengths and rewards.

The q_learning function first creates an epsilon-greedy policy using the make_epsilon_greedy_policy function, which returns a policy function that takes in an observation (state) and returns the probabilities of selecting each action. The q_learning function then iterates through a number of episodes, resetting the environment and selecting an action at each step. It updates the action-value function Q using the Q-learning update rule, which is based on the difference between the current and expected action values. The algorithm continues until the maximum number of episodes has been reached.

The EpisodeStats object is used to track useful statistics such as the lengths and rewards of each episode. The itertools.count function is used to create an infinite iterator that is used to iterate through the steps within an episode. The np.random.choice function is used to randomly select an action based on the action probabilities returned by the policy function. The np.unravel_index function is used to convert a linear index to a tuple of indices, and the np.ravel_multi_index function is used to convert a tuple of indices to a linear index.
    
"""