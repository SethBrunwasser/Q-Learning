import sys
import numpy as np
import math
import random

import gym
import gym_maze

'''
Using OpenAI's 'gym' to practice Q Learning in a maze environment
'''

def simulate():

	learning_rate = get_learning_rate(0)
	explore_rate = get_explore_rate(0)
	discount_factor = 0.99

	num_streaks = 0

	env.render()

	for episode in range(NUM_EPISODES):

		# Resetting the environment
		obv = env.reset()

		# Initial State
		state_0 = state_to_bucket(obv)
		total_reward = 0

		for t in range(MAX_T):

			# Selecting an action
			action = select_action(state_0, explore_rate)

			# Executing the action
			obv, reward, done, _ = env.step(action)

			# Observing the result
			state = state_to_bucket(obv)
			total_reward += reward

			# Updating the Q based on the result
			best_q = np.amax(q_table[state])
			q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * best_q - q_table[state_0 + (action,)])

			# Setting up for the next iteration
			state_0 = state

			if RENDER_MAZE:
				env.render()

			if env.is_game_over():
				sys.exit()

			if done:
				print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

				if t <= SOLVED_T:
					num_streaks += 1
				else:
					num_streaks = 0
				break
			elif t >= MAX_T - 1:
				print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

        # Done when it's solved over 120 times consecutively
		if num_streaks > STREAK_TO_END:
			break

        # Updating parameters
		explore_rate = get_explore_rate(episode)
		learning_rate = get_learning_rate(episode)

def select_action(state, explore_rate):
	# Selecting a random action
	if random.random() < explore_rate:
		action = env.action_space.sample()
	# Selection the action with the highest q
	else:
		action = int(np.argmax(q_table[state]))
	return action

def get_explore_rate(t):
	return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def get_learning_rate(t):
	return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))

def state_to_bucket(state):
	bucket_indice = []
	for i in range(len(state)):
		if state[i] <= STATE_BOUNDS[i][0]:
			buck_index = 0
		elif state[i] > STATE_BOUNDS[i][1]:
			bucket_index = NUM_BUCKETS[i] - 1
		else:
			bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
			offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
			scaling = (NUM_BUCKETS[i]-1)/bound_width
			buck_index = int(round(scaling * state[i] - offset))
		bucket_indice.append(bucket_index)
	return tuple(bucket_indice)

if __name__ == "__main__":
	from gym import envs
	envids = [spec.id for spec in envs.registry.all()]
	for envid in sorted(envids):
		print(envid)
	# Maze environment
	env = gym.make("Maze5-v0")

	MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
	NUM_BUCKETS = MAZE_SIZE

	NUM_ACTIONS = env.action_space.n # ['N', 'S', 'E', 'W']
	STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

	MIN_EXPLORE_RATE = 0.001
	MIN_LEARNING_RATE = 0.2

	DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0

	NUM_EPISODES = 50000
	MAX_T = np.prod(MAZE_SIZE, dtype=int) * 100
	STREAK_TO_END = 100
	SOLVED_T = np.prod(MAZE_SIZE, dtype=int)
	DEBUG_MODE = 0
	RENDER_MAZE = True
	ENABLE_RECORDING = True

	q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)

	recording_folder = "/tmp/maze_q_learning"

	if ENABLE_RECORDING:
		env.monitor.start(recording_folder, force=True)

	simulate()

	if ENABLE_RECORDING:
		env.monitor.close()
