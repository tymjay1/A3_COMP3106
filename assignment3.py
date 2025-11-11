import csv
import sys
import os
import copy

def parseData(directory_path):
  items = os.listdir(directory_path)

  trials = []
  for item in items:
    file_path = os.path.join(directory_path, item)
    trial = []
    with open(file_path, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        state = row[0]
        action = row[1]
        trial.append([state, action])
    trials.append(trial)

  return trials

def generateAllStates(c_bag, c_agent, c_opponent, turn, states):
  if turn == "A":
    for i in range(1, 4):
      new_c_agent = c_agent + i
      new_c_bag = c_bag - i
      if new_c_bag > 0:
        states.append(f"{new_c_bag}/{new_c_agent}/{c_opponent}/-")
        generateAllStates(new_c_bag, new_c_agent, c_opponent, "O", states)
      elif new_c_bag == 0:
        states.append(f"{new_c_bag}/{new_c_agent}/{c_opponent}/O")

  if turn == "O":
    for i in range(1, 4):
      new_c_opponent = c_opponent + i
      new_c_bag = c_bag - i
      if new_c_bag > 0:
        states.append(f"{new_c_bag}/{c_agent}/{new_c_opponent}/-")
        generateAllStates(new_c_bag, c_agent, new_c_opponent, "A", states)
      elif new_c_bag == 0:
        states.append(f"{new_c_bag}/{c_agent}/{new_c_opponent}/A")

def getAllStates():
  states = []

  c_bag = 13
  c_agent = 0
  c_opponent = 0

  states.append(f"{c_bag}/{c_agent}/{c_opponent}/-")
  # Agent start the game first
  generateAllStates(c_bag, c_agent, c_opponent, "A", states)
  # Opponent start the game first
  generateAllStates(c_bag, c_agent, c_opponent, "O", states)

  states = list(set(states))

  return states

def getAllActions(state):
  remain_c_bag = int(state.split("/")[0])

  if remain_c_bag > 2:
    return ['1', '2', '3']
  elif remain_c_bag > 1:
    return ['1', '2']
  elif remain_c_bag > 0:
    return ['1']
  return ["-"]

def getReward(state):
  winner = state.split("/")[3]
  c_agent = int(state.split("/")[1])

  if winner == "A":
    return c_agent
  elif winner == "O":
    return -c_agent
  return 0

def isConverge(first_q, second_q):
  if first_q is None or second_q is None:
    return False

  for state, actions in first_q.items():
      for action in actions:
        diff = abs(first_q[state][action] - second_q[state][action])

        if diff > 0.001:
          return False

  return True

class td_qlearning:

  alpha = 0.10
  gamma = 0.90
  q_function = dict()

  def __init__(self, directory):
    # directory is the path to a directory containing trials through state space
    trials = parseData(directory)
    
    for state in getAllStates():
      self.q_function[state] = dict()
      for action in getAllActions(state):
        self.q_function[state][action] = getReward(state)

    prev_q_function = None
    while not isConverge(self.q_function, prev_q_function):
      prev_q_function = copy.deepcopy(self.q_function)

      for itr in range(1000):
        for trial in trials:
          for i in range(len(trial)-1): # -1 because we don't calculate Q-value for terminate state-action pair
            cur_state = trial[i][0]
            cur_action = trial[i][1]
            next_state = trial[i+1][0]
            next_action = trial[i+1][1]

            error_term = getReward(cur_state) + self.gamma * max(self.q_function[next_state].values()) - self.q_function[cur_state][cur_action]
            self.q_function[cur_state][cur_action] += self.alpha * error_term
    # Return nothing

  def qvalue(self, state, action):
    # state is a string representation of a state
    # action is an integer representation of an action
    if action == 0:
      action = '-'
    q = self.q_function[state][str(action)]

    # Return the q-value for the state-action pair
    return q

  def policy(self, state):
    # state is a string representation of a state
    a = None
    max_q_value = max(self.q_function[state].values())

    for action in self.q_function[state].keys():
      if self.q_function[state][action] == max_q_value:
        if action == '-':
          a = 0
        else:
          a = int(action)

    # Return the optimal action (as an integer) under the learned policy
    return a


if __name__ == "__main__":
  # NOTE: Give path to csv/txt file as command line argument
  # Example: python3 assignment3.py Examples/Example0/Trials

  tdq = td_qlearning(sys.argv[1])
  print(tdq.policy("11/1/1/-"))
  print(tdq.qvalue("8/3/2/-", 2))