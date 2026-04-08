# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
        Run batch value iteration for self.iterations steps.
        Each iteration computes a completely new value table from the previous
        one (Vk from Vk-1), so all states are updated simultaneously using
        only values from the prior iteration. Terminal states are assigned
        value 0; all other states take the maximum Q-value across actions.
        """
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            temp_values = util.Counter()
            for state in states:
                if self.mdp.isTerminal(state):
                    temp_values[state] = 0
                else:
                    actions = self.mdp.getPossibleActions(state)
                    max_q_value = float('-inf')
                    for action in actions:
                        q_value = self.computeQValueFromValues(state, action)
                        max_q_value = max(max_q_value, q_value)
                    temp_values[state] = max_q_value
            self.values = temp_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
        Return the Q-value of (state, action) using the current value table.
        Applies the Bellman equation:
            Q(s, a) = sum over s' of P(s'|s,a) * [R(s,a,s') + discount * V(s')]
        where V(s') is looked up from self.values.
        """
        q_value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        """
        Return the best action from state according to the current value table.
        Computes the Q-value for each available action and returns the action
        with the highest Q-value (the greedy policy with respect to self.values).
        Returns None if the state is terminal and has no legal actions.
        """
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        best_action = None
        best_value = float('-inf')
        for action in actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Run cyclic (asynchronous) value iteration for self.iterations steps.
        Each iteration updates exactly one state in-place, cycling through the
        state list in order. Unlike batch value iteration, the updated value is
        immediately available for subsequent updates in the same pass, allowing
        information to propagate faster. Terminal states are skipped.
        """
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            state = states[iteration % len(states)]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                max_q_value = float('-inf')
                for action in actions:
                    q_value = self.computeQValueFromValues(state, action)
                    max_q_value = max(max_q_value, q_value)
                self.values[state] = max_q_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Run prioritized sweeping value iteration for self.iterations steps.
        States with the largest Bellman error (difference between current value
        and the best Q-value) are updated first using a min-heap priority queue.
        After each update, predecessors of the updated state are re-evaluated
        and re-added to the queue if their error exceeds theta, propagating
        value changes efficiently backward through the state graph.
        """
        priority_queue = util.PriorityQueue()
        states = self.mdp.getStates()
        predecessors = collections.defaultdict(set)
        
        # Build predecessor graph
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob != 0:
                            predecessors[next_state].add(state)
        
        # Initialize priority queue with all non-terminal states
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                max_q_value = float('-inf')
                for action in actions:
                    q_value = self.computeQValueFromValues(state, action)
                    max_q_value = max(max_q_value, q_value)
                diff = abs(self.values[state] - max_q_value)
                priority_queue.push(state, -diff)
        
        # Process iterations
        for iteration in range(self.iterations):
            if priority_queue.isEmpty():
                break
            
            state = priority_queue.pop()
            
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                max_q_value = float('-inf')
                for action in actions:
                    q_value = self.computeQValueFromValues(state, action)
                    max_q_value = max(max_q_value, q_value)
                self.values[state] = max_q_value
            
            # Update predecessors
            for pred in predecessors[state]:
                if not self.mdp.isTerminal(pred):
                    actions = self.mdp.getPossibleActions(pred)
                    max_q_value = float('-inf')
                    for action in actions:
                        q_value = self.computeQValueFromValues(pred, action)
                        max_q_value = max(max_q_value, q_value)
                    diff = abs(self.values[pred] - max_q_value)
                    if diff > self.theta:
                        priority_queue.update(pred, -diff)

