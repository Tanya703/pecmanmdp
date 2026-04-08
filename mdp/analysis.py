# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    """
    BridgeGrid: change one parameter so the agent crosses the bridge.
    Setting noise=0.0 makes movement deterministic, eliminating the risk
    of falling off the narrow bridge, so the agent pursues the high reward.
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    return answerDiscount, answerNoise

def question3a():
    """
    DiscountGrid: prefer the close exit (+1) while risking the cliff (-10).
    Low discount makes the agent prefer nearby rewards over distant ones.
    Zero noise means the agent moves deterministically, so walking near
    the cliff is safe and the shorter risky path is taken.
    """
    answerDiscount = 0.1
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    DiscountGrid: prefer the close exit (+1) while avoiding the cliff (-10).
    Discount < 0.316 ensures the close exit beats the distant exit even via
    the longer safe upper path. Noise=0.2 makes the cliff edge risky enough
    that the agent prefers the upper route away from the cliff.
    """
    answerDiscount = 0.3
    answerNoise = 0.2
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    DiscountGrid: prefer the distant exit (+10) while risking the cliff (-10).
    High discount makes the agent patient enough to pursue the larger reward.
    Zero noise means movement is deterministic, so the shorter risky path
    near the cliff is taken without fear of accidentally falling.
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    DiscountGrid: prefer the distant exit (+10) while avoiding the cliff (-10).
    High discount keeps the agent patient enough to pursue the large reward.
    High noise makes accidental cliff falls very likely, so the agent takes
    the longer but safe upper path to reach the distant exit.
    """
    answerDiscount = 0.9
    answerNoise = 0.5
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    DiscountGrid: avoid both exits and the cliff so the episode never terminates.
    A large positive living reward means the agent earns more by staying alive
    each step than it would gain from any terminal state, so it loops indefinitely
    without exiting or falling into the cliff.
    """
    answerDiscount = 0.9
    answerNoise = 0.0
    answerLivingReward = 10.0
    return answerDiscount, answerNoise, answerLivingReward

def question8():
    """
    Q-learning on the bridge: determine if any epsilon and learning rate
    allow the agent to learn to cross the bridge within the episode limit.
    Returns 'NOT POSSIBLE' if no parameter setting achieves this.
    """
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
