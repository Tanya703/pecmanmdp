# Reinforcement Learning: MDPs & Value Iteration

Project implementing value iteration algorithms for Markov Decision Processes (MDPs), tested on Gridworld and Pacman environments.

---

## Overview

This project implements:
- **Value Iteration** (batch style) for solving known MDPs
- **Asynchronous Value Iteration** (cyclic, one state per iteration)
- **Prioritized Sweeping Value Iteration** (error-driven state updates)
- **Q-Learning agents** for Gridworld, Crawler, and Pacman
- **MDP policy analysis** under varying discount, noise, and living reward parameters

---

## Project Structure

### Files to Edit
| File | Description |
|------|-------------|
| `valueIterationAgents.py` | Value iteration agent for solving known MDPs |
| `qlearningAgents.py` | Q-learning agents for Gridworld, Crawler, and Pacman |
| `analysis.py` | Answers to analysis questions (discount/noise/reward settings) |

### Files to Read (Do Not Edit)
| File | Description |
|------|-------------|
| `mdp.py` | Defines methods on general MDPs |
| `learningAgents.py` | Base classes `ValueEstimationAgent` and `QLearningAgent` |
| `util.py` | Utilities including `util.Counter` and `util.PriorityQueue` |
| `gridworld.py` | The Gridworld MDP implementation |
| `featureExtractors.py` | Feature extractors for (state, action) pairs |

### Files You Can Ignore
`environment.py`, `graphicsGridworldDisplay.py`, `graphicsUtils.py`, `textGridworldDisplay.py`, `crawler.py`, `graphicsCrawlerDisplay.py`, `autograder.py`, `testParser.py`, `testClasses.py`, `test_cases/`, `reinforcementTestClasses.py`

---

## Getting Started

Run Gridworld in manual control mode (arrow keys):
```bash
python gridworld.py -m
```

View all options:
```bash
python gridworld.py -h
```

---

## Questions & Commands

### Q1: Value Iteration
Implement batch value iteration in `ValueIterationAgent`.

```bash
python autograder.py -q q1
python gridworld.py -a value -i 100 -k 10   # run policy 10 times
python gridworld.py -a value -i 5            # check k=5 output
```

### Q2: Bridge Crossing Analysis
Modify one parameter (discount or noise) so the agent crosses the bridge in `BridgeGrid`.

```bash
python autograder.py -q q2
python gridworld.py -a value -i 100 -g BridgeGrid --discount 0.9 --noise 0.2
```

### Q3: Policies (DiscountGrid)
Tune discount, noise, and living reward to produce 5 distinct policy behaviors. Answer in `analysis.py` as `(discount, noise, living_reward)` tuples.

```bash
python autograder.py -q q3
python gridworld.py -a value -g DiscountGrid
```

Behaviors to produce:
- (a) Close exit (+1), risking cliff (-10)
- (b) Close exit (+1), avoiding cliff (-10)
- (c) Distant exit (+10), risking cliff (-10)
- (d) Distant exit (+10), avoiding cliff (-10)
- (e) Avoid both exits and cliff (never terminate)

### Q4: Asynchronous Value Iteration
Implement cyclic value iteration in `AsynchronousValueIterationAgent` (one state updated per iteration).

```bash
python autograder.py -q q4
python gridworld.py -a asynchvalue -i 1000 -k 10
```

### Q5: Prioritized Sweeping Value Iteration
Implement `PrioritizedSweepingValueIterationAgent` using `util.PriorityQueue`. Updates states with the highest Bellman error first.

```bash
python autograder.py -q q5
python gridworld.py -a priosweepvalue -i 1000
```

### Run All Tests
```bash
python autograder.py
```

Run a specific test case:
```bash
python autograder.py -t test_cases/q2/1-bridge-grid
```

---

