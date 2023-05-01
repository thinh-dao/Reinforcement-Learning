In model-free control, it is difficult or even impossible to estimate the transition dynamics of the environment. Therefore, we need to modify our current algorithms (Policy & Value iteration) so that they do not make use of the transition dynamics.

## 1. Policy Evaluation
If we are given the dynamics of the environment, then the standard algorithm to evaluate how good our current policy is to directly estimate the value for each states given the policy until convergence by the bellman equation: 

<img src="Expectation_Bellman.png" alt="bellman" width="75%" height="75%">

Without the transition model, one approach we can think of is estimating the values of states via simulations.

### 1.1 Monte Carlo Policy Evaluation

<img src="Monte_carlo.png" alt="value_monte" width="75%" height="75%">

We can also modify above algorithm so that we can estimate the values of state-action pairs.

<img src="Q_monte.png" alt="q_monte" width="75%" height="75%">

We can write the formula to update $V^{\pi}(s)$ in an incremental form:

<img src="incremental_monte.png" alt="incremental_monte" width="75%" height="75%">

Replacing $\frac {1}{N(s)}$ with $\alpha$ as the learning rate, we get the formula:

<img src="learning_rate.png" alt="learning_rate" width="75%" height="75%">

There are some key limitations with Monte Carlo Policy Evaluation:

<ul>
    <li> High variance: Estimates depend on a random sample of episodes, and the sample may not be representative of the true distribution of episodes. </li>
    <li> Computtional complexity: Cannot work well with large state spaces or when real-time processing is required </li>
    <li> No guarantee of convergence </li>
</ul>


### 1.2 Temporal Difference Policy Evaluation
TDLearning is a combination of Monte Carlo & dynamic programming methods. It is model-free and can immediately update estimates of V after each (s, a, r, s') tuple.
From above, we have the update formula:

<img src="incremental_monte.png" alt="incremental_monte" width="75%" height="75%">

To execute the update, we need to have the full roll-out of an episode to estimate the return value $G_{i,t}$. However, instead of executing the full episode, we can estimate the value of $G_{i,t}$ by the bellman backup operator:

<img src="TDLearning_update.png" alt="TDLearning_update" width="75%" height="75%">

<img src="Diagram.png" alt="TDLearning_diagram" width="75%" height="75%">












