[//]: # (Image References)

[image1]: DDPG_trained.png "DDPG training"
[image3]: MADDPG_trained.png "MADDPG training"


# Project 3: Collaboration and competition

For this project I decided to use a canonical DDPG algorithm applied to multiple agens, but also an unconventional version of MADDPG, which turns to be very close to the previous implementation, except for the critic, where a shared state/action space is used in addition to the original DDPG critic.
This turns to add valuable information without introducing excess feature complexity and consequent difficulty to converge

## Learning Algorithm #1: DDPG

The Deep Deterministic Policy Gradient algorithm is a particular actor-critic method, where the critic is a neural network used to estimate the action value function  Q(s,a) and the actor mu(a) is another neural network that outputs the action maximizing Q. Hence, the training process evolves alternating the following steps:   

- training the Q network, minimizing the temporal difference error, with the actor network parameters fixed.
- training the actor nework mu, maximizing Q(s,mu(s,a)), with the Q network parameter fixed.

The high affinity with the DQN algorithm allows to use also the following improvements:

- target networks: in order to avoid instability issues, the expected Q-value at the time step t+1 is calculated using a network which is frozen periodically; the same is used for the critic network.
- experience replay buffer: the learning steps are carried on by mini-batches backpropagations, after sampling randomly from a buffer of memory, in order to avoid that too much correlated transitions drive the process to overfitting; using a multiagent training, the experience of each agent is collected in the same buffer, in order to update the common networks.

The training step has been performed once per timestep, that is, once every 2 experiences, using the target network soft-update (θ_target = τ*θ_local + (1 - τ)*θ_target). This should improve the learning stability. 

For exploration, I used the Ornstein-Uhlebeck noise, using a Gaussian sampling.

The actor neural network takes the state as input consists of 3 fully connected layers, with relu activations for the first 2 layers (hidden) and a tanh activation for the output (which in fact should be in the range (-1,1)).  
The critic neural network consists of 3 fully connected layers, with relu activations for the first 2 layers (hidden) and a linear activation for the output. The first input, the state, is taken by the fist layer, while the second one, the action, is taken by the second layer together with the output of the first one.

The hyperparameters used are:

- BUFFER_SIZE = int(1e6)  # replay buffer size
- BATCH_SIZE = 256        # minibatch size
- GAMMA = 0.95            # discount factor
- TAU = 5e-3              # for soft update of target parameters
- LR_ACTOR = 1e-3         # learning rate of the actor 
- LR_CRITIC = 1e-3        # learning rate of the critic 

I followed a quite aggressive setting, which turned to be quite effective. Nevertheless, the tuning is quite sensible to variations and I was forced to use a quite greedy choice of GAMMA to succeed.
The reward history during training is shown in the following picture:

![DDPG_trained][image1]
 
that is also visible in the `Tennis.ipynb` file together with a verbose logging of average rewards over the last 100 steps: it has taken 831 episodes to solve the problem, that is, in order to get an average reward greater than 0.5. The related weight files are `checkpoint_actor_ddpg.pth` and `checkpoint_critic_ddpg.pth`.

## Learning Algorithm #2: MADDPG with shared networks

My first attempts to use a canonical version of MADDPG (with or without shared networks) were insuccessful due to the difficulty of starting learning something overall. What my intuition suggested is that the critic space was now too prone to variance, especially for the Tennis game, where the collaboration effects are quite limited (the actors don't see each other!). Hence I decided to start from the DDPG, changing the critic so that:

-it outputs 1 value for each agent using 2 equal network branches (in practice, the same critic for each agent)
-it adds a correction contribution built from the last hidden space, shared between the two agents.

This way, the critic parameters increase is very limited and te results effectively show a slight improvement with respect to the DDPG. More importantly, the learning process has appeared to be much more stable, as well as the sensitivity to the hyperparameters has been reduced.
I could use the same values used in DDPG, except for GAMMA = .99 which reduces the greediness (DDPG was not learning at all with this choice).
The reward history during training is shown in the following picture:

![MADDPG_trained][image2]
 
that is also visible in the `Tennis.ipynb` file together with a verbose logging of average rewards over the last 100 steps: it has taken 829 episodes to solve the problem, that is, in order to get an average reward greater than 0.5. The related weight files are `checkpoint_actor_maddpg_singlenet.pth` and `checkpoint_critic_maddpg_singlenet.pth`.

## Possible improvements

The environment has been successfully solved with the DDPG and MADDPG algorithms.
For the future I would like to further explore the influence of some hyperparameters on the learning stability and speed, with particular attention on:   

- learning steps frequency
- batch size
- soft update rate

In addition, I'd like to evaluate a different exploration policy, like a variational approach on networks parameters, which should lead to a more consistent exploration.
It was evident that at a certain point increasing the buffer size was not helpful: using prioritized experience replay could improve things.

About the network side, I'd like to test also a solution using just the shared space for the last layer in the critic: convergence might be more difficult, but then the learning slope could be steeper.

Last but not least, I'd like to compare the performance of my MADDPG with Evolutionary strategy


