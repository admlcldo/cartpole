# Cart Pole

Trains a neural network on the cartpole environment from OpenAI gym.

The neural network is built using TensorFlow.

Repository contains a jupyter notebook and a regular python script of the same code.

The training algorithm is very simple: 
1) play a batch of episodes
2) take all episodes which performed better than some threshold value
3) reinforce all the moves from these good episodes
4) increase the threshold value

we increase the threshold value by taking the maximum of the current threshold and the average
survival time (in terms of steps) from the batch. The maximum possible number of steps in an episode is 200


Here is an episode early in the training, it dies almost immediately.
![Early Training](https://github.com/admlcldo/cartpole/blob/master/early_training.gif)

Here is an episode near the end of training 500 iterations (batch size 20). It survives
200 steps
![Late Training](https://github.com/admlcldo/cartpole/blob/master/late_training.gif)
