# cartpole

Trains a neural network on the cartpole environment from OpenAI gym.
An episode of cartpole is won when you balance for 200 frames.

The neural network is built using tensorflow.

The training algorithm is very simple: 
1) play a batch of episodes
2) take all episodes which performed better than some threshold value
3) reinforce all the moves from these good episodes
4) increase the threshold value

we increase the threshold value by taking the maximum of the current threshold and the average
survival time from the batch
