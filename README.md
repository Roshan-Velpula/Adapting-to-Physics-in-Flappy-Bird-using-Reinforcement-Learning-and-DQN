# Deep Reinforcement Learning for Flappy Bird with Physics

This project explores the application of Deep Q-Networks (DQN) to the classic game Flappy Bird, with an added twist: we introduce physics elements such as gravity and wind speed to create a more challenging and unpredictable environment. Our approach is inspired by DeepMind's pioneering work on training DQNs for Atari games, adapting it to navigate the complexities of a physics-enhanced Flappy Bird.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have Python installed on your machine along with the necessary libraries, including PyTorch, Pygame, numpy, and matplotlib for running the simulations and plotting results. 

### Running the Tests

To evaluate the performance of the trained agent, use the following command:

```bash
python flappy_dqn_physics.py test_avg

```
This will run the test for 100 episodes and plot the score distribution. 

### To train the model from scratch

```
python flappy_dqn_physics.py train

```

### Switching Physics On/Off

The project allows testing both the vanilla version of Flappy Bird and the version enhanced with physics elements. To switch between these modes:

- For testing the agent trained **without physics**, load the model ```current_model_without_physics2000000.pth.```
- To test the agent trained **with physics**, the default model ```DQN_Flappy_model_physics2000000.pth``` is used.

To modify the physics flag in the test mode, adjust the physics flag to True (for physics) or False (no physics) in the test function definition within the flappy_dqn_physics.py script.

### Project Structure
- flappy_dqn_physics.py: The main script containing the DQN model, training, and testing logic.
- pretrained_model/: Directory containing the pretrained models.
- - DQN_Flappy_model_physics2000000.pth: Model trained with physics.
- - current_model_without_physics2000000.pth: Model trained without physics elements.

### Built With
- PyTorch - The deep learning framework used.
- Pygame - For creating the game environment.
- Numpy - For numerical computations.
- Matplotlib - For plotting results.

### Authors

- Sakshi Singh
- Mariana Jimenez Vega
- Roshan Velpula

### References

- https://github.com/sourabhv/FlapPyBird
- https://github.com/yenchenlin/DeepLearningFlappyBird -> modified FlapPyBird game engine adjusted for reinforcement learning is used from this TensorFlow project
- https://ai.intel.com/demystifying-deep-reinforcement-learning/
- https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
- http://cs229.stanford.edu/proj2015/362_report.pdf
- https://en.wikipedia.org/wiki/Convolutional_neural_network
- https://en.wikipedia.org/wiki/Reinforcement_learning
- https://en.wikipedia.org/wiki/Markov_decision_process
- http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
- https://en.wikipedia.org/wiki/Flappy_Bird
- https://pytorch.org/
- https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial
