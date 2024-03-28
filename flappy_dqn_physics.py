import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from game.flappy_bird import GameState


class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        """ 
        The neural network architecture is the same as DeepMind 
        used in the paper Human-level control through deep reinforcement learning.
        https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
        
        
    
        """
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        #All the hyperparameters above are choosen from the implementation in the research paper as we lack resources to hyperparamter tune
        #Number of actions are 2 -> Flap or no flap
        
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def init_weights(m):
    """
    Function to initialize the weights using uniform distribution

    """
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    
    """
    We are resizing the image to crop the floor and transform the image to grayscale as it converges faster
    We took this methedology from Deepmind's training Atari breakout 
    """
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def train(model, start, physics = True):
    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState(physics=physics)

    # initialize replay memory
    replay_memory = []
    #replay_memory stores experiences (state, action, reward, next state, and terminal flag tuples) 

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    
    # Initialize tracking variables
    train_losses = []
    rewards_per_episode = []
    scores_per_episode = []
    max_q_values = []
    epsilon_history = []

    current_reward = 0
    current_score = 0  # Assuming you have a way to track score in your game
    
    image_data, reward, terminal, score = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)
    
    """
    DeepMind's original DQN architecture for playing Atari games used a stack of four frames
    as the input to the neural network. This design choice allows the network to learn 
    from both the current state and recent past
    
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)
    

    # main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]
        
        if iteration%10000 == 0:
            epsilon_history.append(epsilon)

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        #if random_action:
        #    print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal, score = game_state.frame_step(action)
        current_reward += reward
        current_score += score
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        
        if iteration%10000 == 0:
            max_q_values.append(torch.max(output).item())
        
        """ 
        The new state tensor (state_1) is formed by concatenating the latest image (image_data_1)
        with the last three images from the previous state tensor (state), after removing the oldest image. 
        This operation maintains a rolling window of the four most recent frames, 
        """

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
        
        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)
        
        if iteration%10000 == 0:
            train_losses.append(loss.item())  # Tracking training loss

        # do backward pass
        loss.backward()
        optimizer.step()


        # set state to be state_1
        state = state_1
        
        if terminal:
            rewards_per_episode.append(current_reward)
            scores_per_episode.append(current_score)  
            current_reward = 0  # Reset for the next episode
            current_score = 0  # Reset score for the next episode
        
        iteration += 1

        if iteration % 100000 == 0:
            torch.save(model, "pretrained_model/DQN_Flappy_model_physics" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))
    
    with open('training_logs.pkl', 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'rewards_per_episode': rewards_per_episode,
            'scores_per_episode': scores_per_episode,
            'max_q_values': max_q_values,
            'epsilon_history': epsilon_history,
        }, f)
        
    def plot_and_save(data, title, xlabel, ylabel, filename):
        plt.figure(figsize=(10, 7))
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(filename)
        plt.close()

    plot_and_save(train_losses, 'Training Loss over Iterations', 'Iteration', 'Loss', 'training_loss.png')
    plot_and_save(rewards_per_episode, 'Rewards over Episodes', 'Episode', 'Reward', 'rewards_per_episode.png')
    plot_and_save(scores_per_episode, 'Scores over Episodes', 'Episode', 'Score', 'scores_per_episode.png')
    plot_and_save(max_q_values, 'Max Q Values over Iterations', 'Iteration', 'Max Q Value', 'max_q_values.png')
    plot_and_save(epsilon_history, 'Epsilon over Iterations', 'Iteration', 'Epsilon', 'epsilon_history.png')


def test(model, physics = True):
    game_state = GameState(mode='test', physics = physics)

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal,score = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)
    score_present = 0
    while not terminal:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal,score = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        #score_present = score
        # set state to be state_1
        state = state_1
    
    #print(f"Score acheived: {score}")
    return score
        
def test_avg(model, physics=True, episodes=100):
    scores = []
    
    for _ in tqdm(range(episodes)):
        score = test(model, physics=physics)
        scores.append(score)
    
    print("Testing done")
    # Plotting the score distribution
    plt.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(scores), color='red', linestyle='dashed', linewidth=1)
    plt.title(f'Score Distribution over {episodes} Episodes without Physics - Using vanilla flappy bird agent')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.text(np.mean(scores), plt.ylim()[1]*0.9, 'Mean: {:.2f}'.format(np.mean(scores)), color = 'red')
    plt.savefig('score_distribution_vanila_flappy_bird.png')
    #plt.show()    
    
def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load(
            'pretrained_model\DQN_Flappy_model_physics2000000.pth',
            
            map_location='cpu' if not cuda_is_available else None
        ).eval()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        test(model, physics=True)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = NeuralNetwork()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start, physics=True)
    
    elif mode == 'test_avg':
        model = torch.load(
            'pretrained_model\DQN_Flappy_model_physics2000000.pth',
            map_location='cpu' if not cuda_is_available else None
        ).eval()
        
        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()
        
        test_avg(model, physics=True)


if __name__ == "__main__":
    main(sys.argv[1])
