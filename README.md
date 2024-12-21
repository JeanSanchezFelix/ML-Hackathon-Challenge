
# Space Mission Challenge - IEEE Hackathon 2023

## Overview
In 2030, a team of space enthusiasts launched a mission to explore a distant planet in a far-off galaxy. Their journey is filled with numerous challenges, including data analysis to determine the success rate and building an autonomous system for the spaceship. This project tackles these challenges through two main components:

- **Space Mission Challenge - IEEE Hackathon 2023**: Analyzing mission data to predict the success rate of landing and implementing an autonomous system for spaceship navigation.
  
- **Submission**: Implementing an intelligent agent to autonomously navigate the spaceship and interact with planets using machine learning.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/JeanSanchezFelix/ML-Hackathon-Challenge.git
   ```

## Beginner Challenge - Success Spaceship Landings

### Data Analysis

The core idea is to analyze mission data to determine the success of spaceship landings. The dataset used in this project contains synthetic data about spaceship missions. It includes information such as:

- Launch year
- Mission duration
- Distance to the planet
- Planet gravity
- Initial and final speed of the spaceship
- Fuel consumption and more

### Data Import and Processing

We use `pandas` to import and process the dataset:
```python
import pandas as pd

# Read synthetic space mission data
flights = pd.read_csv('synthetic_space_dataset.csv')
```

Sample of the data:
```python
flights.sample(5)
```

### Model Building

A logistic regression model is used to predict the success of spaceship landings based on mission data. The dataset is split into training and test sets to evaluate the model's performance.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Example of using Logistic Regression for prediction
model = LogisticRegression()
X = flights.drop('success_landing', axis=1)
y = flights['success_landing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
```

## Intermediate Challenge - Spaceship Planet Landing

### Simulation

In this section, we implement an autonomous agent that uses a deep reinforcement learning algorithm to navigate the spaceship. This model interacts with a dynamic environment, applying gravity forces from planets and adjusting its trajectory.

### Game Setup

The game environment is created using `arcade`, where a spaceship interacts with planets that have different characteristics. The agent learns to navigate by training through multiple game iterations.

```python
import arcade
import random
import math

# Game setup constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
SPACESHIP_SPEED = 0.01
GRAVITATIONAL_FORCE = 5

# Initialize the spaceship and planets
class Spaceship:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity_x = 0
        self.velocity_y = 0
        self.angle = 0
        self.pulling_planet = None
```

### Reinforcement Learning Model

The agent uses a neural network model (`Linear_QNet`) to decide the best actions based on the current state. It uses Q-learning to update the model and improve its decision-making process.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
```

### Training the Agent

The agent learns by playing multiple games, adjusting its actions based on feedback (rewards) from the environment. The learning process involves:

- Storing game states
- Making predictions using the trained model
- Updating the model based on the outcome of each action

```python
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # control randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=100_000)
        self.model = Linear_QNet(16, 256, 4)
        self.trainer = QTrainer(self.model, lr=0.001, gamma=self.gamma)

    def get_action(self, state):
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            return [1 if i == move else 0 for i in range(4)]
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            return [1 if i == move else 0 for i in range(4)]
```

### Running the Game

To run the game and train the agent:
```python
# Game loop setup
game = Game("Spaceship Navigation")
game.run()
```

### Results

The agent is trained over multiple games, and the scores are plotted to visualize the learning progress.

## Conclusion

Both projects aim to demonstrate advanced data analysis techniques and machine learning approaches to simulate a space mission's success and develop an autonomous spaceship system.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
