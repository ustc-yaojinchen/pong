import  torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.AvgPool2d(5, 5),
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),

            nn.AvgPool2d(3, 3),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),

            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 32, 3),
            nn.ReLU()
        )
        self.actor = nn.Linear(192, 6)
        self.critic = nn.Linear(192, 1)

    def forward(self, x):
        x = torch.tensor(x.transpose(2,0,1))
        x = x.unsqueeze(0).float()

        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)

        action_probability = nn.functional.softmax(self.actor(x), dim=1)
        value = self.critic(x)

        return action_probability, value