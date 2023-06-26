import gym
import random
import torch
import json
import numpy as np

from model import Model
from agent import Agent
from train import train, test

def main():
    with open('config.json', 'r') as f:
        config = json.loads(f.read())
    
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True

    env = gym.make(config['env'], render_mode='human')
    env.seed(config['seed'])

    model = Model()
    if config['load_model_dir'] != '':
        model.load_state_dict(torch.load(config['load_model_dir'], map_location=lambda storage, loc: storage))
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'])

    train_agent = Agent(config, env, model, optimizer)
    test_agent = Agent(config, env, model, None)
    
    for epoch in range(config['epochs']):
        train(train_agent)
        test(test_agent)

if __name__ == '__main__':
    main()