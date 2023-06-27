import gym
import random
import torch
import json
import os
import numpy as np
from datetime import datetime

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

    env = gym.make(config['env'])
    env.seed(config['seed'])

    if config['log_dir'] == '':
        config['log_dir'] = os.path.join('log', config['env'],  datetime.now().strftime('%b%d_%H_%M'))
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    model = Model()
    if config['load_model_dir'] != '':
        model.load_state_dict(torch.load(config['load_model_dir'], map_location=lambda storage, loc: storage))
    
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'])#考虑到Adam之类的优化器需要考虑历史输入，所以不在train()里面定义

    train_agent = Agent(config, env, model, optimizer)#考虑到以后offline算法，为了保存所有的数据，所以agent不在train()里面进行定义
    test_agent = Agent(config, env, model, None)
    for epoch in range(config['epochs']):
        train(train_agent)
        test(test_agent)

if __name__ == '__main__':
    main()