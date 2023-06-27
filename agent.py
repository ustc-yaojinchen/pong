import os
import torch

class Agent():
    def __init__(self, config, env, model, optimizer):
        self.config = config
        self.env = env
        self.model = model
        self.optimizer = optimizer

        self.observation = None
        self.action_probability = None
        self.action = None
        self.value = None
        self.reward = None
        self.terminated = None

        self.truncated = None 
        self.info = None
        
        self.observations = []#buffer
        self.action_probabilities = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.terminateds = []

    def clear_buffer(self):
        self.observations = []
        self.action_probabilities = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.terminateds = []

    def reset(self):#重开一把
        self.observation = None
        self.action_probability = None
        self.action = None
        self.value = None
        self.reward = None
        self.terminated = None

        self.truncated = None 
        self.info = None
        
        self.observation, self.info = self.env.reset()

    def test_action(self):#测试行动一步，并记录中间数据到buffer里
        self.observations.append(self.observation)

        self.action_probability, self.value = self.model(self.observation)
        self.action = int(self.action_probability[0].data.argmax())

        self.action_probabilities.append(self.action_probability)
        self.actions.append(self.action)
        self.values.append(self.value)

        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(self.action)

        self.rewards.append(self.reward)
        self.terminateds.append(self.terminated)
    
    def train_action(self):#训练行动一步，并记录中间数据到buffer里
        self.observations.append(self.observation)

        self.action_probability, self.value = self.model(self.observation)
        self.action = int(self.action_probability[0].data.multinomial(1))

        self.action_probabilities.append(self.action_probability)
        self.actions.append(self.action)
        self.values.append(self.value)

        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(self.action)
        
        self.rewards.append(self.reward)
        self.terminateds.append(self.terminated)

    def optimize(self):#使用buffer里面的数据进行优化
        policy_loss = 0
        value_loss = 0

        if self.terminated:
            R = 0
        else:
            _, R = self.model(self.observation)
            R = R.data
        
        for t in reversed(range(len(self.observations))):
            R = self.rewards[t] + self.config["gamma"] * R
            Advantage = R - self.values[t]
            entropy =  - (self.action_probabilities[t] * self.action_probabilities[t].log()).sum()
            policy_loss -= Advantage.data * self.action_probabilities[t].log()[0, self.actions[t]] + self.config['beta'] * entropy
            value_loss += (R - self.values[t]) ** 2
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, best_reward=[float('-inf')]):#读取buffer里面的数据作为评估，并保存模型参数。bset_reward作为一个长期使用的存储器
        path = os.path.join(self.config['log_dir'], 'new.pt')
        torch.save(self.model.state_dict(), path)

        reward = sum(self.rewards) / sum(self.terminateds)
        if reward > best_reward[0]:
            best_reward[0] = reward
            path = os.path.join(self.config['log_dir'], 'best_{:.4f}.pt'.format(reward))
            torch.save(self.model.state_dict(), path)

    def log(self):#把buffer里面需要的数据写进log里
        with open(os.path.join(self.config['log_dir'], 'log'), 'a') as f:
            for t in range(len(self.observations)):
                action_probability = self.action_probabilities[t].tolist()[0]
                action_probability = [round(num, 2) for num in action_probability]
                action = int(self.actions[t])
                value = float(self.values[t])
                reward = float(self.rewards[t])
                
                f.write('{:}\t{:}\t{:.4f}\t{:.2f}\n'.format(action_probability, action, value, reward))
                if self.terminateds[t]:
                    f.write('\n')