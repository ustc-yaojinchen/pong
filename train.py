def train(agent):
    for episode in range(agent.config['train_episode']):
        agent.reset()
        while True:
            agent.train_action()

            if agent.terminated:
                agent.optimize()
                agent.log()
                agent.clear_buffer()
                break

            if len(agent.observations) == agent.config['num_step']:
                agent.optimize()
                agent.log()
                agent.clear_buffer()

def test(agent):
    for episode in range(agent.config['test_episode']):
        agent.reset()
        while True:
            agent.test_action()
            if agent.terminated:
                break

    agent.save_model()
    agent.log()
    agent.clear_buffer()