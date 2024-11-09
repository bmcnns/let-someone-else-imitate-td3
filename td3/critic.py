from td3.TD3 import TD3

class TD3CriticWrapper():
    def __init__(self):
        kwargs = {"state_dim": 17, "action_dim": 6, "max_action": 1.0, "discount": 0.99, "tau": 0.005,
                  "policy_noise": 0.2 * 1.0, "noise_clip": 0.5 * 1.0, "policy_freq": 2}

        self.td3 = TD3(**kwargs)
        self.td3.load('td3/TD3_HalfCheetah-v3_9')

    def predict(self, states, actions):
        return self.td3.critic.Q1(states, actions)