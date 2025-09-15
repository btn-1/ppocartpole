import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import SimpleFCN

class PPOPolicy(nn.Module):
    def __init__(self, network, num_actions):
        """ Actor critic network

        network (SimpleFCN): The shared network for actor and critic. SimpleFCN for Polecart.

        num_actions (int):  The number of possible action of the agent. 
        For polecat num_actions = 2 (left or right)
        """
        
        super(PPOPolicy, self).__init__()
        self.network = network
        # Policy head
        self.actor = nn.Linear(network.output_size, num_actions)
        # Value head
        self.critic  = nn.Linear(network.output_size, 1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):

        h = self.network(input)

        actor_output = self.softmax(self.actor(h))
        critic_output = self.critic(h)

        return actor_output, critic_output


    def act(self, x):
        #TODO
        pass 


    
if __name__ == "__main__":
    test_network = SimpleFCN(4)
    test_policy = PPOPolicy(test_network,2)
    x = torch.randn(1, 4)  
    output = test_policy(x)
    print(output)
    print("Output shape:", output[1].shape)
