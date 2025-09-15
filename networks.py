import torch
import torch.nn as nn

class SimpleFCN(nn.Module):
    def __init__(self, input_size, hidden_size=[64,64]):
        """
        Default 2 hidden layers each with 64 neurons to match the original PPO paper
        Proximal Policy Optimization Algorithms” by Schulman et al., 2017.
        """
        super(SimpleFCN, self).__init__()
        #The original PPO paper uses tanh, this uses ReLu.
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
        )
        self.output_size = hidden_size[1]
    def forward(self, x):
        return self.network(x)
    

if __name__ == "__main__":
    test_network = SimpleFCN(4)
    x = torch.randn(1, 4)  
    output = test_network(x)
    print(output)
    print("Output shape:", output.shape)
