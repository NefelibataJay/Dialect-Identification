import torch

class LoRALinearLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, use_lora ,rank, alpha=0.5):
        super(LoRALinearLayer, self).__init__()
        if use_lora:
            std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
            self.W_a = torch.nn.Parameter(torch.randn(input_dim, rank) * std_dev)
            self.W_b = torch.nn.Parameter(torch.randn(rank, output_dim))
            self.alpha = alpha

        self.use_lora = use_lora
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        if not self.use_lora:
            return self.linear(x)
        else:
            return self.linear(x) + self.alpha * (x @ self.W_a @ self.W_b)

    def merge_weights(self):
        if self.use_lora:
            self.linear.weight += self.alpha * (self.W_a @ self.W_b)
        else:
            print("LoRA is not used in this layer")
