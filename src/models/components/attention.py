import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, query_dim: int) -> None:
        super().__init__()

        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(
                torch.empty(query_dim).uniform_(-0.1, 0.1))

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        """ 
        Args: 
            input_vector (torch.Tensor): User tensor of shape (batch_size, hidden_dim, output_dim)

        Returns:
            torch.Tensor: User tensor of shape (batch_size, news_emb_dim).
        """
        # batch_size, hidden_dim, output_dim
        attention = torch.tanh(self.linear(input_vector))

        # batch_size, hidden_dim
        attention_weights = F.softmax(torch.matmul(attention, self.query), dim=1)

        # batch_size, output_dim
        weighted_input = torch.bmm(
                attention_weights.unsqueeze(dim=1), input_vector
                ).squeeze(dim=1)
        
        return weighted_input

