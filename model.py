import torch
import torch.nn as nn


class GeneExpressionPredictionModel(nn.Module):
    def __init__(self):
        super(GeneExpressionPredictionModel, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

        self.embed_1 = nn.Linear(2048, 1024)
        self.embed_2 = nn.Linear(985, 512)
        self.ff = nn.Linear(1024 + 512, 2048)
        self.output = nn.Linear(2048, 978)

    def forward(self, x_fp, x_cp):
        # x_fp: (batch_size, 2048)
        # x_cp: (batch_size, 985)

        x_fp = self.embed_1(x_fp)  # (batch_size, 1024)
        x_fp = self.dropout(x_fp)  # (batch_size, 1024)

        x_cp = self.embed_2(x_cp)  # (batch_size, 512)
        x_cp = self.dropout(x_cp)  # (batch_size, 512)

        x = torch.cat((x_fp, x_cp), dim=1)  # (batch_size, 1024 + 512)
        x = self.ff(x)  # (batch_size, 2048)
        x = self.sigmoid(x)  # (batch_size, 2048)

        x = self.output(x)  # (batch_size, 978)

        return x
