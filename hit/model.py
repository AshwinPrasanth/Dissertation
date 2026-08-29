import torch
import torch.nn as nn


class BranchingMLP(nn.Module):

    def __init__(
        self,
        input_dim=15,
        hidden_dims=(128, 64, 32),
        dropout=0.1,
    ):

        super().__init__()

        layers = []

        prev = input_dim

        for hidden in hidden_dims:

            layers.append(
                nn.Linear(
                    prev,
                    hidden,
                )
            )

            layers.append(
                nn.ReLU()
            )

            layers.append(
                nn.Dropout(
                    dropout
                )
            )

            prev = hidden

        layers.append(
            nn.Linear(
                prev,
                1,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        candidate_features,
    ):

        scores = self.network(
            candidate_features
        )

        return scores.squeeze(
            -1
        )


class PairwiseRankingLoss(
    nn.Module,
):

    def __init__(
        self,
        margin=1.0,
    ):

        super().__init__()

        self.margin = margin

    def forward(
        self,
        pred_scores,
        sb_scores,
    ):

        loss = 0.0

        pairs = 0

        n = len(
            pred_scores
        )

        for i in range(n):

            for j in range(i + 1, n):

                if (
                    sb_scores[i]
                    ==
                    sb_scores[j]
                ):

                    continue

                if (
                    sb_scores[i]
                    >
                    sb_scores[j]
                ):

                    better = i

                    worse = j

                else:

                    better = j

                    worse = i

                loss += torch.relu(

                    self.margin

                    -

                    (

                        pred_scores[better]

                        -

                        pred_scores[worse]

                    )

                )

                pairs += 1

        if pairs == 0:

            return pred_scores.sum() * 0

        return loss / pairs


class BranchingLoss(
    nn.Module,
):

    def __init__(
        self,
        mode="cross_entropy",
    ):

        super().__init__()

        self.mode = mode

        self.ce = nn.CrossEntropyLoss()

        self.mse = nn.MSELoss()

        self.rank = PairwiseRankingLoss()

    def forward(

        self,

        pred_scores,

        chosen,

        sb_scores,

    ):

        if self.mode == "cross_entropy":

            logits = pred_scores.unsqueeze(
                0
            )

            target = chosen.unsqueeze(
                0
            )

            return self.ce(

                logits,

                target,

            )

        if self.mode == "mse":

            return self.mse(

                pred_scores,

                sb_scores,

            )

        if self.mode == "pairwise":

            return self.rank(

                pred_scores,

                sb_scores,

            )

        raise ValueError(
            self.mode
        )


if __name__ == "__main__":

    model = BranchingMLP()

    x = torch.randn(

        87,

        15,

    )

    scores = model(

        x

    )

    print()

    print("=" * 60)

    print("MODEL TEST")

    print("=" * 60)

    print(

        "Input :", x.shape

    )

    print(

        "Output:", scores.shape

    )

    print(

        "Parameters:",

        sum(

            p.numel()

            for p in model.parameters()

        ),

    )