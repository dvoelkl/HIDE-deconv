"""
=====================================================
Model for hierarchical cell type deconvolution
=====================================================
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class HIDE(nn.Module):
    """
    Hierarchical Cell Type Deconvolution Model.

    - Train the model with *.train(...)*
    - Predict with *.predict(...)*
    """

    def __init__(
        self, X_l: list[pd.DataFrame], A_l: list[pd.DataFrame], lambdaNMSE: float = 0.0
    ):
        """
        Constructor of HIDE deconvolution model.

        Parameters
        ----------
        X_l : list[pd.DataFrame]
            List of reference profiles for each cell layer. First element should represent the finest coarsed resolution. (genes x celltypes_l)
        A_l : list[pd.DataFrame]
            List of projection matrices, projecting from the finest coarsed resolution to a higher one. The first element should always
            be an identity matrix. (celltypes_l x celltypes_fine)
        lambdaNMSE : float = 0.0
            Weighting of optional NMSE contribution in general loss.

        """
        super().__init__()

        self.L = len(A_l)
        self.lambdaNMSE = lambdaNMSE

        self.celltype_layer_labels = [X.columns for X in X_l]
        self.gene_labels = X_l[0].index

        self.p, _ = X_l[0].shape
        self.q_l = [len(X.columns) for X in X_l]

        self.A_l = [torch.tensor(A.to_numpy(), dtype=torch.float32) for A in A_l]
        self.X_l = [torch.tensor(X.to_numpy(), dtype=torch.float32) for X in X_l]

        self.g_l = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(self.p, dtype=torch.float32).uniform_(0.001, 0.1),
                    requires_grad=True,
                )
                for _ in range(self.L)
            ]
        )

        self.progress_bar = Progress(
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        )

    def get_loss(self, C, C_est):
        corr_terms = []
        nmse_terms = []

        for layer in range(self.L):
            A = self.A_l[layer]

            C_l = A @ C
            C_est_l = A @ C_est

            mu_t = C_l.mean(dim=1, keepdim=True)
            mu_h = C_est_l.mean(dim=1, keepdim=True)

            vt = C_l - mu_t
            vh = C_est_l - mu_h

            num = (vt * vh).sum(dim=1)
            den = torch.sqrt((vt.pow(2).sum(dim=1) * vh.pow(2).sum(dim=1)))
            r = num / den

            corr_terms.append(-r.mean())

            nmse_num = (C_l - C_est_l).pow(2).sum()
            nmse_den = (C_l).pow(2).sum()
            nmse_terms.append(nmse_num / nmse_den)

        corr_loss = torch.stack(corr_terms).mean()
        nmse_loss = torch.stack(nmse_terms).mean()
        total_loss = corr_loss + self.lambdaNMSE * nmse_loss

        return total_loss, corr_loss, nmse_loss

    def train(self, Y: pd.DataFrame, C: pd.DataFrame, iter: int = 1000) -> list:
        """
        Trains the HIDE model.

        Parameters
        ----------
        Y : pd.DataFrame
            Bulk training data. (genes x samples)
        C : pd.DataFrame
            True cellular composition at finest coarsed cell resolution. (celltypes x samples)
        iter : int = 1000
            Number of training iterations.

        Returns
        -------
        list
            Loss per epoch.
        """
        Y = torch.tensor(Y.to_numpy(), dtype=torch.float32)
        C = torch.tensor(C.to_numpy(), dtype=torch.float32)

        optim = torch.optim.Adam(self.parameters(), lr=0.001)

        losses = []

        with self.progress_bar as p:
            for e in p.track(range(iter)):
                A_l = []
                B_l = []
                for layer in range(self.L):
                    g = torch.sqrt(self.g_l[layer] ** 2)
                    g = g * self.p / g.sum()
                    G_l = g.unsqueeze(1)
                    A_l.append(G_l * Y)
                    B_l.append(G_l * (self.X_l[layer] @ self.A_l[layer]))

                A_stack = torch.vstack(A_l)
                B_stack = torch.vstack(B_l)

                C_est = torch.linalg.lstsq(B_stack, A_stack).solution
                loss, _, _ = self.get_loss(C, C_est)

                loss.backward()
                optim.step()
                optim.zero_grad()

                losses.append(loss.item())

        # Norm weights and make non-negative
        # Option 1 => Might break relationship between layers
        for layer in range(self.L):
            gamma_l = self.g_l[layer] ** 2
            gamma_l = self.p * gamma_l / torch.sum(gamma_l)
            g_l = torch.sqrt(gamma_l)
            self.g_l[layer] = g_l

        return losses

    @torch.no_grad()
    def predict(
        self, Y: pd.DataFrame, norm: bool = False
    ) -> dict[str, list[pd.DataFrame]]:
        """
        Predicts the cellular composition of a bulk.

        Parameters
        ----------
        Y : pd.DataFrame
            Bulk samples to be deconvoluted. (genes x samples)
        norm : bool = False
            Norm the results to one.

        Returns
        -------
        dict[str,list[pd.DataFrame]]
            Dictionary where key "prediction" holds a list of compositions of each cell type layer.
            List elements are ordered in the same way as X_l and A_l. Element 0 corresponds to the finest
            coarsed cell type layer.
        """

        sample_names = list(Y.columns)
        Y = torch.tensor(Y.to_numpy(), dtype=torch.float32)

        A_l = []
        B_l = []
        for layer in range(self.L):
            G_l = self.g_l[layer].unsqueeze(1)
            A_l.append(G_l * Y)
            B_l.append(G_l * (self.X_l[layer] @ self.A_l[layer]))

        A_stack = torch.vstack(A_l)
        B_stack = torch.vstack(B_l)

        C_est = torch.linalg.lstsq(B_stack, A_stack).solution
        C_est[C_est < 0] = 0.0

        if norm:
            C_est = C_est / C_est.sum(dim=0)

        prediction = []
        for layer in range(self.L):
            C_l = self.A_l[layer] @ C_est
            C_l = pd.DataFrame(
                C_l.detach().cpu().numpy(),
                index=self.celltype_layer_labels[layer],
                columns=sample_names,
            )
            prediction.append(C_l)

        return {"prediction": prediction}
