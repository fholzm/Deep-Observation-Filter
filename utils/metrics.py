import torch


class NMSE(torch.nn.Module):
    """Calculate the normalized mean squared error between two signals."""

    def __init__(self, return_dB: bool = False, per_sample: bool = False):
        """Calculate the normalized mean squared error between two signals.

        Parameters
        ----------
        return_dB : bool, optional
            Return the result in dB, by default False
        per_sample : bool, optional
            Calculate the error per sample, by default False
        """
        super(NMSE, self).__init__()
        self.return_dB = return_dB
        self.per_sample = per_sample

    def forward(self, y_pred, y_true):
        """Calculate the normalized mean squared error between two signals.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted signal.
        y_true : torch.Tensor
            Reference signal.

        Returns
        -------
        torch.Tensor
            Relative mean squared error
        """
        if self.per_sample:
            ndim_input = y_pred.ndim
            dim_mean = tuple(range(1, ndim_input))
            nmse = torch.mean((y_pred - y_true) ** 2, dim=dim_mean) / torch.mean(
                y_true**2, dim=dim_mean
            )

        else:
            nmse = torch.mean((y_pred - y_true) ** 2) / torch.mean(y_true**2)

        if self.return_dB:
            nmse = 10 * torch.log10(nmse)  # Convert to dB

        return nmse
