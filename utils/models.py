import torch
import numpy as np
import scipy


class ConvolutionalObervationFilter_FA(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fs = config["samplerate"]
        self.nfft = config["nfft"]
        self.n_coeffs = config["nfft"] - config["model"]["hopsize"] + 1
        self.n_hidden_units = config["model"]["hidden_units"]
        self.beta = config["model"]["corr_beta"]

        # Calculate required number of input features
        self.n_remotemics = len(config["remote_mic"]["position"])

        aperture = 2 * np.sqrt(2) * config["remote_mic"]["scale"]
        self.max_lag = int(np.ceil(np.max(aperture) / config["data"]["c"] * self.fs))

        # Adding 3 input features for carthesian coordinates of the virtual microphone
        self.input_features = (
            int(scipy.special.comb(self.n_remotemics, 2)) * (self.max_lag * 2 + 1) + 3
        )
        self.output_features = self.n_remotemics * self.n_coeffs

        self.input_corr = GCC_ED_mat(
            self.nfft,
            self.max_lag,
            beta=self.beta,
            processor=config["model"]["correlator"],
        )

        if config["model"]["activation"] == "relu":
            act = torch.nn.ReLU()
        elif config["model"]["activation"] == "leakyrelu":
            act = torch.nn.LeakyReLU()
        elif config["model"]["activation"] == "gelu":
            act = torch.nn.GELU()
        elif config["model"]["activation"] == "sigmoid":
            act = torch.nn.Sigmoid()

        self.encoder = torch.nn.Sequential(
            # Stage 1
            torch.nn.Conv1d(
                in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            act,
            torch.nn.Conv1d(
                in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1
            ),
            act,
            # Stage 2
            torch.nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            act,
            torch.nn.Conv1d(
                in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            act,
            # Stage 3
            torch.nn.Conv1d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            act,
            torch.nn.Conv1d(
                in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
            ),
            act,
            # Stage 4
            torch.nn.Conv1d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            act,
            torch.nn.Conv1d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            act,
            torch.nn.Flatten(),
        )

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Linear(128 * 3 + 3, self.n_hidden_units),
            act,
            torch.nn.Dropout(config["train"]["dropout"]),
            torch.nn.Linear(self.n_hidden_units, 128 * 3),
            act,
            torch.nn.Dropout(config["train"]["dropout"]),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, (128, 3)),
            # Stage 1
            torch.nn.ConvTranspose1d(
                in_channels=128, out_channels=64, kernel_size=5, stride=3, padding=1
            ),
            act,
            torch.nn.ConvTranspose1d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            act,
            # Stage 2
            torch.nn.ConvTranspose1d(
                in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            act,
            torch.nn.ConvTranspose1d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            act,
            # Stage 3
            torch.nn.ConvTranspose1d(
                in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1
            ),
            act,
            torch.nn.ConvTranspose1d(
                in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            act,
            # Stage 4
            torch.nn.ConvTranspose1d(
                in_channels=16,
                out_channels=self.n_remotemics,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            act,
            torch.nn.ConvTranspose1d(
                in_channels=self.n_remotemics,
                out_channels=self.n_remotemics,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x, virt_coords, reset_corr=False):
        if x.ndim == 3:
            x = x.unsqueeze(0)

        if virt_coords.ndim == 1:
            virt_coords = virt_coords.unsqueeze(0)

        assert x.shape[0] == virt_coords.shape[0], "Batch size mismatch."

        batch_size = x.shape[0]

        x = self.input_corr(x, reset_corr)

        x = self.encoder(x)
        x = torch.cat((x, virt_coords), dim=-1)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return torch.fft.rfft(
            x.reshape(batch_size, self.n_remotemics, self.n_coeffs), n=self.nfft
        )


class GCC_ED_mat:
    """
    Generalized cross-correlation with cross correlation or phase transform (GCC-PHAT) processor [1] with exponential decay.

    [1] Knapp, C. H., & Carter, G. C. (1976). The generalized correlation method for estimation of time delay. IEEE Transactions on Acoustics, Speech, and Signal Processing, 24(4), 320–327. https://doi.org/10.1109/TASSP.1976.1162830
    """

    def __init__(self, nfft, max_lag, beta=1.0, processor="cc", regularization=0.0):
        """Generalized cross-correlation with cross correlation or phase transform (GCC-PHAT) processor with exponential moving average.

        Args:
            nfft (int): FFT size.
            max_lag (int): Maximum lag in samples.
            beta (float, optional): Decay of exponentially weighted average. Defaults to 1.0 (use just current value).
            processor (str, optional): Processor to use. Either 'cc' or 'phat'. Defaults to 'cc'.
            regularization (float, optional): Regularization factor. Defaults to 0.0.
        """
        assert 0.0 < beta <= 1.0, "Beta must be in range (0, 1]."

        self.nfft = nfft
        self.max_lag = max_lag
        self.regularization = regularization
        self.beta = beta
        self.R_buffer = None

        if processor.lower() == "cc" or processor.lower() == "phat":
            self.processor = processor.lower()
        else:
            raise ValueError("Processor must be either 'cc' or 'phat'.")

    def __call__(self, X, reset=False):
        """
        Args:
            X (torch.Tensor): Input tensor of shape (channels, nfft, blocks) or (batch, channels, nfft, blocks).
            reset (bool, optional): Reset buffer. Defaults to False.

        Returns:
            torch.Tensor: Cross-correlation tensor of shape (batch, channels choose 2, max_lag*2+1).
        """
        assert X.ndim == 3 or X.ndim == 4, "Input tensors must have 3 or 4 dimensions."

        if X.ndim == 3:
            X = X.unsqueeze(0)

        n_blocks = X.shape[-1]
        n_combinations = int(scipy.special.comb(X.shape[1], 2))

        if self.R_buffer is None or reset:
            self.R_buffer = torch.zeros(
                (X.shape[0], n_combinations, X.shape[2]),
                dtype=X.dtype,
                device=X.device,
            )

            n_blocks -= 1
            buffer_init = True
        else:
            buffer_init = False

        weights = self.beta * (1 - self.beta) ** torch.arange(
            n_blocks - 1, -1, -1, dtype=torch.float64, device=X.device
        )
        out_buffer = torch.zeros(
            (X.shape[0], n_combinations, self.max_lag * 2 + 1),
            device=X.device,
        )

        idx = 0
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                R = X[:, i] * torch.conj(X[:, j])

                if buffer_init:
                    self.R_buffer[:, idx] = R[..., 0]
                    R = R[..., 1:]

                R = torch.sum(weights * R, dim=-1)
                R += (1 - self.beta) ** n_blocks * self.R_buffer[:, idx]
                self.R_buffer[:, idx] = R

                if self.processor == "phat":
                    R /= torch.abs(R) + self.regularization

                r = torch.fft.irfft(R, n=self.nfft)
                r = torch.roll(r, self.max_lag)[..., 0 : self.max_lag * 2 + 1]
                out_buffer[:, idx] = torch.nan_to_num(r, nan=0.0)

                idx += 1

        return out_buffer
