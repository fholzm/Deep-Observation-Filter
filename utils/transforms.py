import torch


class OverlapSave(torch.nn.Module):
    """Method for Overlap-Save transforms to/from frequency domain."""

    def __init__(self, nfft: int, hopsize: int = None, complex_input: bool = False):
        """Method for Overlap-Save transforms to/from frequency domain.

        Parameters
        ----------
        nfft : int
            FFT size
        hopsize : int, optional
            Hopsize for blocking the signals, by default nfft / 2
        complex_input : bool, optional
            Complex valued input signals, by default False
        """
        super(OverlapSave, self).__init__()
        self.nfft = nfft

        if hopsize == None:
            self.hopsize = nfft // 2
        else:
            self.hopsize = hopsize

        self.max_filter_length = nfft - hopsize + 1
        self.input_buffer = None
        self.input_buffer_gcc = None

        self.complex_input = complex_input

    def forward(self, x: torch.Tensor, to_fd: bool, reset: bool = False):
        """Transform signals to/from frequency domain for overlap-save processing

        Parameters
        ----------
        x : torch.Tensor
            Input signal
        to_fd : bool
            Flag for transformation to frequency domain or back to time domain
        reset : bool, optional
            Reset internal input buffer for tranform to frequency domain, by default False

        Returns
        -------
        torch.Tensor
            Tranformed input signal
        """
        if to_fd:
            return self.transform_to_fd(x, reset)
        else:
            return self.transform_to_td(x)

    def transform_to_fd(self, x: torch.Tensor, reset: bool = False):
        assert torch.is_complex(x) == self.complex_input, "Input must be complex valued"

        # Initialize buffer
        if self.input_buffer is None or reset:
            pad_size = list(x.shape)
            pad_size[-1] = self.nfft - self.hopsize

            self.input_buffer = torch.zeros(pad_size, device=x.device, dtype=x.dtype)

        next_input_buffer = x[..., 0 : self.input_buffer.shape[-1]]

        # Concatenate with saved buffer
        x = torch.cat([self.input_buffer, x], -1)

        self.input_buffer = next_input_buffer

        # Create overlapping frames of input signal
        x = torch.transpose(x.unfold(-1, self.nfft, self.hopsize), -1, -2)

        if self.complex_input:
            x = torch.fft.fft(x, self.nfft, dim=-2)
        else:
            x = torch.fft.rfft(x, self.nfft, dim=-2)

        return x

    def transform_to_td(self, x: torch.Tensor):
        if self.complex_input:
            x = torch.fft.ifft(x, self.nfft, dim=-2)
        else:
            x = torch.fft.irfft(x, self.nfft, dim=-2)

        return torch.transpose(x[..., -self.hopsize :, :], -1, -2).flatten(-2)

    def transform_coefficients(self, x: torch.Tensor):
        """Transform coefficients to frequency domain for overlap-add processing

        Parameters
        ----------
        x : torch.Tensor
            Filter coefficients in time domain

        Returns
        -------
        torch.Tensor
            Transformed filter coefficients
        """
        assert torch.is_complex(x) == self.complex_input, "Input must be complex valued"
        assert (
            x.shape[-1] <= self.max_filter_length
        ), f"Filter length must be equal or less than {self.max_filter_length} taps"

        if self.complex_input:
            x = torch.fft.fft(x, self.nfft)
        else:
            x = torch.fft.rfft(x, self.nfft)

        return x

    def transform_for_gcc(self, x: torch.Tensor, reset: bool = False):
        assert torch.is_complex(x) == self.complex_input, "Input must be complex valued"

        # Initialize buffer
        if self.input_buffer_gcc is None or reset:
            pad_size = list(x.shape)
            pad_size[-1] = self.nfft - self.hopsize

            self.input_buffer_gcc = torch.zeros(
                pad_size, device=x.device, dtype=x.dtype
            )

        next_input_buffer = x[..., 0 : self.input_buffer_gcc.shape[-1]]

        # Concatenate with saved buffer
        x = torch.cat([self.input_buffer_gcc, x], -1)

        self.input_buffer_gcc = next_input_buffer

        window = torch.hann_window(self.nfft, device=x.device, dtype=x.dtype)

        # Create overlapping frames of input signal
        x = torch.transpose(x.unfold(-1, self.nfft, self.hopsize) * window, -1, -2)

        if self.complex_input:
            x = torch.fft.fft(x, self.nfft, dim=-2)
        else:
            x = torch.fft.rfft(x, self.nfft, dim=-2)

        return x
