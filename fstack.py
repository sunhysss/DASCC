import torch

def hilbert(xr, n=None, dim=-1):
    """
    Compute the discrete-time analytic signal via Hilbert transform.

    Args:
        xr (torch.Tensor): Input real-valued signal.
        n (int, optional): Number of points for the FFT. If None, uses the size of the input along the specified dimension.
        dim (int): Dimension along which to compute the Hilbert transform.

    Returns:
        torch.Tensor: Analytic signal (complex-valued).
    """
    # Ensure input is real-valued
    xr = torch.as_tensor(xr)
    if not torch.is_complex(xr):
        xr = xr.real

    # Determine the number of points for FFT
    if n is None:
        n = xr.size(dim)

    # Perform FFT along the specified dimension
    x_fft = torch.fft.fft(xr, n=n, dim=dim)

    # Create a mask to zero out the negative frequencies
    mask = torch.zeros_like(x_fft, dtype=torch.bool)
    half_n = (n // 2) + 1

    # Set the positive frequencies (and optionally Nyquist frequency)
    idx_positive = torch.arange(1, half_n, device=xr.device)
    mask.index_fill_(dim, idx_positive, True)

    # Apply the mask and scale the positive frequencies
    x_fft_masked = x_fft.clone()
    x_fft_masked[mask] *= 2
    x_fft_masked[~mask] = 0

    # Perform IFFT to get the analytic signal
    analytic_signal = torch.fft.ifft(x_fft_masked, n=n, dim=dim)

    return analytic_signal


def gather_instant_phase_estimator(gather):
    """
    Compute the instantaneous phase estimator for the input signal.

    Args:
        gather (torch.Tensor): Input data (real-valued).

    Returns:
        torch.Tensor: Instantaneous phase estimator (complex-valued).
    """
    # Compute the Hilbert transform to get the analytic signal
    analytic_signal = hilbert(gather, dim=-1)

    # Compute the instantaneous phase
    phi_mat = torch.angle(analytic_signal)

    # Compute the phase shift matrix
    phaseshift_mat = torch.exp(1j * phi_mat)

    return phaseshift_mat


def Fstack(matrix_cell, stack_flag=0, nu=1, dim=None, device='cuda'):
    """
    Stack input data using mean stacking, phase-weighted stacking, or semblance-weighted stacking.

    Args:
        matrix_cell (torch.Tensor): Input data for stacking. Can be 2D/3D tensor.
        stack_flag (int): 
            - 0 for mean stack
            - 1 for phase-weighted stack
            - 2 for semblance-weighted stack
        nu (float): Power term used for phase-weighted stack. Larger nu increases the influence of phase coherence.
        dim (int): Dimension along which to stack the data. If None, the last dimension is used.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Stacked matrix.
    """
    # Ensure input is a PyTorch tensor and move it to the specified device
    matrix_cell = torch.as_tensor(matrix_cell, device=device)
    
    # Determine the stacking dimension
    if dim is None:
        dim = matrix_cell.dim() - 1
    
    # Initialize variables
    n_stack = matrix_cell.size(dim)
    mstack = torch.zeros_like(matrix_cell.select(dim, 0))
    coherency_sum = torch.zeros_like(mstack)

    # Iterate over the stacking dimension
    for i in range(n_stack):
        data_in = matrix_cell.select(dim, i)
        
        if stack_flag == 0:
            coherency_sum += torch.zeros_like(data_in)
        elif stack_flag == 1:
            coherency_sum += gather_instant_phase_estimator(data_in).real
        elif stack_flag == 2:
            coherency_sum += data_in ** 2
        else:
            raise ValueError("Invalid stack_flag. Must be 0, 1, or 2.")
        
        mstack += data_in

    # Compute the mean stack
    mstack /= n_stack

    # Apply weighting if necessary
    if stack_flag > 0:
        if stack_flag in {1}:
            phase_weight = torch.abs(coherency_sum / n_stack) ** nu
        elif stack_flag == 2:
            phase_weight = mstack ** 2 / coherency_sum
        data_stack = mstack * phase_weight
    else:
        data_stack = mstack

    return data_stack