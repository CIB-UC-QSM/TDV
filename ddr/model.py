import torch
import torch.utils.checkpoint as cp
import numpy as np

from .tdv import TDV

class Dataterm(torch.nn.Module):
    """
    Basic dataterm function
    """

    def __init__(self, config):
        super(Dataterm, self).__init__()

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def energy(self, *args, **kwargs):
        raise NotImplementedError

    def prox(self, x, *args, **kwargs):
        raise NotImplementedError

    def grad(self, x, *args, **kwargs):
        raise NotImplementedError


class L2DenoiseDataterm(Dataterm):
    def __init__(self, config):
        super(L2DenoiseDataterm, self).__init__(config)

    def energy(self, x, z, **kwargs):
        return 0.5*(x-z)**2

    def prox(self, x, z, tau, **kwargs):
        return (x + tau * z) / (1 + tau) 

    def grad(self, x, z, **kwargs):
        return x-z

class BlurDenoiseDataterm(Dataterm):
    # El término de energía sería 0.5 * ||A(x) - z||^2
    # Por ende, el gradiente con respecto a x es: A^T (A(x) - z)
    
    def __init__(self, config):
        super(BlurDenoiseDataterm, self).__init__(config)
        # 1. Definimos una matriz de 3x3 llena de unos y dividida por 9.
        # Esto promedia cada píxel con sus vecinos (un desenfoque de caja elemental).
        # Los tamaños en PyTorch para conv2d son: (out_channels, in_channels, height, width)
        self.kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32).cuda() / 9.0

    def grad(self, x, z, **kwargs):
        # 2. Identificamos si la imagen viene en escala de grises (1 canal) o RGB (3 canales)
        channels = x.shape[1]
        # Repetimos el filtro para que actúe en cada canal de forma independiente
        weight = self.kernel.repeat(channels, 1, 1, 1)
        
        # 3. OPERADOR FORWARD: A(x)
        # Aplicamos la convolución 2D. 'padding=1' asegura que la imagen resultante 
        # tenga las mismas dimensiones de alto y ancho que la original.
        # 'groups=channels' procesa cada canal (R, G, B) de forma aislada.
        Ax = torch.nn.functional.conv2d(x, weight, padding=1, groups=channels)
        
        # 4. CÁLCULO DEL RESIDUO: (Ax - z)
        # Evaluamos qué tan diferente es nuestra estimación borrosa frente a la medición real distorsionada.
        residual = Ax - z
        
        # 5. OPERADOR ADJUNTO: A^T aplicado al residuo
        # En álgebra lineal, el adjunto de una convolución espacial es convolucionar con el filtro invertido.
        # Dado que nuestro kernel de promedio es perfectamente simétrico, convolucionar de forma normal 
        # con el mismo filtro equivale exactamente a aplicar el operador adjunto.
        At_residual = torch.nn.functional.conv2d(residual, weight, padding=1, groups=channels)

        return At_residual.contiguous()

class QSMDataterm(Dataterm):
    """
    QSM data fidelity term: ½‖W(Dχ - φ)‖²
    
    Operates in 2D (per-slice) for use inside VNet.
    D is the dipole kernel in k-space (Fourier domain), real and symmetric.
    W is an optional magnitude weight (spatial domain).
    
    Config keys:
        dipole_kernel (np.ndarray): 2D dipole kernel in k-space
        weight (np.ndarray, optional): magnitude weight W (not W²)
        use_prox (bool): required by VNet, determines prox vs grad usage
    """

    def __init__(self, config):
        super(QSMDataterm, self).__init__(config)
        # Register as buffers so they follow model.to(device) automatically
        D = torch.from_numpy(config['dipole_kernel'].astype(np.float32))
        self.register_buffer('D', D)
        self.register_buffer('D2', D * D)  # |D|² precomputed for prox
        
        # Optional magnitude weighting
        if 'weight' in config and config['weight'] is not None:
            W = torch.from_numpy(config['weight'].astype(np.float32))
            self.register_buffer('W2', W * W)  # store W²
        else:
            self.W2 = None
    
    def _forward_op(self, x, D):
        """Forward model A(x) = F⁻¹{D · F{x}}
        Applies dipole convolution in Fourier domain.
        Handles (B, C, H, W) or (B, C, Z, H, W) tensors."""
        # Use fftn over all dimensions starting from the spatial dimensions (index 2 onwards)
        dims = tuple(range(2, x.dim()))
        return torch.fft.ifftn(D * torch.fft.fftn(x, dim=dims), dim=dims).real.contiguous()
    
    def _adjoint_op(self, x, D):
        """Adjoint A^T(x). For QSM, D is real and symmetric in k-space,
        so A^T = A (self-adjoint)."""
        return self._forward_op(x, D)
    
    def energy(self, x, z, kernel=None, weight=None):
        """Per-pixel energy: ½·W²·(Ax - z)²
        Used for monitoring convergence and training loss."""
        D = kernel.unsqueeze(1) if kernel is not None else self.D
        residual = self._forward_op(x, D) - z
        W2 = (weight * weight) if weight is not None else self.W2
        if W2 is not None:
            return 0.5 * W2 * residual ** 2
        return 0.5 * residual ** 2
    
    def grad(self, x, z, kernel=None, weight=None):
        """Gradient: ∇_x [½‖W(Ax-z)‖²] = A^T · W² · (Ax - z)"""
        D = kernel.unsqueeze(1) if kernel is not None else self.D
        residual = self._forward_op(x, D) - z
        W2 = (weight * weight) if weight is not None else self.W2
        if W2 is not None:
            residual = W2 * residual
        return self._adjoint_op(residual, D)
    
    def prox(self, x, z, tau, kernel=None, weight=None):
        """Proximal operator (without weights):
        argmin_u  ½‖u - x‖² + τ · ½‖Du - z‖²"""
        D = kernel.unsqueeze(1) if kernel is not None else self.D
        D2 = D * D
        dims = tuple(range(2, x.dim()))
        X_k = torch.fft.fftn(x, dim=dims)
        Z_k = torch.fft.fftn(z, dim=dims)
        U_k = (X_k + tau * D * Z_k) / (1 + tau * D2)
        return torch.fft.ifftn(U_k, dim=dims).real.contiguous()

class VNet(torch.nn.Module):
    """
    Variational Network
    """

    def __init__(self, config, efficient=False):
        super(VNet, self).__init__()

        self.efficient = efficient
        
        self.S = config['S']

        # setup the stopping time
        if config['T_mode'] == 'fixed':
            self.register_buffer('T', torch.tensor(config['T']['init']))
        elif config['T_mode'] == 'learned':
            self.T = torch.nn.Parameter(torch.Tensor(1))
            self.reset_scalar(self.T, **config["T"])
            self.T.L_init = 1e+3
        else:
            raise RuntimeError('T_mode unknown!')

        if config['lambda_mode'] == 'fixed':
            self.register_buffer('lmbda', torch.tensor(config['lambda']['init']))
        elif config['lambda_mode'] == 'learned':
            self.lmbda = torch.nn.Parameter(torch.Tensor(1))
            self.reset_scalar(self.lmbda, **config["lambda"])
            self.lmbda.L_init = 1e+3
        else:
            raise RuntimeError('lambda_mode unknown!')

        # setup the regularization
        R_types = {
            'tdv': TDV,
        }
        self.R = R_types[config['R']['type']](config['R']['config'])

        # setup the dataterm
        self.use_prox = config['D']['config']['use_prox']
        D_types = {
            'denoise': L2DenoiseDataterm,
            'blur': BlurDenoiseDataterm,
            'qsm': QSMDataterm,
        }
        self.D = D_types[config['D']['type']](config['D']['config'])

    def reset_scalar(self, scalar, init=1., min=0, max=1000):
        scalar.data = torch.tensor(init, dtype=scalar.dtype)
        # add a positivity constraint
        scalar.proj = lambda: scalar.data.clamp_(min, max)

    def forward(self, x, z, kernel=None, weight=None, get_grad_R=False):

        x_all = x.new_empty((self.S+1,*x.shape))
        x_all[0] = x
        if get_grad_R:
            grad_R_all = x.new_empty((self.S, *x.shape))

        # define the step size
        tau = self.T / self.S
        for s in range(1,self.S+1):
            # compute a single step
            
            # CRITICO: Custom CUDA operators (optoth en TDV) exigen contiguidad absoluta.
            # Asegurar que x y sus gradientes son siempre contiguos en memoria.
            if not x.is_contiguous():
                x = x.contiguous()
                
            if x.dim() == 5:
                # 3D Volume processing (B, C, Z, H, W) -> Extract 2.5D Triplets for TDV
                B, C, Z, H, W = x.shape
                # Pad Z by 1 on each side
                x_pad = torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 1), mode='replicate')
                # Unfold to triplets. unfold(2) -> (B, C, Z, H, W, 3). squeeze C -> (B, Z, H, W, 3).
                triplets = x_pad.squeeze(1).unfold(1, 3, 1).permute(0, 1, 4, 2, 3).contiguous().view(B*Z, 3, H, W)
                
                if self.efficient and triplets.requires_grad:
                    grad_triplets = cp.checkpoint(self.R.grad, triplets, use_reentrant=False)
                else:
                    grad_triplets = self.R.grad(triplets)
                    
                grad_triplets = grad_triplets.view(B, Z, 3, H, W)
                grad_R_pad = torch.zeros_like(x_pad)
                for i in range(3):
                    grad_R_pad[:, 0, i:i+Z, :, :] += grad_triplets[:, :, i, :, :]
                grad_R = grad_R_pad[:, :, 1:-1, :, :] / 3.0
            else:
                if self.efficient and x.requires_grad:
                    grad_R = cp.checkpoint(self.R.grad, x, use_reentrant=False)
                else:
                    grad_R = self.R.grad(x)
                
            if not grad_R.is_contiguous():
                grad_R = grad_R.contiguous()

            if self.use_prox: #se usa el operador proximal, solo para L2 default
                x = self.D.prox(x - tau * grad_R, z, self.lmbda / self.S, kernel=kernel, weight=weight)
            else: #siempre usaremos este else (desactivamos operador proximal) en los ejemplos posteriores
                x = x - tau * grad_R - self.lmbda/self.S * self.D.grad(x, z, kernel=kernel, weight=weight)
            
            if get_grad_R:
                grad_R_all[s-1] = grad_R
            x_all[s] = x
        
        if get_grad_R:
            return x_all, grad_R_all
        else:
            return x_all

    def set_end(self, s):
        assert 0 < s
        self.S = s

    def extra_repr(self):
        s = "S={S}"
        return s.format(**self.__dict__)
