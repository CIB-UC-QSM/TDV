import torch
import torch.utils.checkpoint as cp

from ddr import TDV

class Dataterm(torch.nn.Module):
    """
    Basic dataterm function
    """

    def __init__(self, config):
        super(Dataterm, self).__init__()

    def forward(self, x, *args):
        raise NotImplementedError

    def energy(self):
        raise NotImplementedError

    def prox(self, x, *args):
        raise NotImplementedError

    def grad(self, x, *args):
        raise NotImplementedError


class L2DenoiseDataterm(Dataterm):
    def __init__(self, config):
        super(L2DenoiseDataterm, self).__init__(config)

    def energy(self, x, z):
        return 0.5*(x-z)**2

    def prox(self, x, z, tau):
        return (x + tau * z) / (1 + tau) 

    def grad(self, x, z):
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

    def grad(self, x, z):
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

        return At_residual

class QSMDataterm(Dataterm):
    # partimos por implementar QSM en 2D, osea una slice. Considerando que una imagen a color es soportada,
    # y en QSM solo tenemos valores entre -1 y 1, habra que ver como funciona trasladar esas capacidades
    # a esta clase (o si es mejor crear otra). 

    def __init__(self, config):
        super(QSMDataterm, self).__init__(config)
        # El constructor guardará el kernel dipolar 2D precalculado.
        self.D = torch.from_numpy(config['dipole_kernel']).cuda().float()

    def grad(self, x, z):
        # x: estimación actual de la susceptibilidad (Real-space)
        # z: mapa de fase local medido (Real-space)
        
        # 1. Forward Operator A(x): Convolución dipolar en el dominio de Fourier
        # Pasamos x al K-space usando FFT2 centrado
        X_k = torch.fft.fft2(x)
        
        # Multiplicamos elemento a elemento por el Kernel Dipolar D
        # Nota: Asegúrate de manejar las dimensiones de Batch/Canales si es necesario
        AX_k = X_k * self.D
        
        # Devolvemos al espacio real para calcular el residuo físico
        Ax = torch.fft.ifft2(AX_k).real
        
        # 2. Calcular el residuo: A(x) - z
        residual = Ax - z
        
        # 3. Adjoint Operator A^T aplicado al residuo:
        # En QSM, el kernel dipolar en Fourier es puramente real y simétrico (D = D^*),
        # por lo que el operador adjunto es exactamente igual al operador forward.
        RES_k = torch.fft.fft2(residual)
        AT_RES_k = RES_k * self.D
        
        # Retornamos el gradiente final al espacio real
        grad_data = torch.fft.ifft2(AT_RES_k).real
        
        return grad_data

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

    def forward(self, x, z, get_grad_R=False):

        x_all = x.new_empty((self.S+1,*x.shape))
        x_all[0] = x
        if get_grad_R:
            grad_R_all = x.new_empty((self.S, *x.shape))

        # define the step size
        tau = self.T / self.S
        for s in range(1,self.S+1):
            # compute a single step
            if self.efficient and x.requires_grad:
                grad_R = cp.checkpoint(self.R.grad, x)
            else:
                grad_R = self.R.grad(x)
            if self.use_prox: #se usa el operador proximal, solo para L2 default
                x = self.D.prox(x - tau * grad_R, z, self.lmbda / self.S)
            else: #siempre usaremos este else (desactivamos operador proximal) en los ejemplos posteriores
                x = x - tau * grad_R - self.lmbda/self.S * self.D.grad(x, z)
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
