# Instrucciones para Reentrenamiento de TDV-VNet para QSM

**Contexto General:**
Actúa como un ingeniero experto en Deep Learning y Resonancia Magnética (QSM). El objetivo es implementar el reentrenamiento de una red Variational Network (TDV-VNet) adaptada específicamente para el problema inverso de QSM, basándote en la teoría descrita previamente y en el código actual del repositorio.

Actualmente, el modelo TDV realiza un *denoising* genérico en imágenes naturales. Queremos reentrenarlo de manera *unrolled* de extremo a extremo para que el regularizador aprenda las características y artefactos (ej. *streaking*) específicos del mapeo de susceptibilidad magnética ($\chi$).

**Archivos a implementar:**
Debes escribir dos archivos completos en PyTorch, sin omitir lógica, asumiendo buenas prácticas de optimización de memoria (uso de GPU) y sin usar placeholders genéricos.

### 1. `dataset_qsm.py`
- Crea la clase `QSMDataset(Dataset)` nativa de PyTorch.
- **Carga de Datos**: Debe leer volúmenes de entrenamiento (idealmente desde archivos `.mat` o `.h5` pre-procesados) que contengan:
  1. Fase medida (input).
  2. Susceptibilidad real de referencia (`chi_gt`).
  3. Pesos de magnitud (`W`).
  4. Kernel dipolar tridimensional en k-space (puede ser global o por volumen).
- **Modo 2.5D (3 Canales)**: Para ser compatible con el regularizador actual, la red espera 3 canales espaciales. El dataset debe extraer tripletes de slices contiguos `(z-1, z, z+1)` a lo largo del eje axial y agruparlos como canales `(3, H, W)`. Esto aplica para la fase, `W` y `chi_gt` (aunque el cálculo de la pérdida puede enfocarse solo en el slice central).
- **Salida**: Retorna un diccionario con los tensores: `{'phase': phase, 'chi_gt': chi_gt, 'weight': W, 'kernel': kernel}`.
- *Tip*: Considera si el dataset aplica normalización a la fase o si esto se delega al pipeline de la red.

### 2. `train_qsm.py`
- **Configuración y Argumentos**: Usa `argparse` para definir hiperparámetros: `lr` (típicamente bajo para unrolling, e.g., 1e-4), `epochs`, `batch_size`, y las rutas de los datos y checkpoints.
- **Configuración del Dataterm (`model.py`)**: 
  - La VNet debe configurarse modificando su diccionario interno: `config['D']['type'] = 'qsm'`.
  - Dado que usamos los pesos de magnitud $W$ en el término de datos (lo que hace que no tenga solución de forma cerrada simple en Fourier), debemos usar el descenso de gradiente implementado. Para esto, configura `config['D']['config']['use_prox'] = False`.
- **Transfer Learning Correcto (Crucial)**: 
  - Carga el checkpoint base de denoising genérico (`tdv3-3-25-f32-color.pth`).
  - Queremos aprovechar los filtros de las convoluciones, pero no los parámetros de escala del Dataterm o multiplicadores del ADMM.
  - Extrae y carga **únicamente** los pesos correspondientes al módulo del regularizador TDV (es decir, `vnet.R.state_dict()`).
  - Los parámetros escalares o variables iterativas de la red (como el factor $\lambda$ del Dataterm) deben ser inicializados a valores por defecto estables para evitar que la red comience atascada en un mínimo local de denoising.
- **Bucle de Entrenamiento (Restricciones Matemáticas)**:
  - En el forward pass, inicializa las variables necesarias en el dataterm e inyecta la fase y los pesos.
  - Calcula el MSE Loss entre la salida final de la VNet y el target `chi_gt`.
  - Tras calcular los gradientes con `loss.backward()` y actualizar los pesos con `optimizer.step()`, **DEBES aplicar las restricciones del TDV**.
  - **Iteración de Proyección**: Itera sobre los parámetros del modelo. Si un parámetro (ej. filtros de convolución) tiene el método `p.proj()`, debes ejecutarlo.
    ```python
    for p in vn.parameters():
        if hasattr(p, 'proj'):
            p.proj()
    ```
    *Nota: Esto asegura que los filtros del TDV mantengan media cero y normas espectrales acotadas, según la implementación original en `ddr/conv.py`.*
- **Checkpoints**: Guarda el estado de la red (incluyendo `config`) por cada época o cuando la pérdida de validación mejore.

**Directriz Final**: Antes de escribir el código, analiza detenidamente cómo `QSMDataterm` se instancia en `model.py` (métodos `energy`, `grad`) para asegurar que el pipeline de `train_qsm.py` inyecte correctamente el `kernel`, `phase`, y `weight` en el loop iterativo de la Variational Network.
