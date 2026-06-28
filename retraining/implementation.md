# QSM-Specific TDV Retraining Plan

Con el debuggeo del pipeline base finalizado y habiendo demostrado que la red pre-entrenada con imágenes naturales (BSDS400) es incompatible con la física de la anatomía cerebral en QSM, estamos listos para iniciar la Fase de Reentrenamiento.

## Objetivo
Ejecutar el reentrenamiento específico para QSM de la red VNet, congelando los pesos del término de datos y entrenando únicamente el regularizador $R$, asegurando que las no-linealidades aprendan a preservar venas y tejido cerebral real.

## Tareas Propuestas

### 1. Auditoría y Mejora del Dataset (`dataset_qsm.py`)
- Revisar que la clase `QSMDataset` implementada en la carpeta `retraining/` construya correctamente los tensores tridimensionales (k-1, k, k+1) usando `stride-1` para evitar los bugs de borde descubiertos hoy.
- **[CRÍTICO]** Integrar el *fix* de la normalización del peso de magnitud (`weight_np /= weight_np[_brain_mask].mean()`) directamente en el pipeline de datos del entrenamiento, para que la red no colapse durante el ADMM *unrolled*.

### 2. Auditoría del Ciclo de Entrenamiento (`train_qsm.py`)
- Verificar la correcta carga de los pesos pre-entrenados y la proyección/congelamiento (`p.proj()`) de los parámetros del término de datos, para que la red no se atasque.
- Asegurar que la función de pérdida (Loss) compare la salida final del ADMM reconstruido contra el *Ground Truth* (`chi_gt`).
- Configurar el escalado `alpha` de entrenamiento en un rango robusto.

### 3. Ejecución de Prueba (Dry-Run)
- Correr 1 época de entrenamiento con un `batch_size` ajustado para utilizar eficientemente los 12GB de tu GPU RTX 5070 Ti, monitoreando el consumo de VRAM y asegurando que la Loss empiece a bajar.

## User Review Required

> [!IMPORTANT]
> Revisa la última imagen generada (`wTDV_QSM_torch.png` con `alpha=0.04`). Esta es la verdadera línea base matemática (la red en modo lineal puro, sin arruinar el centro del cerebro). Si estás de acuerdo en que esta es la máxima calidad posible sin reentrenar, aprueba este plan haciendo click en **Proceed** para que audite los scripts de la otra IA y lancemos el entrenamiento.
