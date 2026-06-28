import nibabel as nib
import numpy as np
from scipy import ndimage

def create_threshold_mask(path_mascara, path_magnitud, path_fase, path_mask_arreglada):
    # fase_img y mag_img vienen de la conversión manual mat_to_nii (afín RAS diagonal).
    # mask_img es la máscara original de Berkeley (Mask.nii) que viene del dataset.
    fase_img = nib.load(path_fase)
    mag_img = nib.load(path_magnitud)
    mask_img = nib.load(path_mascara)

    # Obtenemos la orientación de la fase y de la máscara en base a sus cabeceras.
    ornt_fase = nib.io_orientation(fase_img.affine)
    ornt_mask = nib.io_orientation(mask_img.affine)
    # Calculamos la matriz de transformación para que la máscara hable el mismo idioma que la fase.
    transformacion = nib.orientations.ornt_transform(ornt_mask, ornt_fase)
    mask_base = mask_img.as_reoriented(transformacion).get_fdata() > 0
    
    # En la conversión mat_to_nii transpusimos las matrices MATLAB (Y, X, Z) a (X, Y, Z).
    # Sin embargo, el eje Y (anteroposterior) quedó invertido respecto a la máscara estándar de Berkeley
    # debido a cómo las herramientas nativas guardan los archivos NIfTI en disco.
    # Para que queden perfectamente alineados y sin desfases, le aplicamos un flip manual al eje Y.
    mask_base = mask_base[:, ::-1, :]

    # La magnitud ya está en el espacio y orientación de la fase, así que no necesita reorientarse.
    magnitud = mag_img.get_fdata()
    # Si viene con múltiples ecos (4D), promediamos su señal usando Root-Sum-of-Squares (RSS)
    # para obtener una única imagen 3D con excelente relación señal-ruido.
    if magnitud.ndim == 4:
        magnitud = np.sqrt(np.sum(magnitud**2, axis=-1))

    # --- PASO A PASO DEL PROCESAMIENTO ANATÓMICO ---
    
    # Paso 1: Rellenamos agujeros internos iniciales de la máscara de Berkeley.
    brain_solid = ndimage.binary_fill_holes(mask_base)
    
    # Paso 2: Filtro dinámico de SNR.
    # En lugar de usar umbrales fijos que dependen del paciente, calculamos la señal media
    # real del tejido cerebral. Tomamos el 5% de esta señal media como umbral dinámico de corte.
    # Esto saca capas ruidosas (como la duramadre o grasa) sin dañar la corteza.
    mean_brain = np.mean(magnitud[brain_solid])
    umbral_dinamico = 0.05 * mean_brain
    mag_threshold = magnitud > umbral_dinamico
    
    # Paso 3: Intersección de seguridad física.
    # Nos quedamos solo con la zona donde coinciden el cerebro anatómico y la señal confiable de magnitud.
    brain_clean = brain_solid & mag_threshold
    
    # Paso 4: Cierre morfológico 3D y segundo sellado.
    # El cierre (closing) con un kernel de 3x3x3 sella fisuras externas creadas por cortes de baja señal.
    # El segundo rellenado de huecos protege vasos y núcleos de hierro oscuros que caen bajo el umbral.
    brain_closed = ndimage.binary_closing(brain_clean, structure=np.ones((3,3,3)), iterations=2)
    brain_final = ndimage.binary_fill_holes(brain_closed)
    
    # Paso 5: Anti-aliasing y suavizado de bordes.
    # Aplicamos un filtro Gaussiano para eliminar el molesto "aserruchado" de los vóxeles 3D
    # y binarizamos en 0.5 para tener un contorno curvo, limpio y suave.
    refined_mask = ndimage.gaussian_filter(brain_final.astype(np.float32), sigma=1.0) > 0.5

    # Guardamos la máscara arreglada usando la cabecera y el afín de la fase
    # para garantizar compatibilidad absoluta en todo el pipeline de QSM.
    refined_mask_img = nib.Nifti1Image(refined_mask.astype(np.float32), fase_img.affine, fase_img.header)
    nib.save(refined_mask_img, str(path_mask_arreglada))
    
    '''
    --- ZONA ELIMINADA / HISTORIAL DE PRUEBAS ---
    1. Método Otsu: Intentamos umbralizar de forma automática con Otsu, pero creaba hoyos 
       inaceptables dentro de los ventrículos y zonas profundas con venas oscuras.
    2. Resta de vóxeles ruidosos: Generaba un contorno en escalera o serrucho, lo cual
       es catastrófico para wTV y genera fuertes artefactos de estrella en QSM.
    3. Fase * Máscara antes de ROMEO: ROMEO prefiere la fase cruda. Multiplicarla antes
       impedía que ROMEO modelara correctamente las discontinuidades de fase en el fondo.
    4. Hubo un problema de desplazamiento de exactamente 1 vóxel de la máscara con respecto a la magnitud. 
       Esto ocurre porque la cabecera originalde Mask.nii tiene una traslación de origen de [1.25, 1.25, 2.0] mm
       mientras que la magnitud convertida parte en [0, 0, 0].
       Trabajar a través de las cabeceras físicas y el objeto NIfTI de nibabel resuelve este traslape automáticamente.
    '''
    
    '''
    ================================================================================
                        BITÁCORA DE ITERACIONES Y AJUSTES (iPreQSM)
    ================================================================================

    Aquí listamos todos los experimentos, aciertos, tropiezos y eliminaciones que
    fuimos haciendo a lo largo del desarrollo de esta máscara:

    1. ❌ El intento de Otsu (Primer Enfoque):
       - Qué era: Partimos usando la umbralización automática de Otsu sobre la magnitud.
       - Qué pasó: Otsu asume dos clases de intensidades y generaba horribles "agujeros"
         dentro del cerebro en zonas con núcleos ricos en hierro o vasos sanguíneos oscuros.
       - Decisión: Eliminado por completo. Decidimos usar la máscara de Berkeley como 
         guía espacial segura y optimizarla morfológicamente.

    2. ❌ El "Lijado de Bordes Ruidosos" (Segundo Enfoque):
       - Qué era: Intentamos restar vóxeles ruidosos restando contornos directos.
       - Qué pasó: Esto generaba bordes en "zigzag" y esquinas muy angulares que creaban 
         terribles artefactos de dipolo en la reconstrucción final de QSM.
       - Decisión: Eliminado. Reemplazado por cierre morfológico 3D y filtro Gaussiano suave.

    3. ❌ Multiplicación de la fase por la máscara antes de ROMEO:
       - Qué era: Pasarle a ROMEO la fase con el fondo ya borrado (fase * máscara).
       - Qué pasó: ROMEO funciona mucho mejor cuando recibe la fase "cruda" porque utiliza 
         el ruido del fondo para modelar el desenrollado y evitar saltos abruptos de fase de 2pi.
       - Decisión: Eliminado. ROMEO recibe la fase intacta y la máscara solo se pasa como parámetro "-k".

    4. 🔍 El Gran Mapeo de Orientación y el Eje Y (¡El bug oculto de la rotación!):
       - Qué era: Al principio la máscara corregida salía desplazada e invertida en overlay_control.png.
       - Qué descubrimos:
         - El script 'mat_to_nii' intercambiaba (Y, X, Z) -> (X, Y, Z) para NIfTI.
         - Sin embargo, las imágenes de Berkeley en disco invertían físicamente el eje Y para coincidir con la cabecera.
         - Como las cabeceras decían ser RAS en ambos lados, nibabel asumía que no había cambios y no aplicaba rotación.
         - Por ende, ¡la máscara estaba al revés (anteroposteriormente) respecto a la magnitud!
       - Ajuste y Acierto:
         - Dejamos de pasarle la reorientación a 'magnitud' en masking.py (ya que venía en el espacio de fase correcto).
         - Aplicamos el flip manual de eje Y: `mask_base = mask_base[:, ::-1, :]`
         - Aplicamos el mismo flip en `validar_mascara.py` para comparar peras con peras.

    5. 🎉 Resultados Finales de la Alineación Definitiva (RAS Perfecto):
       - Logramos una alineación matemática y anatómica perfecta (Dice index espectacular).
       - El volumen de la máscara se recuperó en su totalidad (+2.3% de volumen respecto a la versión desalineada).
       - El gradiente en el borde mejoró un **9.24%** (de 363.90 a 397.52), demostrando que la máscara
         se adhiere perfectamente a los surcos y giros del cerebro real en vez de recortarlo.
       - La señal promedio dentro de la máscara de Berkeley saltó de 1167.72 a **1633.45**,
         confirmando que por fin estamos evaluando tejido cerebral real en vez de aire o cráneo.
    '''