# NanoParticle Analyzer

Clasificador y analizador de nanopartículas a partir de imágenes de microscopía electrónica de transmisión (TEM). Este proyecto forma parte de un Trabajo de Fin de Grado.

## Descripción

Herramienta de visión por computador que utiliza deep learning para segmentar, clasificar y medir nanopartículas en imágenes TEM. El flujo de trabajo completo es:

1. **Carga** de una imagen `.tif` mediante interfaz gráfica (PyQt5).
2. **Preprocesado** opcional (denoising, contraste, eliminación automática de la escala).
3. **Segmentación** de nanopartículas con una red U-Net entrenada.
4. **Post-procesado** con watershed para separar partículas solapadas.
5. **Clasificación** de cada partícula en 5 formas geométricas mediante una red densa.
6. **Medición** del área de cada partícula en unidades reales usando la barra de escala.
7. **Visualización** con mapa de color y histogramas por clase.

## Clases de clasificación

| Forma        | Color |
|--------------|-------|
| Bipirámides  | Rojo  |
| Hexágonos    | Verde |
| Círculos     | Azul  |
| Cuadrados    | Rosa  |
| Rectángulos  | Cyan  |

## Requisitos

- Python 3.10
- TensorFlow 2.11.0
- OpenCV 4.7.0
- PyQt5 5.15.9
- Matplotlib 3.7.1
- NumPy 1.23.5

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

```bash
python GUI.py
```

### Funcionalidades de la interfaz

- **Cargar imagen**: Botón *Browse* para seleccionar archivos `.tif`.
- **Escala manual**: *"Introducir Escala"* → indicar el valor real y dibujar una línea en la imagen.
- **Escala automática**: Activarla en *Opciones* para leer el valor directamente de la barra de escala.
- **Tamaño de entrada**: Ajustable desde 256×256 hasta 1984×1984 (pasos de 64 píxeles). Tamaños menores dan más error en las medidas; tamaños mayores pueden clasificar píxeles de fondo como partículas.
- **Preprocesado**: Denoising (Gaussian blur) y aumento de contraste (CLAHE) configurables en *Opciones*.
- **Watershed**: Aplicado por defecto para separar partículas que se tocan entre sí.

### Flujo de trabajo

1. Seleccionar una imagen y ajustar escala/tamaño/opciones.
2. Pulsar *"Segmentar"* y esperar a que se genere la máscara.
3. En la ventana de segmentación, pulsar *"Iniciar Clasificación"*.
4. Tras el procesado, se mostrará un histograma con la distribución de tamaños por clase.

## Estructura del proyecto

```
├── GUI.py                         # Interfaz gráfica de usuario (PyQt5)
├── SegmentationModule.py          # Módulo de segmentación, clasificación y análisis
├── UNet.ipynb                     # Entrenamiento de modelos de segmentación (U-Net)
├── Clasificacion.ipynb            # Entrenamiento del clasificador de formas
├── gui.ui                         # Diseño Qt Designer de la ventana principal
├── segmented_window.ui            # Diseño Qt Designer de la ventana de segmentación
├── histogram.ui                   # Diseño Qt Designer de la ventana de histograma
├── Dockerfile                     # Contenedor para ejecutar la aplicación
├── requirements.txt               # Dependencias de Python
├── Memoria.pdf                    # Documento completo del TFG
├── scale.jpg                      # Imagen de referencia para la escala
└── README.md                      # Este archivo
```

## Entrenamiento de modelos

### Segmentación (`UNet.ipynb`)

- Dataset: [TEM-Nano-Particle-Cell-Dataset](https://github.com/ivanv99/TEM-Nano-Particle-Cell-Dataset) (público).
- Arquitectura: U-Net con codificador (64→128→256→512), bottleneck (1024) y decodificador con conexiones skip.
- Aumento de datos: rotación, desplazamiento, shear, zoom, flip horizontal.
- Modelo generado: `modeloPropio.h5`.

### Clasificación (`Clasificacion.ipynb`)

- Dataset: Imágenes propias del CIQUS (no incluidas en el repositorio por ser propiedad del centro).
- Arquitectura: Red densa de 6 capas (2048→512→256→256→128→128→64→5) con softmax.
- Aumento de datos: rotación y flip aleatorios (mínimo 140 muestras por clase).
- Modelo generado: `modelo_denso_prueba_variacion_pardo.h5`.

> **Nota:** Los modelos entrenados no están incluidos en el repositorio debido a su tamaño. Deben generarse ejecutando los notebooks correspondientes.

## Docker

### Windows (con VcXsrv)

1. Descargar e instalar [VcXsrv](https://sourceforge.net/projects/vcxsrv/files/latest/download).
2. Construir y ejecutar el contenedor:

```bash
docker build -t nanoparticle-analyzer .
docker run nanoparticle-analyzer
```

[Tutorial en vídeo](https://www.youtube.com/watch?v=SZvwDqSPuTU)

### Linux

```bash
docker build -t nanoparticle-analyzer .
docker run --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix nanoparticle-analyzer
```

## Limitaciones conocidas

- El tamaño de entrada influye en la precisión de las medidas: menor tamaño → más error; mayor tamaño → más falsos positivos.
- El watershed se aplica por defecto para mitigar partículas solapadas, decisión condicionada por la limitación en los datos de entrenamiento.
- El módulo de clasificación requiere imágenes de entrenamiento propiedad del CIQUS, no disponibles públicamente.

## Autor

Trabajo de Fin de Grado — Universidad de Santiago de Compostela

## Licencia

Este proyecto es de carácter académico. Los datos de entrenamiento del clasificador son propiedad del CIQUS.
