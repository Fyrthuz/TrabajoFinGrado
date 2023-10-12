# TFG
Clasificador de nanoparticulas

# ANOTACIONES
Los modelos no están subidos debido a que son muy pesados para subirlos a Github.

# Versiones
Python 3.10

# Librerias
numpy

tensorflow

open-cv2

PySimpleGUI

matplotlib

# Instalacion librerias
pip install opencv-python

pip install tensorflow

pip install numpy

pip install PySimpleGUI

pip install matplotlib

# Ejecucion
python GUI.py

# Aclaraciones
Para meter la escala manualmente clickar en "Introducir Escala" indicar el valor
y dibujar una linea en la imagen.

Tambien se puede aumentar el contraste y el ruido de la imagen para mejorar el resultado de la prediccion en algunos casos.

Se puede elegir el tamano pero cuanto menos tamano mas error en las medidas y cuanto mayor tamano se clasificaran pixeles como particulas cuando no lo son.

Tambien se aplica watershed por defecto para minimizar el hecho de las particulas que se tocan entre si, esta decision viene de la mano de la limitacion en los datos de entrenamiento.

Esperar a que se segmente y veras el resultado por pantalla, para continuar e iniciar la clasificacion, tras un tiempo se mostrara un histograma con los tamanos medidos

# ATENCION
para ejecutar con docker windows descargar esto
https://sourceforge.net/projects/vcxsrv/files/latest/download

Ejemplo ejecucion contenedor:
https://www.youtube.com/watch?v=SZvwDqSPuTU

docker run "nombre_contenedor"

# Archivos
Unet.ipynb ---> Hace el entrenamiento de las diferentes arquitecturas de segmentacion
Clasificacion.ipynb ---> Hace el entrenamiento de las diferentes arquitecturas de clasificacion
Memoria.pdf ---> Resumen del trabajo completo


