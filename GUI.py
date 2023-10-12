import sys
import typing
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QFileDialog, QLabel, QWidget, QVBoxLayout, QInputDialog
from PyQt5.QtCore import QPoint, QRect
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5 import FigureCanvasQT
import numpy as np
import cv2

import SegmentationModule as sm

#import QPainter
from PyQt5.QtGui import QPainter
#import QtCore
from PyQt5 import QtCore
#import QPen
from PyQt5.QtGui import QPen

class SegmentWindow(QDialog):

    def __init__(self,mask,image_width,ratio,segmented_class,img):
        super(SegmentWindow, self).__init__()
        loadUi('segmented_window.ui', self)
        self.mask = mask
        self.setWindowTitle("Imagen segmentada")

        self.Segmented_Module = segmented_class

        self.actual_image = img

        mask_width = self.mask.shape[1]

        area_ratio = (mask_width*mask_width)/(image_width*image_width)

        self.ratio = ratio/(area_ratio)

        
        #reshape the image to be displayed
        self.displayed_image = cv2.resize(self.mask,(768,768))

        imagen_Q = QImage(self.displayed_image, 768, 768, QImage.Format_Grayscale8)
        #convert image into a pixmap
        pixmap = QPixmap.fromImage(imagen_Q)
        self.image.setPixmap(pixmap)
        self.image.repaint()
        self.atras.clicked.connect(self.atras_clicked)
        self.botonclasificar.clicked.connect(self.botonclasificar_clicked)

    def botonclasificar_clicked(self):
        
        white_pixels,color_mask = self.Segmented_Module.separar_regiones(self.mask)

        areas = self.Segmented_Module.get_areas(white_pixels,self.ratio)

        print(self.actual_image.shape,color_mask.shape)
        plots = self.Segmented_Module.get_histogram(areas,self.actual_image,color_mask)
        plots.show()

    
    def atras_clicked(self):
        #destroy the window
        self.close()
        self.destroy()




class MainWindow(QDialog):


    def __init__(self,img_size = (1024,1024)):
        super(MainWindow, self).__init__()
        loadUi('gui.ui', self)
        self.browse.clicked.connect(self.browsefiles)
        self.origin = QPoint()
        self.end = QPoint()


        self.Segementation = sm.SegmentationModule()
        self.escala.setVisible(False)
        self.introducir_escala.setVisible(False)
        self.introducir_escala.clicked.connect(self.introducir_escala_clicked)
        #events
        self.image.mousePressEvent = self.getFirstPoint
        #clicked
        self.introducir_tam.clicked.connect(self.introducir_tam_clicked)
        self.introducir_tam.setVisible(False)
        self.tamano.setVisible(False)

        self.scale = 1
        self.ratio = 1
        self.img_size = img_size
        self.image_array = None
        self.distance = 1

        self.contrast = False

        self.watershed = False

        self.denoise = False

        self.auto_scale = False

        self.segmentar.setVisible(False)
        self.segmentar.clicked.connect(self.segmentar_clicked)

        self.opciones.setVisible(False)
        self.opciones.clicked.connect(self.opciones_clicked)

        self.setWindowTitle("NanoParticle Analyzer")

    def opciones_clicked(self):
        #Open a dialog to introduce True or False
        denoise, ok = QtWidgets.QInputDialog.getItem(self, 'Denoise', '¿Desea aplicar un filtro de denoise?:',["True","False"],1,False)
        if ok:
            if denoise == "True":
                self.denoise = True
            else:
                self.denoise = False
        else:
            return
        #Open a dialog to introduce True or False
        contrast, ok = QtWidgets.QInputDialog.getItem(self, 'Contrast', '¿Desea aplicar un filtro de contraste?:',["True","False"],1,False)
        if ok:
            if contrast == "True":
                self.contrast = True
            else:
                self.contrast = False
        else:
            return
        #Open a dialog to introduce True or False
        watershed, ok = QtWidgets.QInputDialog.getItem(self, 'Watershed', '¿Desea aplicar un filtro de watershed?:',["True","False"],1,False)
        if ok:
            if watershed == "True":
                self.watershed = True
            else:
                self.watershed = False
        else:
            return
        #Open a dialog to introduce True or False
        auto_scale, ok = QtWidgets.QInputDialog.getItem(self, 'Scale', '¿Desea leer la escala automáticamente(puede no ser correcta)?:',["True","False"],1,False)
        if ok:
            if auto_scale == "True":
                self.auto_scale = True
            else:
                self.auto_scale = False
        image = None
        if ok:
            if auto_scale == "True":
                image = self.Segementation.prepare_image_without_scale(self.filename.text(),size=(768,768),contrast=self.contrast,noise=self.denoise)[1][0]
                self.escala.setText("Escala: 1:" + str(self.Segementation.prepare_image_without_scale(self.filename.text(),size=(768,768))[0][1]))
                self.ratio = self.Segementation.prepare_image_without_scale(self.filename.text(),size=(768,768))[0][0]
            else:
                self.escala.setText("Escala: 1:1")
                image = self.Segementation.prepare_image_to_segmentation(self.filename.text(),size=(768,768),contrast=self.contrast,noise=self.denoise)[0]
        self.image_array = image
        imagen_Q = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
        #convert image into a pixmap
        pixmap = QPixmap.fromImage(imagen_Q)

        #display the image in the label
        self.image.setPixmap(pixmap)
        self.image.repaint()



    def segmentar_clicked(self):
        image = None
        #init the segmented window
        if self.auto_scale:
            image = self.Segementation.prepare_image_without_scale(self.filename.text(),size=self.img_size,contrast=self.contrast,noise=self.denoise)[1]
        else:
            image = self.Segementation.prepare_image_to_segmentation(self.filename.text(),contrast = self.contrast,noise =self.denoise,size=self.img_size)
        #display waiting dialog
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Comenzando segmentación, presione OK para continuar...")
        msg.setWindowTitle("Segmentación")
        msg.exec()
        image = self.Segementation.predict_mask(image)
        msg.destroy()
        #display a message box
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Imagen segmentada correctamente")
        msg.setWindowTitle("Segmentación")
        mask = self.Segementation.obtain_mask(image[0])
        if self.watershed:
            image,mask = self.Segementation.watershed(mask,image[0])
        else:
            pass

        self.segmented_window = SegmentWindow(mask,768,self.ratio,self.Segementation,self.Segementation.prepare_image_to_segmentation(self.filename.text(),size=self.img_size,noise=self.denoise,contrast=self.contrast)[0])
        self.segmented_window.show()
        msg.exec_()

    def introducir_tam_clicked(self):
        #list of sizes to choose
        sizes = [ (i,i) for i in range(256,2049,64)]
        #Open a dialog to introduce the size of the input image
        tam, ok = QtWidgets.QInputDialog.getItem(self, 'Tamaño', 'Introduce el tamaño de la imagen:',[str(i) for i in sizes],0,False)
        if ok:
            self.img_size = (int(tam.split(",")[0].replace("(","").replace(" ","")),int(tam.split(",")[0].replace("(","").replace(" ","")))
            self.tamano.setText("Tamaño: " + str(tam.split(",")[0].replace("(","").replace(" ","")) + "x" + str(tam.split(",")[0].replace("(","").replace(" ","")))
        else:
            return

    def introducir_escala_clicked(self):
        #open a dialog to introduce the scale
        scale, ok = QtWidgets.QInputDialog.getInt(self, 'Escala', 'Introduce la escala:',min=1)
        if ok:
            self.scale = scale
            self.escala.setText("Escala: 1:" + str(scale))
        else:
            return
        #display a message box 
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Escala introducida correctamente")
        msg.setWindowTitle("Escala")
        msg.exec_()
        #display a message box
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Dibuje una recta sobre la imagen para medir la distancia")
        msg.setWindowTitle("Medir distancia")
        msg.exec_()
        #Get the point of the mouse when it is pressed
        self.image.mousePressEvent = self.getFirstPoint
        self.image.mouseMoveEvent = self.getDistance
        #Get the point of the mouse when it is released
        self.image.mouseReleaseEvent = self.getDistance


    def browsefiles(self):
        #filter to only show files with .tif extension and directories
        filter = "Image Files (*.tif);;All Files (*)"
        filename = QFileDialog.getOpenFileName(self, 'Open File', './',filter=filter)
        if filename[0] == '':
            return
        try:
            self.filename.setText(filename[0])

            image = self.Segementation.prepare_image_to_segmentation(filename[0],size=(768,768))[0]
            self.image_array = image
            imagen_Q = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
            #convert image into a pixmap
            pixmap = QPixmap.fromImage(imagen_Q)

            #display the image in the label
            self.image.setPixmap(pixmap)
            self.image.repaint()

            self.escala.setVisible(True)
            self.escala.setText("Escala: 1:1")
            self.introducir_escala.setVisible(True)
            self.introducir_tam.setVisible(True)
            self.tamano.setVisible(True)
            self.tamano.setText("Tamaño: 1024x1024")
            self.segmentar.setVisible(True)
            self.ratio = self.Segementation.prepare_image_without_scale(filename[0],size=(768,768))[0][0]
            self.opciones.setVisible(True)
            self.denoise = False
            self.contrast = False
            self.watershed = False
            self.auto_scale = False

        except:
            #display a message box
            msg = QtWidgets.QMessageBox()       
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error al abrir el archivo")
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        

    def getFirstPoint(self, event):
        #substract (30,90) from event.pos
        x, y = event.pos().x() , event.pos().y()
        self.origin = QPoint(x,y)
        #Draw a rectangle on the image while the mouse is pressed
        self.image.setPixmap(QPixmap.fromImage(QImage(self.image_array, self.image_array.shape[1], self.image_array.shape[0], QImage.Format_RGB888)))
        #subsract event pos from origin to get the width and height of the label

    def getDistance(self, event):
        if event.type() == QtCore.QEvent.MouseButtonRelease:
            imagen_Q = QImage(self.image_array[0], self.image_array.shape[1], self.image_array.shape[0], QImage.Format_RGB888)
            #convert image into a pixmap
            pixmap = QPixmap.fromImage(imagen_Q)
            painter = QPainter(pixmap)
            #set the pen color
            painter.setPen(QPen(QtCore.Qt.red, 2, QtCore.Qt.SolidLine))
            painter.drawLine(self.origin, self.end)
            #set the pixmap back to the label
            self.image.setPixmap(pixmap)
            self.image.repaint()#Draw a straig
            painter.end()
            #disconnect the mouse move event from it
            self.image.mouseMoveEvent = None
            self.image.mouseReleaseEvent = None
            self.image.mousePressEvent = None
        else:
                
            image_copy = self.image.pixmap().copy()
            
            #convert image into a pixmap
            painter = QPainter(image_copy)

            #set the pen color
            painter.setPen(QPen(QtCore.Qt.red, 2, QtCore.Qt.SolidLine))
            #draw a straight line
            x,y = event.pos().x()  , self.origin.y()
            painter.drawLine(self.origin, QPoint(x,y))
            #set the pixmap back to the label
            self.image.setPixmap(image_copy)
            self.image.repaint()
            self.end = QPoint(x,y)
            painter.end()
            #disconnect the mouse move event from it
            
            #distancia entre los puntos
        self.distance = ((self.origin.x() - self.end.x())**2 + (self.origin.y() - self.end.y())**2)**0.5
        #check the event type
        

        print("Distance: " + str(self.distance))


app = QApplication(sys.argv)
widget = MainWindow()
widget.show()
sys.exit(app.exec_())
