from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt


class SegmentationModule:
    def __init__(self, model_path="./Modelos/modeloPropio.h5",model_classification = './Modelos/modelo_denso_prueba_variacion_pardo.h5',model_path_numeros="./Modelos/modelonumeros.h5"):
        self.model = keras.models.load_model(model_path)
        self.numbers_model = keras.models.load_model(model_path_numeros)
        self.model_classification = keras.models.load_model(model_classification)
        self.particles_classes = ["bipiramides","hexagonos","circulos","cuadrados","rectangulos"]
        self.colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]

    def classify(self,img):

        #add padding to the image to make it square
        desired_size = int(max(img.shape[0],img.shape[1])*1.5)
        delta_w = desired_size - img.shape[1]
        delta_h = desired_size - img.shape[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0]
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)

        img = cv2.resize(img,(100,100))

        img  = img.reshape((1,)+img.shape)

        return np.argmax(self.model_classification.predict(img))

    #devuelve tensor de tamaño (1,160,160,3) para meter en la red
    def prepare_image_to_segmentation(self,path,size=(1024,1024),contrast=False,noise=False):
        im = cv2.imread(path,0)
        im = cv2.resize(im,size)
 
        #delete noise
        if noise:
            im = cv2.GaussianBlur(im,(5,5),0)

        if contrast:
            #increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            im = clahe.apply(im)
        
        
        
        im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
        im = im.reshape((1,)+im.shape)
        
        return im
    
    #obtenemos el array numpy de la imagen segmentada
    def obtain_mask(self,i):
        mask = np.argmax(i, axis=-1)
        mask = np.expand_dims(mask,axis=-1)
        #convert to uint8
        mask = mask.astype(np.uint8)

        if np.count_nonzero(mask == 0) < np.count_nonzero(mask == 1):
            mask = cv2.bitwise_not(mask)
        
        mask = mask * 255
        mask = mask.reshape(mask.shape[0],mask.shape[1])
        return mask

    def display_mask(self,i):
        mask = np.argmax(i, axis=-1)
        mask = np.expand_dims(mask,axis=-1)
        plt.imshow(mask)
        plt.show()


    #SOLO FUNCIONA CON LAS FOTOS DEL CITIUS
    def separar_regiones_escala(self,img,path="./prueba/"):  
        output = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        
        if not os.path.exists(path):
            os.mkdir(path)
        
        nums = []

        pos_nums = []

        width = img.shape[1]

        scale = None
        
        for i in range(0, numLabels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            output = img.copy()
            #las constantes hacen que funcione con las fotos del citius
            if area > width*0.0976 and h > width*0.0244 and i!=0:
                componentMask = (labels == i).astype("uint8") * 255
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
                #crop image in the bounding box
                componentMask = componentMask[y:y+h,x:x+w]
                #cv2.imwrite("./Modelosprueba/"+str(i)+".png",componentMask)
                nums.append(componentMask)
                pos_nums.append(x)
            
            if w > img.shape[1]*0.05 and i!=0:
                componentMask = (labels == i).astype("uint8") * 255


                scale = w

                #crop image in the bounding box
                componentMask = componentMask[y:y+h,x:x+w]
                cv2.imwrite("./prueba/"+str(i)+".png",componentMask)
        number = 0
        #order nums by the value of pos_nums
        nums = [x for _,x in sorted(zip(pos_nums,nums))]
        #iterate over images and make predictions with modelo_numeros
        for i in range(len(nums)):
            #resize 
            nums[i] = cv2.resize(nums[i]/255,(28,28))
            #make prediction
            pred = self.numbers_model.predict(nums[i].reshape(1,28,28,1))
            #add to number
            number = number + pred.argmax()*10**(len(nums)-i-1)
            #save image
            cv2.imwrite(path+str(i)+".png",nums[i]*255)
        try:
            ratio_pixel_measure = number/scale
        except:
            ratio_pixel_measure = 1
            scale = 1
        
        if number == 0:
            number = 1
        #tenemos en nums las imagenes de cada numero, aplicarle una red neuronal para clasificarlos
        #el area de un pixel es el lado del pixel al cuadrado
        ratio_pixel_area = ratio_pixel_measure**2
        return ratio_pixel_area,number


    def get_scale_area(self,im,segmented_scale):
        contours, _ = cv2.findContours(segmented_scale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #get the first contour and save it in a variable
        cnt = contours[0]
        #get the region delimited by cnt in the original image
        x,y,w,h = cv2.boundingRect(cnt)
        #crop the image
        crop = im[y:y+h,x:x+w]
        #save the image without the crop area
        crop = 255 - crop
        crop //= 255
        cv2.imwrite("scale.jpg",crop)

        #put 255 in the crop area
        im[y:y+h,x:x+w] = 255
        #get the bounding box of the objects in the crop image
        return self.separar_regiones_escala(crop,"./prueba/"),im.reshape(1,im.shape[0],im.shape[1],1)
    


    def prepare_image_without_scale(self,path,size=(1024,1024),noise=False,contrast=False):
        #read the image
        gray = cv2.cvtColor(self.prepare_image_to_segmentation(path,size=size)[0], cv2.COLOR_BGR2GRAY)

        #Solo me interesa las regiones blancas para sacar la escala
        ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        #create tall vertical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))

        #create short long horizontal kernel
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))

        #detect vertical lines
        vertical_lines = cv2.erode(thresh, kernel, iterations=5)
        vertical_lines = cv2.dilate(vertical_lines, kernel, iterations=5)

        #detect horizontal lines
        horizontal_lines = cv2.erode(thresh, kernel2, iterations=5)
        horizontal_lines = cv2.dilate(horizontal_lines, kernel2, iterations=5)

        #add the two images
        lines = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

        #perform final erosion and binarization
        #lines = cv2.erode(lines, kernel, iterations=3)
        linesaux = lines.copy()
        ret,img_final = cv2.threshold(lines, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        
        scale, img_final = self.get_scale_area(gray,img_final)
        img_final = img_final[0]

        #delete noise
        if noise:
            img_final = cv2.GaussianBlur(img_final,(5,5),0)

        if contrast:
            #increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_final = clahe.apply(img_final)

        

        
        #convert img_final to rgb
        img_final = cv2.cvtColor(img_final ,cv2.COLOR_GRAY2RGB)
        
        img_final = img_final.reshape((1,)+img_final.shape)


        return scale,img_final
    

    def separar_regiones(self,img,segmented=None,path="./prueba/",get_original = False):

        img = img.astype("uint8")

        output_mask = np.zeros((img.shape[0],img.shape[1],3),dtype="uint8")

        
        if segmented is None:
            segmented = img.copy()

        segmented[0,:] = 0
        segmented[-1,:] = 0
        segmented[:,0] = 0
        segmented[:,-1] = 0

        img[0,:] = 0
        img[-1,:] = 0
        img[:,0] = 0
        img[:,-1] = 0
        
        

        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        output = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        
        cell_areas = []
        for i in range(5):
            cell_areas.append([])

        print("[INFO] {} unique connected components found".format(numLabels))

        if not os.path.exists(path):
            os.mkdir(path)
        
        for i in range(0, numLabels):
            # if this is the first component then we examine the
            # *background* (typically we would just ignore this
            # component in our loop)
            if i == 0:
                text = "examining component {}/{} (background)".format(
                    i + 1, numLabels)
            # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format( i + 1, numLabels)
            # print a status message update for the current connected
            # component
            print("[INFO] {}".format(text))
            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]



            componentMask = (labels == i).astype("uint8") * 255

            if area > 10 and i != 0 and i != 1:
                  
                #Miramos si queremos cogerla la imagen segmentada original o la modificada con watershed
                if get_original:
                    #crop the regio of the image in segmented
                    cropped = segmented[y:y+h,x:x+w]

                    tipo_particula = self.classify(cropped)
                    #count the number of pixels with value 1
                    cell_areas[tipo_particula].append(np.count_nonzero(cropped != 0))

                    #where the values on componentMask are different to 0, put the value on output_mask of tipo_particula
                    output_mask[componentMask != 0] = self.colors[tipo_particula]




                    cv2.imwrite(path+"cropped"+str(i)+".png",cropped)
                else:
                    cv2.imwrite(path+str(i)+".png",componentMask)
                    cropped = componentMask[y:y+h,x:x+w]
                    cv2.imwrite(path+"cropped"+str(i)+".png",cropped)
                    
                    tipo_particula = self.classify(cropped)
                    #count the number of pixels with value different to 0
                    cell_areas[tipo_particula].append(np.count_nonzero(cropped != 0))  
                    output_mask[componentMask != 0] = self.colors[tipo_particula]

        return cell_areas,output_mask
    
    
    def watershed(self,predicted_mask,img):
        #convert to gray the image
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.uint8)
    
        kernel = np.ones((3,3),np.uint8)  
        #opening
        predicted_mask = cv2.morphologyEx(predicted_mask, cv2.MORPH_OPEN, kernel, iterations = 2)
        #sure background area  
        sure_bg = cv2.dilate(predicted_mask,kernel,iterations=10)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(predicted_mask,cv2.DIST_L2,0)
        ret, sure_fg = cv2.threshold(dist_transform,0.15*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1

        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        #markers = cv2.watershed(img,markers)

        markers[markers == 1] = 0
        markers[markers > 1] = 255
        

        #markers = cv2.watershed(img,markers)

        #img[markers == -1] = [0,255,255]


        markers = markers.astype(np.uint8)

        return img,markers
    
    def get_histogram(self,particles_sizes,image,color_mask):
        bins = 25
        #create an matplotlib object with 3 rows and 2 column
        fig, axs = plt.subplots(3, 2,figsize=(10,10))
        fig.suptitle('Histograms of particles',fontsize=16,fontweight='bold')
        #the first plot has 2 different images, the original and the segmented
        axs[0,0].imshow(image)
        axs[0,0].imshow(color_mask,alpha=0.3)
        axs[0,0].set_title('Original image')
        #second is for the first histogram
        axs[0,1].hist(particles_sizes[0],bins=bins,color=(self.colors[0][0]/255,self.colors[0][1]/255,self.colors[0][2]/255,1))
        axs[0,1].set_title('Histogram of ' + str(self.particles_classes[0]) + ' particles')
        axs[0,1].set_xlabel('Size')
        axs[0,1].set_ylabel('Quantity')
        #third is for the second histogram
        axs[1,0].hist(particles_sizes[1],bins=bins,color=(self.colors[1][0]/255,self.colors[1][1]/255,self.colors[1][2]/255,1))
        axs[1,0].set_title('Histogram of ' + str(self.particles_classes[1]) + ' particles')
        axs[1,0].set_xlabel('Size')
        axs[1,0].set_ylabel('Quantity')
        #fourth is for the third histogram
        axs[1,1].hist(particles_sizes[2],bins=bins,color=(self.colors[2][0]/255,self.colors[2][1]/255,self.colors[2][2]/255,1))
        axs[1,1].set_title('Histogram of ' + str(self.particles_classes[2]) + ' particles')
        axs[1,1].set_xlabel('Size')
        axs[1,1].set_ylabel('Quantity')
        #fifth is for the fourth histogram
        axs[2,0].hist(particles_sizes[3],bins=bins,color=(self.colors[3][0]/255,self.colors[3][1]/255,self.colors[3][2]/255,1))
        axs[2,0].set_title('Histogram of ' + str(self.particles_classes[3]) + ' particles')
        axs[2,0].set_xlabel('Size')
        axs[2,0].set_ylabel('Quantity')
        #sixth is for the fifth histogram
        axs[2,1].hist(particles_sizes[4],bins=bins,color=(self.colors[4][0]/255,self.colors[4][1]/255,self.colors[4][2]/255,1))
        axs[2,1].set_title('Histogram of ' + str(self.particles_classes[4]) + ' particles')
        axs[2,1].set_xlabel('Size')
        axs[2,1].set_ylabel('Quantity')


        #show the plots
        plt.tight_layout()
        #plt.show()
        #save the plot
        fig.savefig('histogram.png')
        #return the fig
        return fig
    
    def predict_mask(self,image):
        return self.model.predict(image)
    
    def get_areas(self,white_pixels,ratio):
        areas = []
        for i in range(0,len(white_pixels)):
            areas.append([j*ratio for j in white_pixels[i]])
        return areas
 

