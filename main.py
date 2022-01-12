import cv2
from numpy.core.fromnumeric import shape
from GUI import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets,uic
from PyQt5.QtGui import QPixmap, QImage
import pyqtgraph as pg
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QFileDialog
import sys
import os
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import qimage2ndarray
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
import scipy.ndimage
from PyQt5.QtWidgets import QMessageBox


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        pg.setConfigOption('background','w')
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.lstFilesDCM = []
        self.frames = []
        self.imgs = []
        self.diag_arr = []
        self.diagonal = []
        self.ui.browse_button.clicked.connect(self.browse)
        self.ui.generate_button.clicked.connect(self.generate)
        self.ui.save_button.clicked.connect(self.save_slice)
        self.ui.axial_hSlider.valueChanged.connect(self.update_image)
        self.ui.axial_vSlider.valueChanged.connect(self.update_image)
        self.ui.sagittal_hSlider.valueChanged.connect(self.update_image)
        self.ui.sagittal_vSlider.valueChanged.connect(self.update_image)
        self.ui.coronal_hSlider.valueChanged.connect(self.update_image)
        self.ui.coronal_vSlider.valueChanged.connect(self.update_image)
        self.ui.mpr_hSlider.valueChanged.connect(self.mpr)
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Warning)
        
        
    def browse(self):
        """
        Browse a directory containing a series of axial CT images
        """
        sender = self.sender()
        self.statusBar().showMessage(sender.text() + ' was pressed')
        self.zoom_scala = 1
        self.rotate_degree = 0

        if(sender.text() == "Open"):

            folderpath  = QFileDialog.getExistingDirectory(self, 'Select Folder')
            lstFilesDCM = []  # create an empty list
            for dirName, subdirList, fileList in os.walk(folderpath ):
                for filename in fileList:
                    if ".dcm" in filename.lower():  # check whether the file's DICOM
                        lstFilesDCM.append(os.path.join(dirName,filename))
            
            for i in range(len(lstFilesDCM)):
                RefDs = pydicom.dcmread(lstFilesDCM[i])
                self.frames.append(RefDs)

        if self.frames:
            self.open()
        else:
            self.msg.critical(self, 'ERROR OCCURED','Browse DICOM file')
            
        

    def open(self):
        """
        Open browsed images and initializeseach plane and plot the middle slice
        """

        img_shape = list(self.frames[0].pixel_array.shape)
        img_shape.append(len(self.frames))

        img3d = np.zeros(img_shape)

        # fill 3D array with the images from the files
        for i, s in enumerate(self.frames):
            img2d = s.pixel_array
            img3d[:, :, i] = img2d

        # Normalize images
        self.imgs = cv2.normalize(img3d, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        self.imgs = self.imgs.astype(np.uint8)

        # Get the middle lice of all planes
        axial_image = self.imgs[:, :, img_shape[2]//2]
        coronal_image = np.flip(self.imgs[img_shape[0]//2, :, :].T)
        sagittal_image = np.flip(self.imgs[:, img_shape[1]//2, :].T)
        self.diag_arr = scipy.ndimage.interpolation.rotate(self.imgs, angle= 45, axes=(0,1))

        # Set value of sliders based on shapes of generated images
        self.update_shape()

        # Plot Axial plane
        self.plot_line_on_view(self.ui.window1, axial_image, axial_image.shape[1]//2, axial_image.shape[0]//2, Qt.red, Qt.yellow)
        
        # Plot Coronal plane
        self.plot_line_on_view(self.ui.window2, coronal_image, coronal_image.shape[1]//2, coronal_image.shape[0]//2, Qt.red, Qt.green)
        

        # Plot Sagittal plane
        self.plot_line_on_view(self.ui.window3, sagittal_image, sagittal_image.shape[1]//2, sagittal_image.shape[0]//2, Qt.yellow, Qt.green)

        # Plot MPR plane
        scene = QtGui.QGraphicsScene(self)            
        scene.addPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.diag_arr[:,:, self.diag_arr.shape[2]//2])))
        self.ui.window4.setScene(scene)

        
    def update_image(self):
        """
        Update Axial, Coronal, Sagittal planes based on sliders' values
        """
        if len(self.frames):
            # Values of Sliders
            self.sv_loc = self.ui.sagittal_vSlider.value()
            self.av_loc = self.ui.axial_vSlider.value()
            self.ah_loc = self.ui.axial_hSlider.value()
            self.sh_loc = self.ui.sagittal_hSlider.value()
            self.cv_loc = self.ui.coronal_vSlider.value()
            self.ch_loc = self.ui.coronal_hSlider.value()

            # Get slices of each plane based on sliders' values 
            self.axial = (self.imgs[:, :, self.sv_loc])
            self.coronal = np.flip(self.imgs[self.av_loc, :, :].T)
            self.sagittal = np.flip(self.imgs[:, self.ah_loc, :].T)

            # Axial Plane
            self.plot_line_on_view(self.ui.window1, self.axial, self.ah_loc, self.av_loc, Qt.red, Qt.yellow)
        
            # Coronal Plane
            self.plot_line_on_view(self.ui.window2, self.coronal, self.ch_loc, self.cv_loc, Qt.red, Qt.green)
        
            # Sagittal Plane
            self.plot_line_on_view(self.ui.window3, self.sagittal, self.sh_loc, self.sv_loc, Qt.yellow, Qt.green)

        else:
            self.msg.critical(self, 'ERROR OCCURED','No DICOM folder chosen')



    def update_shape(self):
        """
        Set maximum and initial values of all sliders
        """
        if(len(self.frames)):
            # Maximum Values
            self.v1, self.v2, self.v3 = self.imgs.shape
            self.ui.sagittal_vSlider.setMaximum(self.v3- 1)
            self.ui.coronal_vSlider.setMaximum(self.v3 - 1)
            self.ui.sagittal_hSlider.setMaximum(self.v1-1)
            self.ui.axial_vSlider.setMaximum(self.v2-1)
            self.ui.coronal_hSlider.setMaximum(self.v1-1)
            self.ui.axial_hSlider.setMaximum(self.v1-1)
            self.ui.mpr_hSlider.setMaximum(self.diag_arr.shape[2]-1)
            
            # Initial Values
            self.ui.sagittal_vSlider.setValue(self.ui.sagittal_vSlider.maximum()//2)
            self.ui.coronal_vSlider.setValue(self.ui.coronal_vSlider.maximum()//2)
            self.ui.sagittal_hSlider.setValue(self.ui.sagittal_hSlider.maximum()//2)
            self.ui.axial_vSlider.setValue(self.ui.axial_vSlider.maximum()//2)
            self.ui.coronal_hSlider.setValue(self.ui.coronal_hSlider.maximum()//2)
            self.ui.axial_hSlider.setValue(self.ui.axial_hSlider.maximum()//2)

        else:
            self.msg.critical(self, 'ERROR OCCURED','No DICOM folder chosen')


    def generate(self):
        """
        Generate MPR Plane based on user input of angle and plane
        """
        if len(self.frames):
            angle = self.ui.angle_mpr.text()
            index = self.ui.planes_box.currentIndex()
            # Axial Plane
            if index == 0:
                axes = (0,1)
            # Coronal Plane
            elif index == 1:
                axes = (0,2)
            # Sagittal Plane
            elif index == 2:
                axes = (1,2)
            # Rotate 3D image based on given parameters
            self.diag_arr = scipy.ndimage.interpolation.rotate(self.imgs, angle= int(angle), axes=axes)
            self.ui.mpr_hSlider.setMaximum(self.diag_arr.shape[2]-1)

        else:
            self.msg.critical(self, 'ERROR OCCURED','No DICOM folder chosen')

    def mpr(self):
        """
        Update MPR window based on slider value
        """
        if len(self.frames):
            self.mh_loc = self.ui.mpr_hSlider.value()
        
            if len(self.diag_arr):
                self.diagonal = self.diag_arr[:,:,self.mh_loc]
            
            if len(self.diagonal):
                scene = QtGui.QGraphicsScene(self)            
                scene.addPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(self.diagonal)))
                self.ui.window4.setScene(scene)

        else:
            self.msg.critical(self, 'ERROR OCCURED','No DICOM folder chosen')

    def plot_line_on_view(self, win, img, sh, sv, c1, c2):
        """
        Plot images on windows with horizontal and vertical lines
        """
        scene = QtGui.QGraphicsScene(self)            
        scene.addPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(img)))
        line = QtCore.QLineF(sh, 0, sh, img.shape[0])
        scene.addLine(line,pen=QPen(c1, 3))
        line = QtCore.QLineF(0, sv, img.shape[1], sv)
        scene.addLine(line,pen=QPen(c2, 3))
        win.setScene(scene)

    def save_slice(self):
        """
        Saves slice of chosen plane
        """
        if len(self.frames):
            fname, _filter = QFileDialog.getSaveFileName(self, 'save file', '~/untitled', "Image Files (*.jpg)")
            if fname:
                # Axial Plane
                if self.ui.save_box.currentIndex() == 0:
                    cv2.imwrite(fname, self.axial)
                # Coronal Plane
                elif self.ui.save_box.currentIndex() == 1:
                    cv2.imwrite(fname, self.coronal)
                # Sagittal Plane
                elif self.ui.save_box.currentIndex() == 2:
                    cv2.imwrite(fname, self.sagittal)
                # MPR Plane
                elif self.ui.save_box.currentIndex() == 3:
                    cv2.imwrite(fname, self.diagonal)
        else:
            self.msg.critical(self, 'ERROR OCCURED','No DICOM folder chosen')
            
           
def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()