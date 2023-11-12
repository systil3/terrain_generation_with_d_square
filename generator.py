import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFrame, QColorDialog
from PyQt5.QtGui import QColor

import qimage2ndarray
from visuals import *
import time
from os import path
from terrain import *
form_class = uic.loadUiType("program_revised.ui")[0]

class WindowClass(QMainWindow, form_class):
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.visualPixmap = QPixmap()
        self.visualLabel.setPixmap(self.visualPixmap)
        self.generateButton.clicked.connect(self.generate)
        self.generateButton.setAutoRepeat(True)
        self.segmentSlider.setRange(1,10)
        self.segmentSlider.setSingleStep(1)
        self.segmentSlider.valueChanged.connect(self.dispSegmentValue)
        self.roughnessSlider.setRange(1,20)
        self.roughnessSlider.setSingleStep(1)
        self.roughnessSlider.valueChanged.connect(self.dispRoughValue)
        self.denoiseSlider.setRange(0,50)
        self.denoiseSlider.setSingleStep(1)
        self.denoiseSlider.valueChanged.connect(self.dispDenoiseValue)
        self.denoiseStrength = self.denoiseSlider.value()
        self.normalizeFunctionSlider.setRange(0,100)
        self.normalizeFunctionSlider.setSingleStep(1)
        self.normalizeFunctionSlider.setValue(50)

        self.minHeightSlider.valueChanged.connect(self.dispMinHeightValue)
        self.maxHeightSlider.valueChanged.connect(self.dispMaxHeightValue)

        #0.5s to 2s
        self.repeatPeriodSlider.setRange(1,50)
        self.denoiseSlider.setSingleStep(1)

        self.visual_width = 800
        self.visual_height = 800
        self.visual = np.zeros((self.visual_height, self.visual_width))
        qImg = qimage2ndarray.array2qimage(self.visual).rgbSwapped()
        self.visualLabel.setPixmap(QPixmap(qImg))
        self.colorSchemeBox.currentIndexChanged.connect(self.setColorScheme)
        self.saveImgButton.clicked.connect(self.save_Image)

    def dispSegmentValue(self):
        n = self.segmentSlider.value()
        self.segmentLabel.setText(str(2**n + 1))

    def dispRoughValue(self):
        r = self.roughnessSlider.value()
        self.roughnessLabel.setText(str(r))

    def dispDenoiseValue(self):
        d = self.denoiseSlider.value()
        self.denoiseLabel.setText(str(d))
        self.denoiseStrength = d

    def dispMinHeightValue(self):
        h = self.minHeightSlider.value()
        self.minHeightLabel.setText(str(h))

    def dispMaxHeightValue(self):
        h = self.maxHeightSlider.value()
        self.maxHeightLabel.setText(str(h))

    def generate(self):
        if self.repeatCheckBox.isChecked():
            period = (self.repeatPeriodSlider.value()*1.5/50) + 0.5
            self.generate_terrain_loop(period)
        else:
            self.generate_terrain()

    def generate_terrain_loop(self, period, cycles=50):
        self.generate_terrain()
        for i in range(cycles):
            self.generate_terrain()
            self.setColorScheme()
            print("xxx")
            time.sleep(period)

    def generate_terrain(self):
        # terrain input : 0~255
        try:
            n = 2**(self.segmentSlider.value()) + 1  # Change this to your desired grid size (power of 2)
            roughness = self.roughnessSlider.value()  # Adjust this for terrain roughness
            seed = 0  # Adjust this for the initial seed value

            X = np.arange(n)
            Y = np.arange(n)

            terrain = generate_terrain(n, roughness / 3, seed).astype(np.int32)
            terrain = terrain[1:n, 1:n]

            #normalize (min ~ max -> 0 ~ 255)
            linear_norm = (100 - self.normalizeFunctionSlider.value()) / 100
            sig_norm = self.normalizeFunctionSlider.value() / 100
            terr_min_height = terrain.min()
            terr_max_height = terrain.max()
            min_height = self.minHeightSlider.value()
            max_height = self.maxHeightSlider.value()
            terrain = ((terrain - terr_min_height) * (max_height-min_height) // (terr_max_height - terr_min_height)) + min_height
            if self.gaussianNoiseCheckBox.isChecked():
                terrain = add_noise_on_planar_section(terrain, min_region_size=10, diff_thresh=0, mean=1, sigma=1)
            terrain = terrain * linear_norm + sig_mat(terrain) * sig_norm

            terrain = cv2.fastNlMeansDenoising(terrain.astype(np.uint8), None,
            self.denoiseStrength, 3, 25)

            #visualize
            width, height = 1024, 1024
            self.visual = np.zeros((width, height), dtype=np.int32)

            square_size = width // (n-1)

            for i in range(n-1):
                for j in range(n-1):
                    self.visual[square_size*i: square_size*(i+1),
                    square_size*j: square_size*(j+1)] = terrain[i][j].astype(np.uint8)

            self.setColorScheme()
            if self.autoSaveCheckBox.isChecked():
                self.save_Image()

        except Exception as e:
            print(e)

    def setColorScheme(self):
        try:
            curr_scheme = self.colorSchemeBox.currentText()

            if curr_scheme == "grayscale":
                resized = cv2.resize(self.visual.astype(np.float32),
                    dsize=(self.visual_width, self.visual_height), interpolation=cv2.INTER_LINEAR)
                qImg = qimage2ndarray.array2qimage(resized).rgbSwapped()
                self.visualLabel.setPixmap(QPixmap(qImg))
            elif curr_scheme == "normal":
                self.c_visual = convert_gray_to_rgb_matrix(self.visual)
                self.c_visual = setColorScheme(self.c_visual).astype(np.uint8)
                resized = cv2.resize(self.c_visual.astype(np.float32),
                    dsize=(self.visual_width, self.visual_height), interpolation=cv2.INTER_LINEAR)
                qImg = qimage2ndarray.array2qimage(resized)
                self.visualLabel.setPixmap(QPixmap(qImg))
        except Exception as e:
            print(e)

    def save_Image(self):

        filenum = 0
        file_extension = ".png"
        while path.exists("./results/" + str(filenum) + file_extension):
            filenum += 1
        file_name = str(filenum)

        curr_scheme = self.colorSchemeBox.currentText()
        target_path = "./results/" + str(filenum) + file_extension
        if curr_scheme == "grayscale":
            cv2.imwrite(target_path, self.visual)
        else:
            cv2.imwrite(target_path, np.flip(self.c_visual, axis=2))
