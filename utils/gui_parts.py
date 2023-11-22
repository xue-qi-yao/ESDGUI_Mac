from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QCoreApplication, QPoint, QTimer
from PyQt5.QtWidgets import *
import cv2
import time
import numpy as np

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    process_img_signal = pyqtSignal(np.ndarray, int)
    

    def run(self):
        frame_idx = 0
        # capture from web cam
        cap = cv2.VideoCapture(0)  # TODO: Camera input
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        time.sleep(0.5)
        if cap.isOpened():
            while True:
                frame_idx += 1
                ret, cv_img = cap.read()
                #print(ret)
                time.sleep(0.15)  # TODO: removing sleep for camera
                if ret:
                    #print(cv_img)
                    
                    self.change_pixmap_signal.emit(cv_img)
                    self.process_img_signal.emit(cv_img, frame_idx)
                    
                else:
                    assert "Cannot get frame"
        else:
            cap.release()
            assert "Cannot get frame"

class usbVideo(QLabel):
    def __init__(self, size, parent=None):
        QLabel.__init__(self, parent)
        # 打开USB摄像头
        self.cap = cv2.VideoCapture(0)
        # 设置定时器以获取新的帧
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)
        self.size = size
        
    def update_frame(self):
        # 从摄像头读取一帧
        ret, frame = self.cap.read()
        if ret:
            # 将OpenCV图像转换为QImage
            img = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888).rgbSwapped()
            # img = img.scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
            # 将QImage转换为QPixmap，并将其设置为QLabel的图像
            pixmap = QPixmap.fromImage(img)
            self.setPixmap(pixmap)
            # 根据resolution自动适应widget和mainwindow的窗口大小
            self.setScaledContents(True)

class SAMMB(QMessageBox):
    def __init__(self, func1, func2):  
        super().__init__()
        self.setWindowTitle("SAM Choices")
        self.setText("Which SAM type do you want to use?")
        self.fullBtn = self.addButton('Full Image', QtWidgets.QMessageBox.YesRole)
        self.promptBtn = self.addButton("User Prompts", QtWidgets.QMessageBox.YesRole)
        self.fullBtn.clicked.connect(func1)
        self.promptBtn.clicked.connect(func2)

class PromptBox(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self.setGeometry(30,30,600,400)
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.show()

    def paintEvent(self, event):
        qp = QtGui.QPainter(self)
        br = QtGui.QBrush(QtGui.QColor(100, 10, 10, 40))  
        qp.setBrush(br)   
        qp.drawRect(QtCore.QRect(self.begin, self.end))       

    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        print(self.begin)
        print(self.end)
        self.update()
    
class CustomMB(QDialog):
    def __init__(self, labels):
        super().__init__()

        self.setWindowTitle("Name the New Segmentation")
        self.labels = labels
        QBtn = QDialogButtonBox.Save | QDialogButtonBox.Cancel
        self.finished = False
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.checkText)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.hlayout = QHBoxLayout()
        self.name = QLabel('Name:')
        self.message = QLineEdit()
        self.hlayout.addWidget(self.name)
        self.hlayout.addWidget(self.message)
        self.layout.addLayout(self.hlayout)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def getText(self):
        return self.message.text()

    def checkText(self):
        if self.message.text() == "":
            checkMsg = "Name cannot be empty"
            dlg = QMessageBox()
            dlg.setWindowTitle("Warning!!!")
            dlg.setText(checkMsg)
            dlg.setIcon(QMessageBox.Warning)
            button = dlg.exec()
            if button == QMessageBox.Ok:
                self.message.setText("")
        elif self.labels.findText(self.message.text()) != -1:
            checkMsg = "Name already exists"
            dlg = QMessageBox()
            dlg.setWindowTitle("Warning!!!")
            dlg.setText(checkMsg)
            dlg.setIcon(QMessageBox.Warning)
            button = dlg.exec()

            if button == QMessageBox.Ok:
                self.message.setText("")
        else:
            checkMsg = "Successfully create new label"
            self.finished = True
            self.accept()

def get_image_format(image):
    qformat = QImage.Format_Indexed8
    if len(image.shape) == 3:
        if (image.shape[2]) == 4:
            qformat = QImage.Format_RGBA8888
        else:
            qformat = QImage.Format_RGB888

    return qformat   

def hex_to_rgb(hex1):
    h = hex1.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))[::-1]