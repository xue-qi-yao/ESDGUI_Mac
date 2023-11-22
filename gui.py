import os
import torch
import cv2
import sys
import time
import warnings
import argparse
import numpy as np
import pandas as pd
from threading import Thread
import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import torch
import random

from utils.guis import PhaseCom, draw_segmentation, add_text
from utils.parser import ParserUse
from utils.gui_parts import *
from utils.report_tools import generate_report
from canvas import Canvas

warnings.filterwarnings("ignore")
DEFAULT_STYLE = """
                QProgressBar{
                    border: 3px solid black;
                    background-color: white;
                    text-align: center;
                    height: 20px;
                }
                QProgressBar::chunk {
                    background-color: green;
                }
                """
COMBOBOX = """
            QComboBox {
                border: 1px solid grey;
                border-radius: 3px;
                padding: 1px 2px 1px 2px;  
                min-width: 10em;
                min-height: 20px;  
            }
            QComboBox QAbstractItemView::item 
            {
                min-height: 20px;
            }
        """


def add_text(fc, results, fps, frame):
    w, h, c = frame.shape
    cv2.putText(frame, "   Time: {:<55s}".format(fc.split("-")[-1].split(".")[0]), (30, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)
    cv2.putText(frame, "  Phase: {:<15s}".format(results), (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, " Trainee: {:<15s}".format(fps), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # cv2.putText(frame, " Blood vessel".format(fps), (140, w - 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    # cv2.putText(frame, " Muscularis".format(fps), (140, w - 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # cv2.putText(frame, " Submucosa".format(fps), (140, w - 120),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    return frame

class ImageProcessingThread(QThread):
    processed_frame = pyqtSignal(np.ndarray)

    def __init__(self, start_x, end_x, start_y, end_y, cfg):
        super().__init__()
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.frames_to_process = []
        self.phaseseg = PhaseCom(arg=cfg)
        self.processing_interval = 2

    def run(self):
        while True:
            if len(self.frames_to_process) >= self.processing_interval:
                frame = self.frames_to_process[-1]
                pred = self.process_image(frame)
                self.frames_to_process = []
                self.processed_frame.emit(pred)

    def process_image(self, img):
        cv_img = img[self.start_x:self.end_x, self.start_y:self.end_y]  # Crop images
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pred = self.phaseseg.phase_frame(rgb_image)  # TODO:可以用time查看該語句的延遲時間

        return pred


    def add_frame(self, frame):
        self.frames_to_process.append(frame)


class Ui_iPhaser(QMainWindow):
    def __init__(self):
        super(Ui_iPhaser, self).__init__()

    def setupUi(self, cfg):
        self.setObjectName("iPhaser")
        self.resize(1825, 1175)
        self.setMinimumHeight(965)
        self.setMinimumWidth(965)
        self.setStyleSheet("QWidget#iPhaser{background-color: #2e5cb8}")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        old_pos = self.frameGeometry().getRect()
        curr_x = old_pos[2]
        curr_y = old_pos[3]
        self.redo = False
        self.point_size = 1
        self.mQImage = QPixmap('./images/test.jpg')
        self.cbFlag = 0
        self.size = QSize(curr_x - 25 - 500, curr_y - 65 - 250)
        self.old_pos = self.frameGeometry().getRect()
        self.save_folder = "../Records"
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        self.down_ratio = 1  # cfg.down_ratio
        # Statue parameters
        self.init_status()
        Vblocks = ['case information', 'phase recognition', 'online analytics']
        Hblocks = ['training session']
        self.FRAME_WIDTH, self.FRAME_HEIGHT, self.stream_fps = self.get_frame_size()
        self.MANUAL_FRAMES = self.stream_fps * cfg.manual_set_fps_ratio
        self.manual_frame = 0
        self.enable_seg = False
        self.force_seg = False
        self.seg_alpha = 0.2
        # self.FRAME_WIDTH = 1280
        # self.FRAME_HEIGHT = 720
        self.CODEC = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.pbFlag = True
        # self.date_time = datetime.now().strftime("%d/%m/%Y-%H:%M:%S.%f")
        self.fps = 0
        self.selected_tool = 0
        self.f_image = None
        self.surgeons = QComboBox()
        self.surgeons.setObjectName("SurgeonsName")
        self.surgeons.setStyleSheet(COMBOBOX)
        self.surgeons.setEditable(True)
        self.surgeons.lineEdit().setAlignment(Qt.AlignCenter)
        self.surgeons.lineEdit().setFont(QFont("Arial", 16, QFont.Bold))
        self.surgeons.lineEdit().setReadOnly(True)
        self.surgeons.setCurrentIndex(-1)
        self.pred = "--"
        self.pop_image = {}
        self.pop_idx = []
        self.total_diff = []
        self.pop_image_count = []
        self.seg_pred = torch.zeros([4, 224, 224])  # .numpy()
        self.log_data = []
        self.actionList = {}
        self.last_idx = []
        self.count_image = {}
        self.image_count = []
        self.ru = 0
        self.aaa = 0
        self.bbb = 0
        self.index2phase = {0: "idle", 1: "marking", 2: "injection", 3: "dissection"}
        self.phase_probs = {'Marking': 0,
                            'Injection': 0,
                            'Dissection': 0,
                            'Idle': 1}
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.resizeEvent = self.windowResized

        # # 在主窗口中添加usbVideo控件
        # self.usbVideo = usbVideo(self.size, parent=self.centralwidget)
        # self.usbVideo.setGeometry(QtCore.QRect(500, 250, curr_x-25-500, curr_y-65-250))
        # newly added
        self.DisplayVideo = QtWidgets.QLabel(self.centralwidget)
        self.DisplayVideo.setGeometry(QtCore.QRect(500, 250, curr_x - 25 - 500, curr_y - 65 - 250))
        self.DisplayVideo.setScaledContents(True)
        self.DisplayVideo.setStyleSheet("background-color: black;")
        self.DisplayVideo.setText("")
        self.DisplayVideo.setObjectName("DisplayVideo")
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.DisplayVideo.setSizePolicy(size_policy)



        self.video = False
        self.disply_width = 1080
        self.display_height = 720
        self.start_x = 0
        self.end_x = 450
        self.start_y = 0
        # self.start_y = 450
        self.end_y = 450
        # cv_img[0:1150, 450:1800]
        self.save_folder = os.path.join("../Records")
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)
        self.down_ratio = 1  # cfg.down_ratio
        self.start_time = "--:--:--"
        self.trainee_name = "--"
        self.manual_set = "--"
        # model prediction
        # self.centralwidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.setCentralWidget(self.usbVideo)
        # self.setCentralWidget(self.centralwidget)
        self.label_mask = QImage()
        self.image = QImage()
        self.canvas = Canvas(self.label_mask, self.image, parent=self.centralwidget)
        # self.canvas.setGeometry(QtCore.QRect(500, 250, curr_x-25-500, curr_y-65-250))
        # self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.canvas.setScaledContents(True)
        # self.setCentralWidget(self.centralwidget)
        # self.centralWidget.addWidget(self.canvas)
        self.bu = 0
        self.fileButton = QPushButton('File', self)
        self.fileButton.setFont(QFont("Arial", 12, QFont.Bold))
        self.fileButton.setGeometry(QtCore.QRect(0, 0, 60, 25))
        self.fileButton.setStyleSheet("background-color:#dee0e3;cocanvas import Canvaslor:black;")
        self.fileButton.setObjectName('FileButton')
        self.settingButton = QPushButton('Setting', self)
        self.settingButton.setFont(QFont("Arial", 12, QFont.Bold))
        self.settingButton.setGeometry(QtCore.QRect(60, 0, 60, 25))
        self.settingButton.setStyleSheet("background-color:#dee0e3;color:black;")
        self.settingButton.setObjectName('SettingButton')

        self.startButton = QPushButton(self)
        # self.startButton.setFont(QFont("Arial",12, QFont.Bold))
        self.startButton.setGeometry(QtCore.QRect(self.width() - 300, 100, 80, 80))
        self.startButton.setStyleSheet("background-color: DarkGreen;")
        self.startButton.setObjectName('StartButton')
        self.start_pixmap = QtGui.QIcon()
        self.start_pixmap.addFile("./images/start.png", QtCore.QSize(80, 80), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.startButton.setIcon(self.start_pixmap)
        self.startButton.setIconSize(QtCore.QSize(150, 150))
        self.startButton.pressed.connect(self.onButtonClickStart)

        self.stopButton = QPushButton(self)
        # self.stopButton.setFont(QFont("Arial",12, QFont.Bold))
        self.stopButton.setGeometry(QtCore.QRect(self.width() - 200, 100, 80, 80))
        self.stopButton.setStyleSheet("background-color:DarkGrey;")
        self.stopButton.setObjectName('StopButton')
        self.stop_pixmap = QtGui.QIcon()
        self.stop_pixmap.addFile("./images/stop.png", QtCore.QSize(80, 80), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stopButton.setIcon(self.stop_pixmap)
        self.stopButton.setIconSize(QtCore.QSize(150, 150))
        self.stopButton.clicked.connect(self.onButtonClickStop)

        self.layoutWidget = QtWidgets.QWidget(self)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 20, 440, 800))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setObjectName("verticalLayout")

        self.layoutWidget1 = QtWidgets.QWidget(self)
        self.layoutWidget1.setGeometry(QtCore.QRect(500, 20, 640, 225))
        self.layoutWidget.setObjectName("layoutWidget1")
        self.verticalLayout1 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.verticalLayout1.setObjectName("verticalLayout1")

        self.layoutWidget2 = QtWidgets.QWidget(self)
        self.layoutWidget2.setGeometry(QtCore.QRect(500, 270, 440, 225))
        self.layoutWidget.setObjectName("layoutWidget2")
        self.verticalLayout2 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.verticalLayout2.setObjectName("verticalLayout2")

        self.VLayout1 = QtWidgets.QVBoxLayout()
        self.VLayout1.setObjectName("VLayout1")
        self.trainLabelTitle = QtWidgets.QLabel('Training Session')
        self.trainLabelTitle.setObjectName("TrainLabelTitle")
        self.trainLabelTitle.setStyleSheet("color:white;")
        self.trainLabelTitle.setFont(QFont("Arial", 18, QFont.Bold))
        self.VLayout1.addWidget(self.trainLabelTitle, 16)
        self.trainLabel = QtWidgets.QWidget()
        self.trainLabel.setObjectName("TrainLabel")
        self.trainLabel.setAttribute(Qt.WA_StyledBackground, True)
        self.trainLabel.setStyleSheet("background-color: #99ccff; border-radius:5px")
        self.VLayout1.addWidget(self.trainLabel, 84)
        self.verticalLayout1.addLayout(self.VLayout1)

# start of new summary report
        egrid = QGridLayout()
        group1 = QGroupBox()
        group1.setObjectName("DurationGroup")
        group1.setStyleSheet("QGroupBox#DurationGroup{border:0;}")
        hbox = QHBoxLayout()
        hbox.setObjectName("DurationLayout")
        e1 = QLabel("Duration:")
        e1.setObjectName("Duration")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        e1.setStyleSheet("color: white;")
        self.flag = False
        self.Timer = QtCore.QTimer()
        self.Timer.timeout.connect(self.countTime)
        self.Timer.start(1)
        self.hour = 0
        self.minute = 0
        self.second = 0
        e2 = QLabel('{:02d}'.format(self.hour))
        e2.setAlignment(Qt.AlignCenter)
        e2.setObjectName("DurationHour")
        e2.setFont(QFont("Arial", 16, QFont.Bold))
        e2.setStyleSheet("background-color: white;")
        self.duraHour = e2
        e3 = QLabel("hrs:")
        e3.setObjectName("DurationHourUnit")
        e3.setFont(QFont("Arial", 16, QFont.Bold))
        e3.setStyleSheet("color: white;")
        e4 = QLabel('{:02d}'.format(self.minute))
        e4.setAlignment(Qt.AlignCenter)
        e4.setObjectName("DurationMinute")
        e4.setFont(QFont("Arial", 16, QFont.Bold))
        e4.setStyleSheet("background-color: white;")
        self.duraMinute = e4
        e5 = QLabel("min:")
        e5.setObjectName("DurationMinuteUnit")
        e5.setFont(QFont("Arial", 16, QFont.Bold))
        e5.setStyleSheet("color: white;")
        e6 = QLabel('{:02d}'.format(self.second))
        e6.setAlignment(Qt.AlignCenter)
        e6.setObjectName("DurationSecond")
        e6.setFont(QFont("Arial", 16, QFont.Bold))
        e6.setStyleSheet("background-color: white;")
        self.duraSecond = e6
        e7 = QLabel("sec")
        e7.setObjectName("DurationSecondUnit")
        e7.setFont(QFont("Arial", 16, QFont.Bold))
        e7.setStyleSheet("color: white;")
        hbox.addWidget(e1)
        hbox.addWidget(e2)
        hbox.addWidget(e3)
        hbox.addWidget(e4)
        hbox.addWidget(e5)
        hbox.addWidget(e6)
        hbox.addWidget(e7)
        hbox.setSpacing(5)
        group1.setLayout(hbox)
        group2 = QGroupBox()
        group2.setObjectName("SurgeonsGroup")
        group2.setStyleSheet("QGroupBox#SurgeonsGroup{border:0;}")
        hbox_1 = QHBoxLayout()
        hbox_1.setObjectName("SurgeonsLayout")
        e8 = QLabel("Surgeons:")
        e8.setObjectName("Surgeons")
        e8.setFont(QFont("Arial", 16, QFont.Bold))
        e8.setStyleSheet("color: white;")
        # e9.setAlignment(Qt.AlignCenter)
        e10 = QLabel()
        hbox_1.addWidget(e8)
        hbox_1.addWidget(self.surgeons)
        hbox_1.addWidget(e10)
        hbox_1.setSpacing(10)
        # hbox_1.setAlignment(Qt.AlignLeft)
        group2.setLayout(hbox_1)
        group2.setAlignment(Qt.AlignLeft)
        group3 = QGroupBox()
        group3.setObjectName("ReportGroup")
        group3.setStyleSheet("QGroupBox#ReportGroup{border:0;}")
        hbox_2 = QHBoxLayout()
        hbox_2.setObjectName("ReportLayout")
        self.reportButton = QPushButton("Generate Report")
        self.reportButton.setObjectName("ReportButton")
        self.reportButton.setFont(QFont("Arial", 16))
        self.reportButton.setStyleSheet("QPushButton"
                                        "{"
                                        "background-color: green;"
                                        "color: white;"
                                        "padding: 5px 15px;"
                                        "margin-top: 10px;"
                                        "outline: 1px;"
                                        "min-width: 8em;"
                                        "}")
        self.reportButton.clicked.connect(self.generateReport)
        hbox_2.addWidget(self.reportButton)
        hbox_2.setAlignment(Qt.AlignCenter)
        group3.setLayout(hbox_2)
        egrid.addWidget(group1, 0, 0)
        egrid.addWidget(group2, 1, 0)
        egrid.addWidget(group3, 2, 0)
        self.VLayout2 = QtWidgets.QVBoxLayout()
        self.VLayout2.setObjectName("VLayout2")
        self.summaryReportTitle = QtWidgets.QLabel('Summary report')
        self.summaryReportTitle.setObjectName("SummaryReportTitle")
        self.summaryReportTitle.setStyleSheet("color:white;")
        self.summaryReportTitle.setFont(QFont("Arial", 18, QFont.Bold))
        self.VLayout2.addWidget(self.summaryReportTitle, 16)
        self.summaryReport = QtWidgets.QWidget()
        self.summaryReport.setObjectName("summaryReport")
        self.summaryReport.setAttribute(Qt.WA_StyledBackground, True)
        self.summaryReport.setStyleSheet("background-color: #99ccff; border-radius:5px")
        self.summaryReport.setLayout(egrid)
        self.VLayout2.addWidget(self.summaryReport, 84)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: black;")
        self.summaryReportOutput1 = QtWidgets.QWidget()
        self.summaryReportOutput2 = QtWidgets.QWidget()
        self.VLayout2.addWidget(line)
        self.VLayout2.addWidget(self.summaryReportOutput1)
        self.VLayout2.addWidget(self.summaryReportOutput2)

        self.verticalLayout2.addLayout(self.VLayout2)

# end of new summary report

        Vpercent = 100 / len(Vblocks)
        for i in Vblocks:
            self.setVLayout(i, Vpercent)

        self.imageLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageLabel.setGeometry(QtCore.QRect(0, 0, 400, 400))  # Set initial size and position
        self.imageLabel.setAlignment(QtCore.Qt.AlignBottom | QtCore.Qt.AlignLeft)  # Align to the left bottom corner
        self.imageLabel.setObjectName("CUHK_logol")
        # Set the size policy of the image label to Ignored
        self.imageLabel.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        # Load the image
        image_path = "images/CUHK_logol.png"  # Replace with the actual path to your image
        image = QtGui.QPixmap(image_path)

        # Resize the image to fit within the available space while maintaining the aspect ratio
        scaled_image = image.scaled(self.imageLabel.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        # Set the scaled image as the pixmap for the image label
        self.imageLabel.setPixmap(scaled_image)

        # Adjust the position and size of the image label when the main window is resized
        self.centralwidget.resizeEvent = self.windowResized

        self.setupCaseInformation()
        self.setupTrainer()
        self.setupPhaseRecog()
        self.setupAnalytics()
        self.setCentralWidget(self.centralwidget)
        self.retranslateUi()

        self.camera = cv2.VideoCapture(0)
        self.process_frames = False

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(30)

        self.processing_thread = ImageProcessingThread(start_x=self.start_x,
                                                       end_x=self.end_x,
                                                       start_y=self.start_y,
                                                       end_y=self.end_y,
                                                       cfg=cfg)
        self.phase_probs = {'Idle': 1}
        self.processing_thread.processed_frame.connect(self.update_pred)
        self.processing_thread.moveToThread(QCoreApplication.instance().thread())
        self.processing_thread.start()

    def windowResized(self, event):
        # Get the current window size
        width = self.centralwidget.width()
        height = self.centralwidget.height()

        # Calculate the new size and position of the video widget
        videoWidth = width - 500 -25
        videoHeight = height - 250 -25

        # Set the new geometry of the video widget
        self.DisplayVideo.setGeometry(QtCore.QRect(970, 250, videoWidth, videoHeight))

        # Get the current window size
        width = self.centralwidget.width()
        height = self.centralwidget.height()

        # Calculate the new position of the image label
        imageWidth = self.imageLabel.pixmap().width()
        imageHeight = self.imageLabel.pixmap().height()
        imageX = 50  # Align to the left side of the window
        imageY = height - imageHeight - 50 # Align to the bottom of the window

        # Set the new geometry of the image label
        self.imageLabel.setGeometry(QtCore.QRect(imageX, imageY, imageWidth, imageHeight))

        self.startButton.setGeometry(QtCore.QRect(self.width() - 300, 100, 80, 80))
        self.stopButton.setGeometry(QtCore.QRect(self.width() - 200, 100, 80, 80))

    def update_image(self):
        """Convert from an opencv image to QPixmap"""
        # Collect settings of functional keys
        # cv_img = cv_img[30:1050, 695:1850]
        ret, frame = self.camera.read()
        if ret:
            if self.WORKING:
                self.processing_thread.add_frame(frame)
                self.display_frame(frame)
            else:
                self.display_frame(frame)

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.manual_frame = self.manual_frame - 1
        if self.manual_frame <= 0:
            self.manual_frame = 0
            self.manual_set = "--"
        if self.INIT:
            self.date_time = datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
            if self.manual_frame > 0:
                self.pred = self.manual_set
            rgb_image = add_text(self.date_time, self.pred, self.trainee.text(), rgb_image)
        if self.WORKING:
            # print('write', rgb_image.shape)
            self.date_time = datetime.datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
            rbg_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            rgb_image = add_text(self.date_time, self.pred, self.trainee.text(), rgb_image)
            # self.output_video.write(rbg_image)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.size.width(), self.size.height(),  Qt.KeepAspectRatio)
        p = QPixmap.fromImage(p)
        self.DisplayVideo.setPixmap(p)

    def update_pred(self, pred):
        self.manual_frame = self.manual_frame - 1
        pred_index = np.argmax(pred)
        prob = np.exp(pred) / sum(np.exp(pred))
        self.pred = self.index2phase[pred_index]
        add_log = [datetime.datetime.now(), self.trainee.text(), self.mentor.text(), self.bed.text(), self.pred]
        add_log += prob.tolist()
        self.log_data.append(add_log)
        pred_percentages = ((np.exp(pred)/np.exp(pred).sum()) * 100).tolist()
        self.phase1_prob.setValue(pred_percentages[0])
        self.phase2_prob.setValue(pred_percentages[1])
        self.phase3_prob.setValue(pred_percentages[2])
        self.phase4_prob.setValue(pred_percentages[3])
        self.phase_probs = {'Marking': self.phase1_prob,
                            'Injection': self.phase2_prob,
                            'Dissection': self.phase3_prob,
                            'Idle': self.phase4_prob}
        states = [False] * 4
        states[pred_index] = True
        self.phase1_state.setChecked(states[0])
        self.phase2_state.setChecked(states[1])
        self.phase3_state.setChecked(states[2])
        self.phase4_state.setChecked(states[3])

    def onButtonClickStart(self):
        self.startButton.setStyleSheet("background-color: DarkGrey;")
        self.startButton.setEnabled(False)
        self.stopButton.setStyleSheet("background-color: DarkGreen;")
        self.stopButton.setEnabled(True)
        self.canvas.clear()
        self.flag = True
        self.WORKING = True
        self.video = False
        # video_file_name = os.path.join(self.save_folder, self.e1.text().replace(":", "_").replace(" ",
        #                                                                                           "-") + "_" + self.start_time.replace(
        #     ":", "-") + ".avi")
        # self.output_video = cv2.VideoWriter(video_file_name, self.CODEC, self.stream_fps,
        #                                     (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        self.start_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_file = os.path.join(self.save_folder, self.trainee.text() + "_" + self.start_time.replace(":",
                                                                                                              "-") + ".csv")
        self.startTime = datetime.datetime.now()
        lineEdits = self.trainLabel.findChildren(QLineEdit)
        for lineEdit in lineEdits:
            if lineEdit.objectName() == 'TraineeInput':
                if self.surgeons.findText(lineEdit.text()) == -1 and lineEdit.text() != '':
                    self.surgeons.addItem(lineEdit.text())
                    self.surgeons.setCurrentIndex(-1)


    def onButtonClickStop(self):
        self.startButton.setStyleSheet("background-color: DarkGreen;")
        self.startButton.setEnabled(True)
        self.stopButton.setStyleSheet("background-color: DarkGrey;")
        self.stopButton.setEnabled(False)
        self.flag = False
        self.WORKING = False
        # self.video = False
        # self.output_video.release()
        # self.thread.terminate()
        # self.thread.wait(1)

        self.pred = "--"
        self.init_status()
        self.save_log_data()
        # self.DisplayTrainee.setText("--")

    def init_status(self):
        self.WORKING = False
        self.INIT = False
        self.TRAINEE = "NONE"
        self.PAUSE_times = 0
        self.INDEPENDENT = True
        self.HELP = False
        self.STATUS = "--"

    def pick_color(self):
        color = QColorDialog.getColor()
        idx = self.labelSelector.currentIndex()
        if color.isValid() and idx != -1:
            self.canvas.brush_color = color
            self.color.pop(idx)
            self.color.insert(idx, color)
            label = self.findChild(QLabel, f"Object{idx + 1}")
            label.setStyleSheet(f"background-color: {color.name()}")

    def resizeEvent(self, event):
        old_pos = self.frameGeometry().getRect()
        curr_x = old_pos[2]
        curr_y = old_pos[3]
        self.size = QSize(curr_x - 25 - 500, curr_y - 65 - 250)
        self.pos = QSize(500, 250)
        # self.usbVideo.setGeometry(QtCore.QRect(500, 250, curr_x-25-500, curr_y-65-250))
        # self.canvas.label_mask.scaled(curr_x-25-500, curr_y-65-250, QtCore.Qt.KeepAspectRatio)

    def setupTrainer(self):
        e1 = QLabel('Mentor:')
        e1.setObjectName("Mentor")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        e1.setStyleSheet("color:white;")
        e2 = QLabel('Lesion Location:')
        e2.setObjectName("LesionLocation")
        e2.setFont(QFont("Arial", 16, QFont.Bold))
        e2.setStyleSheet("color:white;")
        e3 = QLineEdit()
        e3.setFixedHeight(35)
        e3.setObjectName("MentorInput")
        e3.setStyleSheet("background-color: white;color:black")
        e3.setFont(QFont("Arial", 14))
        e3.setText("Jeffery")
        e3.setAlignment(Qt.AlignCenter)
        self.mentor = e3
        e4 = QLineEdit()
        e4.setFixedHeight(35)
        e4.setObjectName("LesionLocationInput")
        e4.setStyleSheet("background-color: white;color:black")
        e4.setFont(QFont("Arial", 14))
        e4.setText("Stomach")
        e4.setAlignment(Qt.AlignCenter)
        self.lesion = e4
        e5 = QLabel("Trainee:")
        e5.setObjectName("Trainee")
        e5.setFont(QFont("Arial", 16, QFont.Bold))
        e5.setStyleSheet("color:white;")
        e6 = QLabel('Bed:')
        e6.setObjectName("Bed")
        e6.setFont(QFont("Arial", 16, QFont.Bold))
        e6.setStyleSheet("color:white;")
        e7 = QLineEdit()
        e7.setFixedHeight(35)
        e7.setObjectName("TraineeInput")
        e7.setStyleSheet("background-color: white;color:black")
        e7.setFont(QFont("Arial", 14))
        e7.setText("John")
        e7.setAlignment(Qt.AlignCenter)
        self.trainee = e7
        e8 = QLineEdit()
        e8.setFixedHeight(35)
        e8.setObjectName("BedInput")
        e8.setStyleSheet("background-color: white;color:black")
        e8.setFont(QFont("Arial", 14))
        e8.setText("1")
        e8.setAlignment(Qt.AlignCenter)
        self.bed = e8
        e = QGridLayout()
        e.addWidget(e1, 0, 0)
        e.addWidget(e2, 0, 1)
        e.addWidget(self.mentor, 1, 0)
        e.addWidget(self.lesion, 1, 1)
        e.addWidget(e5, 2, 0)
        e.addWidget(e6, 2, 1)
        e.addWidget(self.trainee, 3, 0)
        e.addWidget(self.bed, 3, 1)
        self.trainLabel.setLayout(e)

    def load_image(self):
        if self.video:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Please stop the video before loading a new image")
            msg.setWindowTitle("Warning")
            msg.exec_()
        else:
            self.DisplayVideo.clear()
            file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.xpm *.jpg *.bmp)')
            self.filepath = file_path
            if file_path:
                image = QImage(file_path).scaled(self.size.width(), self.size.height(), Qt.KeepAspectRatio)
                self.canvas.image = image.convertToFormat(QImage.Format_RGB32)
                self.canvas.beginLabel = self.beginLabel
                self.canvas.endLabel = self.endLabel
                self.label_mask = QImage(image.size(), QImage.Format_ARGB32)
                self.label_mask.fill(Qt.transparent)
                self.canvas.label_mask = self.label_mask
                self.canvas.setPixmap(QPixmap.fromImage(image))
                self.canvas.setGeometry(QtCore.QRect(500, 250, self.size.width(), self.size.height()))
                self.labelSelector.setCurrentIndex(-1)

    def on_combobox_changed(self):
        if self.labelSelector.currentIndex() != -1:
            idx = self.labelSelector.currentIndex()
            if self.ru != 0:
                self.canvas.brush_color = self.color[idx]
                self.last_idx.append(self.labelSelector.currentIndex())
                if str(self.labelSelector.currentIndex()) not in self.count_image.keys():
                    self.count_image[str(self.labelSelector.currentIndex())] = 0
                self.image_count.append(self.count_image[str(self.labelSelector.currentIndex())])
                self.bbb = len(self.image_count)
            self.ru += 1

    def onSliderValueChanged(self, value):
        self.canvas.brush_size = value
        self.thickness.setText(str(value))

    def onLineEditsChanged(self, text):
        if text.isnumeric() and int(text) <= self.slider.maximum() and int(text) >= self.slider.minimum():
            self.canvas.brush_size = int(text)
            self.slider.setValue(int(text))
            self.pbFlag = True
        elif text.isnumeric() and int(text) > self.slider.maximum():
            self.thickness.setText(str(self.slider.maximum()))
            self.canvas.brush_size = self.slider.maximum()
            self.slider.setValue(self.slider.maximum())
            self.pbFlag = True
        elif text.isnumeric() and int(text) < self.slider.minimum():
            self.thickness.setText(str(self.slider.minimum()))
            self.canvas.brush_size = self.slider.minimum()
            self.slider.setValue(self.slider.minimum())
            self.pbFlag = True
        else:
            self.canvas.brush_size = 0
            self.pbFlag = True

    def onButtonPaint(self):
        idx = self.labelSelector.currentIndex()
        if idx != -1:
            self.canvas.brush_color = self.color[idx]
            self.canvas.erase = False

    def onButtonErase(self):
        self.canvas.erase = True
        # self.canvas.brush_color.setAlphaF(0.01)

    def onButtonSave(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Label Mask', '', 'Images (*.png)')
        if file_path:
            self.canvas.label_mask.save(file_path)

    def onButtonAddLabel(self):
        new_color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        new_r, new_g, new_b = hex_to_rgb(new_color)
        new_color1 = QColor(new_b, new_g, new_r)
        if new_color1 not in self.color:
            dlg = CustomMB(self.labelSelector)
            dlg.exec_()
            if dlg.finished:
                self.labelSelector.addItem(dlg.getText())
                self.color.append(new_color1)
                count = self.labelSelector.count()
                self.labelSelector.setCurrentIndex(count - 1)
                e1_group = QGroupBox()
                e1_group.setObjectName(f"Object{count}Group")
                e1_group.setStyleSheet(f"QGroupBox#Object{count}Group" + "{border:0;}")
                e1_button = QRadioButton()
                e1_button.setChecked(True)
                e1_button.setStyleSheet(
                    "QRadioButton"
                    "{"
                    "color : green;"
                    "}"
                    "QRadioButton::indicator"
                    "{"
                    "width : 20px;"
                    "height : 20px;"
                    "}")
                e1_button.setObjectName(f"Object{count}Button")
                e1 = QLabel()
                e1.setObjectName(f"Object{count}")
                e1.setStyleSheet(f"background-color : {new_color};")
                e1.setFixedWidth(20)
                e1.setFixedHeight(20)
                e2 = QLabel(dlg.getText().title())
                e2.setObjectName(f"Object{count}Name")
                e2.setFont(QFont("Arial", 14, QFont.Bold))
                e2.setStyleSheet("color:black;")
                hbox_1 = QHBoxLayout()
                hbox_1.setObjectName(f"Object{count}Layout")
                hbox_1.addWidget(e1_button)
                hbox_1.addWidget(e1)
                hbox_1.addWidget(e2)
                hbox_1.setAlignment(Qt.AlignLeft)
                e1_group.setLayout(hbox_1)
                self.elayout.addWidget(e1_group, count - 1, 0)
                self.elayout.setAlignment(Qt.AlignLeft)

    def setupCaseInformation(self):
        num_widget = self.verticalLayout.count()
        while num_widget > 0:
            widget = self.verticalLayout.itemAt(num_widget - 1).widget()
            if widget.objectName() == 'CaseInformation':
                break
            num_widget -= 1
        vlayout = QVBoxLayout(widget)
        vlayout.setObjectName("CaseInformationVlayout")
        self.e1 = QLabel('Patient ID:')
        self.e1.setObjectName("PID")
        self.e1.setFont(QFont("Arial", 16, QFont.Bold))
        self.e1.setStyleSheet("color:white;")
        vlayout.addWidget(self.e1)
        hlayout = QHBoxLayout()
        e2 = QLineEdit()
        e2.setFixedHeight(35)
        e2.setFixedWidth(180)
        e2.setObjectName("PID1")
        e2.setStyleSheet("background: white;border-radius:5px;color: black")
        e2.setAlignment(Qt.AlignCenter)
        e2.setFont(QFont("Arial", 14))
        e2.setText("Jenny")
        hlayout.addWidget(e2)
        e3 = QFrame()
        e3.setFrameShape(QFrame.HLine)
        e3.setFrameShadow(QFrame.Plain)
        e3.setLineWidth(2)
        e3.setObjectName("PIDSpace")
        hlayout.addWidget(e3)
        e4 = QLineEdit()
        e4.setFixedHeight(35)
        e4.setFixedWidth(180)
        e4.setObjectName("PID2")
        e4.setStyleSheet("background: white;border-radius:5px;color:black")
        e4.setAlignment(Qt.AlignCenter)
        e4.setFont(QFont("Arial", 14))
        e4.setText("798xxx(x)")
        hlayout.addWidget(e4)
        hlayout.setSpacing(15)
        vlayout.addLayout(hlayout)
        e5 = QLabel('Date:')
        e5.setObjectName("PIDDate")
        e5.setFont(QFont("Arial", 16, QFont.Bold))
        e5.setStyleSheet("color:white;")
        vlayout.addWidget(e5)
        hlayout1 = QHBoxLayout()
        e6 = QLineEdit()
        e6.setFixedHeight(35)
        e6.setFixedWidth(105)
        e6.setObjectName("PIDDateYear")
        e6.setStyleSheet("background-color: white;border-radius:5px;color:black")
        e6.setAlignment(Qt.AlignCenter)
        e6.setFont(QFont("Arial", 14))
        e6.setText("2023")
        hlayout1.addWidget(e6)
        e7 = QFrame()
        e7.setFrameShape(QFrame.HLine)
        e7.setFrameShadow(QFrame.Plain)
        e7.setLineWidth(2)
        e7.setObjectName("PIDDateSpace")
        hlayout1.addWidget(e7)
        e8 = QLineEdit()
        e8.setFixedHeight(35)
        e8.setFixedWidth(105)
        e8.setObjectName("PIDDateMonth")
        e8.setStyleSheet("background-color: white;border-radius:5px;color:black")
        e8.setAlignment(Qt.AlignCenter)
        e8.setFont(QFont("Arial", 14))
        e8.setText("Nov")
        hlayout1.addWidget(e8)
        e9 = QFrame()
        e9.setFrameShape(QFrame.HLine)
        e9.setFrameShadow(QFrame.Plain)
        e9.setLineWidth(2)
        e9.setObjectName("PIDDateSpace1")
        hlayout1.addWidget(e9)
        e10 = QLineEdit()
        e10.setFixedHeight(35)
        e10.setFixedWidth(105)
        e10.setObjectName("PIDDateDay")
        e10.setStyleSheet("background-color: white;border-radius:5px;color: black")
        e10.setAlignment(Qt.AlignCenter)
        e10.setFont(QFont("Arial", 14))
        e10.setText("10")
        hlayout1.addWidget(e10)
        hlayout1.setSpacing(15)
        vlayout.addLayout(hlayout1)

    def setupPhaseRecog(self):
        num_widget = self.verticalLayout.count()
        while num_widget > 0:
            widget = self.verticalLayout.itemAt(num_widget - 1).widget()
            if widget.objectName() == 'PhaseRecognition':
                break
            num_widget -= 1
        widget1 = QtWidgets.QWidget(self)
        e1 = QLabel('Idle')
        e1.setObjectName("Idle")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        e1.setStyleSheet("color:white;")
        self.phase1_state = QRadioButton()
        self.phase1_state.setChecked(False)
        # e2.setTristate(True)
        self.phase1_state.setObjectName("IdleCheck")
        self.phase1_state.setStyleSheet("QRadioButton"
                                        "{"
                                        "color : green;"
                                        "}"
                                        "QRadioButton::indicator"
                                        "{"
                                        "width : 20px;"
                                        "height : 20px;"
                                        "}")
        self.phase1_prob = QProgressBar()
        self.phase1_prob.setObjectName("IdleProgress")
        self.phase1_prob.setStyleSheet(DEFAULT_STYLE)
        self.phase1_prob.setValue(10)
        self.phase1_prob.setTextVisible(False)
        e4 = QLabel('Marking')
        e4.setObjectName("Marking")
        e4.setFont(QFont("Arial", 16, QFont.Bold))
        e4.setStyleSheet("color:white;")
        self.phase2_state = QRadioButton()
        self.phase2_state.setChecked(False)
        # e2.setTristate(True)
        self.phase2_state.setObjectName("MarkingCheck")
        self.phase2_state.setStyleSheet("QRadioButton"
                                        "{"
                                        "color : green;"
                                        "}"
                                        "QRadioButton::indicator"
                                        "{"
                                        "width : 20px;"
                                        "height : 20px;"
                                        "}")
        self.phase2_prob = QProgressBar()
        self.phase2_prob.setObjectName("MarkingProgress")
        self.phase2_prob.setStyleSheet(DEFAULT_STYLE)
        self.phase2_prob.setValue(15)
        self.phase2_prob.setTextVisible(False)
        e7 = QLabel('Injection')
        e7.setObjectName("Injection")
        e7.setFont(QFont("Arial", 16, QFont.Bold))
        e7.setStyleSheet("color:white;")
        self.phase3_state = QRadioButton()
        self.phase3_state.setChecked(True)
        # e2.setTristate(True)
        self.phase3_state.setObjectName("InjectionCheck")
        self.phase3_state.setStyleSheet("QRadioButton"
                                        "{"
                                        "color : green;"
                                        "}"
                                        "QRadioButton::indicator"
                                        "{"
                                        "width : 20px;"
                                        "height : 20px;"
                                        "}")
        self.phase3_prob = QProgressBar()
        self.phase3_prob.setObjectName("InjectionProgress")
        self.phase3_prob.setStyleSheet(DEFAULT_STYLE)
        self.phase3_prob.setValue(70)
        self.phase3_prob.setTextVisible(False)
        e10 = QLabel('Dissection')
        e10.setObjectName("Dissection")
        e10.setFont(QFont("Arial", 16, QFont.Bold))
        e10.setStyleSheet("color:white;")
        self.phase4_state = QRadioButton()
        self.phase4_state.setChecked(False)
        # e2.setTristate(True)
        self.phase4_state.setObjectName("DissectionCheck")
        self.phase4_state.setStyleSheet("QRadioButton"
                                        "{"
                                        "color : green;"
                                        "}"
                                        "QRadioButton::indicator"
                                        "{"
                                        "width : 20px;"
                                        "height : 20px;"
                                        "}")
        self.phase4_prob = QProgressBar()
        self.phase4_prob.setObjectName("DissectionProgress")
        self.phase4_prob.setStyleSheet(DEFAULT_STYLE)
        self.phase4_prob.setValue(10)
        self.phase4_prob.setTextVisible(False)
        egrid = QGridLayout()
        egrid.addWidget(e1, 0, 0)
        egrid.addWidget(self.phase1_state, 0, 1)
        egrid.addWidget(self.phase1_prob, 0, 2)
        egrid.addWidget(e4, 1, 0)
        egrid.addWidget(self.phase2_state, 1, 1)
        egrid.addWidget(self.phase2_prob, 1, 2)
        egrid.addWidget(e7, 2, 0)
        egrid.addWidget(self.phase3_state, 2, 1)
        egrid.addWidget(self.phase3_prob, 2, 2)
        egrid.addWidget(e10, 3, 0)
        egrid.addWidget(self.phase4_state, 3, 1)
        egrid.addWidget(self.phase4_prob, 3, 2)
        egrid.setAlignment(Qt.AlignCenter)
        widget1.setLayout(egrid)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: black;")

        a1 = QLabel("Predicted phase")
        a1.setFont(QFont("Arial", 16, QFont.Bold))
        phase_pred = max(self.phase_probs)
        a2 = QLabel(phase_pred)
        a2.setFont(QFont("Arial", 26, QFont.Bold))
        a2.setStyleSheet("color: darkblue;")
        VLayout = QtWidgets.QVBoxLayout()
        VLayout.addWidget(widget1)
        VLayout.addWidget(line)
        VLayout.addWidget(a1, alignment=Qt.AlignCenter)
        VLayout.addWidget(a2, alignment=Qt.AlignCenter)
        widget.setLayout(VLayout)
        

    def e1_button_toggled(self):
        pass

    def countTime(self):
        if self.flag:
            current_time = datetime.datetime.now()
            diff_time = (current_time - self.startTime).total_seconds()
            hour = int(diff_time // 3600)
            minute = int(diff_time % 3600 // 60)
            second = int(diff_time % 60)
            self.duraHour.setText('{:02d}'.format(hour))
            self.duraMinute.setText('{:02d}'.format(minute))
            self.duraSecond.setText('{:02d}'.format(second))

    def generateReport(self):
        self.reportButton.setEnabled(False)
        report_path = generate_report("../Records")
        fullpath = os.path.realpath("./reports")
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(fullpath))
        self.reportButton.setEnabled(True)

    def setupAnalytics(self):
        num_widget = self.verticalLayout.count()
        while num_widget > 0:
            widget = self.verticalLayout.itemAt(num_widget - 1).widget()
            if widget.objectName() == 'OnlineAnalytics':
                break
            num_widget -= 1
        # Create the gray rectangles
        gray_rect1 = QFrame()
        gray_rect1.setFrameShape(QFrame.StyledPanel)
        gray_rect1.setStyleSheet("background-color: gray;")
        gray_rect1.setMinimumWidth(300) 
        gray_rect2 = QFrame()
        gray_rect2.setFrameShape(QFrame.StyledPanel)
        gray_rect2.setStyleSheet("background-color: gray;")
        gray_rect2.setMinimumWidth(300) 
        gray_rect3 = QFrame()
        gray_rect3.setFrameShape(QFrame.StyledPanel)
        gray_rect3.setStyleSheet("background-color: gray;")
        gray_rect3.setMinimumWidth(300) 
        gray_rect4 = QFrame()
        gray_rect4.setFrameShape(QFrame.StyledPanel)
        gray_rect4.setStyleSheet("background-color: gray;")
        gray_rect4.setMinimumWidth(300) 

        # Create the labels
        e1 = QLabel('Time:')
        e1.setObjectName("Time")
        e1.setFont(QFont("Arial", 16, QFont.Bold))
        e1.setStyleSheet("color:white;")
        e3 = QLabel('Mentor:')
        e3.setObjectName("Mentor")
        e3.setFont(QFont("Arial", 16, QFont.Bold))
        e3.setStyleSheet("color:white;")
        e5 = QLabel('Trainee:')
        e5.setObjectName("Trainee")
        e5.setFont(QFont("Arial", 16, QFont.Bold))
        e5.setStyleSheet("color:white;")
        e7 = QLabel('NT-index:')
        e7.setObjectName("NT-index")
        e7.setFont(QFont("Arial", 16, QFont.Bold))
        e7.setStyleSheet("color:white;")

        # Create the layout for each row
        row1_layout = QHBoxLayout()
        row1_layout.addWidget(e1)
        row1_layout.addWidget(gray_rect1)
        row2_layout = QHBoxLayout()
        row2_layout.addWidget(e3)
        row2_layout.addWidget(gray_rect2)
        row3_layout = QHBoxLayout()
        row3_layout.addWidget(e5)
        row3_layout.addWidget(gray_rect3)
        row4_layout = QHBoxLayout()
        row4_layout.addWidget(e7)
        row4_layout.addWidget(gray_rect4)

        # Create the main vertical layout
        VLayout = QVBoxLayout()
        VLayout.addLayout(row1_layout)
        VLayout.addLayout(row2_layout)
        VLayout.addLayout(row3_layout)
        VLayout.addLayout(row4_layout)

        # Set the layout for the widget
        widget.setLayout(VLayout)


    def setVLayout(self, name, percent):
        widget = QtWidgets.QLabel(name.title())
        widget.setObjectName(name.title().replace(' ', '') + 'Title')
        widget.setFont(QFont('Arial', 18, QFont.Bold))
        widget.setStyleSheet("color: white;")
        self.verticalLayout.addWidget(widget, 4)
        widget1 = QtWidgets.QWidget()
        widget1.setObjectName(name.title().replace(' ', ''))
        widget1.setAttribute(Qt.WA_StyledBackground, True)
        widget1.setStyleSheet(
            f"QWidget#{name.title().replace(' ', '')}" + "{background-color: #99ccff; border-radius:5px;}")
        self.verticalLayout.addWidget(widget1, 21)
        gapWidget = QtWidgets.QWidget()
        gapWidget.setFixedWidth(50)  # Set the desired width for the gap
        gapWidget.setObjectName(name.title().replace(' ', '')+'Gap')
        self.verticalLayout.addWidget(gapWidget, 5)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("iPhaser", "AI-Endo"))

    def get_frame_size(self):
        capture = cv2.VideoCapture(0)  # TODO: change camera

        # Default resolutions of the frame are obtained (system dependent)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # fps = int(capture.get(cv2.CAP_PROP_FPS))
        fps = 30
        capture.release()
        return frame_width, frame_height, fps

    def save_log_data(self):
        datas = zip(*self.log_data)
        data_dict = {}
        # [datetime.datetime.now(), self.trainee.text(), self.mentor.text(), self.bed.text(), self.pred]
        names = ["Time", "Trainee", "Trainer", "Bed", "Prediction", "Phase idle", "Phase marking", "Phase injection", "Phase dissection"]
        for name, data in zip(names, datas):
            data_dict[name] = list(data)
        pd_log = pd.DataFrame.from_dict(data_dict)
        curent_date_time = "_" + datetime.datetime.now().strftime("%H-%M-%S") + ".csv"
        pd_log.to_csv(self.log_file.replace(".csv", curent_date_time), index=False, header=True)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-s", default=False, action='store_true', help="Whether save predictions")
    parse.add_argument("-q", default=False, action='store_true', help="Display video")
    parse.add_argument("--cfg", default="test_camera", type=str)

    cfg = parse.parse_args()
    cfg = ParserUse(cfg.cfg, "camera").add_args(cfg)

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    ui = Ui_iPhaser()
    ui.setupUi(cfg)
    ui.show()
    app.installEventFilter(ui)
    sys.exit(app.exec_())