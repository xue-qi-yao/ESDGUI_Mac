import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QVBoxLayout, QPushButton, QSizePolicy, QColorDialog, QWidget, QComboBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint
import cv2
from PyQt5.QtCore import QTimer

class Canvas(QLabel):
    def __init__(self, label_mask=None, image=None, parent=None):
        QLabel.__init__(self, parent)
        self.setMouseTracking(True)
        self.drawing = False
        self.last_point = QPoint()
        self.brush_size = 5
        self.brush_color = QColor(0,0,0,0)
        self.label_mask = label_mask
        self.erase = False
        if image is None:
            self.image = QImage()
        else:
            self.image = image
        self.copy_image = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    
    def mouseMoveEvent(self, event):
        if self.drawing:
            if self.erase:
            
                pixmap = self.pixmap()
                painter = QPainter(pixmap)
                color = QColor(self.image.pixel(event.pos()))
                #rint(color.red(), color.green(), color.blue())
                painter.setPen(QPen(color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                #painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.drawLine(self.last_point, event.pos())
                painter.end()
                
                
                label_painter = QPainter(self.label_mask)
                label_painter.setPen(QPen(Qt.transparent, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                label_painter.setCompositionMode(QPainter.CompositionMode_Clear)
                label_painter.drawLine(self.last_point, event.pos())
                label_painter.end()

                
                self.last_point = event.pos()
                self.update()
            else:
                '''
                painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(self.last_point, event.pos())
                '''
                pixmap = self.pixmap()
                
                
                label_painter = QPainter(self.label_mask)
                label_painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                label_painter.drawLine(self.last_point, event.pos())
                label_painter.end()
                
                painter = QPainter(pixmap)
                painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                #painter.setCompositionMode(QPainter.CompositionMode_Clear)
                painter.drawLine(self.last_point, event.pos())
                painter.end()
                
                self.last_point = event.pos()
                self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Image Labeling Tool')

        self.label_mask = QImage()
        self.image = QImage()
        self.canvas = Canvas(self.label_mask, self.image)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.video = None

        self.frame_index = 0
        self.frame_index_label = QLabel()
        self.selected_fps = 30  # 默认值为2fps

        load_button = QPushButton('Load Image')
        load_button.clicked.connect(self.load_image)

        save_button = QPushButton('Save Label Mask')
        save_button.clicked.connect(self.save_label_mask)

        pick_color_button = QPushButton('Pick Color')
        pick_color_button.clicked.connect(self.pick_color)
        
        load_video_button = QPushButton('Load Video')
        load_video_button.clicked.connect(self.load_video)
        pause_button = QPushButton('Pause/Resume')
        pause_button.clicked.connect(self.pause_resume)
        
        # 创建下拉菜单并添加FPS选项
        self.video_capture = None
        self.fps_combobox = QComboBox()
        self.fps_combobox.addItem('2 fps', 2)
        self.fps_combobox.addItem('5 fps', 5)
        self.fps_combobox.addItem('10 fps', 10)
        self.fps_combobox.addItem('30 fps', 30)
        self.fps_combobox.currentIndexChanged.connect(self.update_fps)



        layout = QVBoxLayout()
        layout.addWidget(self.fps_combobox)
        layout.addWidget(self.canvas)
        layout.addWidget(load_button)
        layout.addWidget(save_button)
        layout.addWidget(pick_color_button)
        layout.addWidget(load_video_button)
        layout.addWidget(pause_button)
        layout.addWidget(self.frame_index_label)
        


        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Images (*.png *.xpm *.jpg *.bmp)')
        if file_path:
            image = QImage(file_path)
            self.label_mask = QImage(image.size(), QImage.Format_ARGB32)
            self.label_mask.fill(Qt.transparent)
            self.canvas.label_mask = self.label_mask
            self.canvas.image = image
            self.canvas.copy_image = image.copy()
            self.canvas.setPixmap(QPixmap.fromImage(image))
    
            
    def update_fps(self, index):
        # 获取选定的FPS值
        self.selected_fps = self.fps_combobox.itemData(index)
        print(self.selected_fps)
            
    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', 'Videos (*.avi *.mp4 *.mkv)')
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            self.timer = QTimer()
            self.timer.timeout.connect(self.display_video_frame)
            self.timer.start(int(1000 / self.selected_fps))
    
    def update_frame_index_label(self, displayed_frame_index):
        self.frame_index_label.setText(f"Now the Frame is: {displayed_frame_index*self.selected_fps}")
        
    def display_video_frame(self):
        if self.video_capture is None:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.timer.stop()
            return

        if self.frame_index % self.selected_fps == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.label_mask = QImage(image.size(), QImage.Format_ARGB32)
            self.label_mask.fill(Qt.transparent)
            self.canvas.label_mask = self.label_mask
            self.canvas.image = image
            self.canvas.copy_image = image.copy()
            self.canvas.setPixmap(QPixmap.fromImage(image))
            self.update_frame_index_label(displayed_frame_index=self.frame_index // self.selected_fps)

        self.frame_index += 1


            
    def next_frame(self):
        if self.video is not None:
            ret, frame = self.video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                self.label_mask = QImage(image.size(), QImage.Format_ARGB32)
                self.label_mask.fill(Qt.transparent)
                self.canvas.label_mask = self.label_mask
                self.canvas.image = image
                self.canvas.copy_image = image.copy()
                self.canvas.setPixmap(QPixmap.fromImage(image))
                self.frame_index +=1
                self.update_frame_index_label()
            else:
                self.timer.stop()
                self.video.release()
                self.video = None

    def pause_resume(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(int(1000 / self.selected_fps))

    def save_label_mask(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save Label Mask', '', 'Images (*.png)')
        if file_path:
            self.label_mask.save(file_path)

    def pick_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.canvas.brush_color = color

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())