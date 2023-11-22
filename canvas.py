import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QVBoxLayout, QPushButton, QSizePolicy, QGraphicsScene, QGraphicsView, QColorDialog, QWidget
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QRect

class Canvas(QLabel):
    def __init__(self, label_mask, image, parent=None):
        QLabel.__init__(self, parent)
        self.setMouseTracking(True)
        self.drawing = False
        self.prompt = False
        self.beginLabel = QLabel()
        self.endLabel = QLabel()
        self.last_point = QPoint()
        self.brush_size = 5
        self.brush_color = QColor(0,0,0,0)
        self.label_mask = label_mask
        self.erase = False
        self.image = image
        self.begin = QPoint()
        self.end = QPoint()
        self.copy_image = self.image.copy()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            if self.prompt:
                self.begin = event.pos()
    
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
            elif self.prompt:
                pass
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
            if self.prompt:
                self.end = event.pos()
        if self.prompt:
            qp = QPainter(self.pixmap())
            qp.setPen(QPen(Qt.red, 2, Qt.SolidLine))  
            qp.drawRect(QRect(self.begin, self.end))
            self.beginLabel.setText(f"Begin: {self.begin.x(), self.begin.y()}")
            self.endLabel.setText(f"End: {self.end.x(), self.end.y()}")
            self.update()
            
            #mask = prompt_sam_predict(self.image_path, self.box, self.device)
            

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Image Labeling Tool')

        self.label_mask = QImage()
        self.canvas = Canvas(self.label_mask)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        load_button = QPushButton('Load Image')
        load_button.clicked.connect(self.load_image)

        save_button = QPushButton('Save Label Mask')
        save_button.clicked.connect(self.save_label_mask)

        pick_color_button = QPushButton('Pick Color')
        pick_color_button.clicked.connect(self.pick_color)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(load_button)
        layout.addWidget(save_button)
        layout.addWidget(pick_color_button)

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
            self.canvas.setPixmap(QPixmap.fromImage(image))

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
