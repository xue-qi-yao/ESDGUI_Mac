from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QSizePolicy

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Main layout
        main_layout = QVBoxLayout()

        # Top-level container widget
        container_widget = QWidget()
        container_widget.setLayout(main_layout)

        # First row
        first_row_layout = QHBoxLayout()
        first_row_layout.addWidget(QPushButton("Button 1"))
        first_row_layout.addWidget(QPushButton("Button 2"))
        first_row_layout.addWidget(QPushButton("Button 3"))

        # Second row
        second_row_layout = QHBoxLayout()
        second_row_layout.addWidget(QPushButton("Button 4"))
        second_row_layout.addWidget(QPushButton("Button 5"))
        second_row_layout.addWidget(QPushButton("Button 6"))

        # Add rows to main layout
        main_layout.addLayout(first_row_layout)
        main_layout.addLayout(second_row_layout)

        # Set size policies for container and buttons
        container_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        for button in container_widget.findChildren(QPushButton):
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.setCentralWidget(container_widget)
        self.setWindowTitle("Auto Fit Example")
        self.show()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    app.exec_()