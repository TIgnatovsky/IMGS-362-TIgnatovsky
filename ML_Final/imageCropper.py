import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QPushButton, QVBoxLayout, QWidget, QShortcut, QLineEdit, QLabel
)
from PyQt5.QtGui import QPixmap, QPen, QPainter, QKeySequence
from PyQt5.QtCore import Qt, QRectF, QPointF
from PIL import Image

OUTPUT_SIZE = (64, 64)


class GraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.pixmap_item = None
        self.image_path = None

        self.start_point = None
        self.current_rect_item = None
        self.crop_rect = None

        self.scale_factor = 1.0

    def load_image(self, path):
        self.scene.clear()

        pixmap = QPixmap(path)

        # Rotate 90° clockwise
        transform = QPainter().transform()
        transform.rotate(90)
        pixmap = pixmap.transformed(transform)

        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.image_path = path
        self.setSceneRect(QRectF(pixmap.rect()))
        self.resetTransform()
        self.scale_factor = 1.0

    # Zoom with mouse wheel
    def wheelEvent(self, event):
        zoom_in = 1.01
        zoom_out = 0.99

        if event.angleDelta().y() > 0:
            factor = zoom_in
        else:
            factor = zoom_out

        self.scale(factor, factor)
        self.scale_factor *= factor

    # Middle mouse = pan
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake_event = event
            super().mousePressEvent(fake_event)
            return

        if event.button() == Qt.LeftButton and self.pixmap_item:
            self.setDragMode(QGraphicsView.NoDrag)
            self.start_point = self.mapToScene(event.pos())

            if self.current_rect_item:
                self.scene.removeItem(self.current_rect_item)

            self.current_rect_item = QGraphicsRectItem()
            pen = QPen(Qt.red, 2)
            self.current_rect_item.setPen(pen)
            self.scene.addItem(self.current_rect_item)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.start_point:
            current_point = self.mapToScene(event.pos())
            square_point = self.get_square_point(self.start_point, current_point)

            rect = QRectF(self.start_point, square_point).normalized()
            self.current_rect_item.setRect(rect)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)
            return

        if event.button() == Qt.LeftButton and self.start_point:
            current_point = self.mapToScene(event.pos())
            square_point = self.get_square_point(self.start_point, current_point)

            self.crop_rect = QRectF(self.start_point, square_point).normalized()
            self.start_point = None

        super().mouseReleaseEvent(event)

    def get_square_point(self, start, current):
        dx = current.x() - start.x()
        dy = current.y() - start.y()

        side = min(abs(dx), abs(dy))

        end_x = start.x() + side if dx > 0 else start.x() - side
        end_y = start.y() + side if dy > 0 else start.y() - side

        return QPointF(end_x, end_y)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt Crop Tool (Zoom + Pan)")

        self.view = GraphicsView()

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        save_btn = QPushButton("Save Crop")
        save_btn.clicked.connect(self.save_crop)

        shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        shortcut.activated.connect(self.save_crop)

        shortcut2 = QShortcut(QKeySequence(Qt.Key_Enter), self)
        shortcut2.activated.connect(self.save_crop)

        self.filename_input = QLineEdit()
        self.filename_input.setPlaceholderText("Enter base filename (e.g. sprite)")

        self.ascii_input = QLineEdit()
        self.ascii_input.setPlaceholderText("Enter value of ASCII character (otherwise it is 32)")

        self.dir_label = QLabel("No output directory selected")

        dir_btn = QPushButton("Select Output Folder")
        dir_btn.clicked.connect(self.select_output_dir)

        layout = QVBoxLayout()
        layout.addWidget(load_btn)
        layout.addWidget(save_btn)
        layout.addWidget(self.filename_input)
        layout.addWidget(self.ascii_input)
        layout.addWidget(dir_btn)
        layout.addWidget(self.dir_label)
        layout.addWidget(self.view)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.output_dir = ""
        self.file_counter = 1
        self.last_base_name = ""

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.view.load_image(path)

    def select_output_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = folder
            self.dir_label.setText(f"Output: {folder}")

    def save_crop(self):
        if not self.view.crop_rect or not self.view.image_path:
            print("No selection made")
            return

        if not self.output_dir:
            print("Select an output directory")
            return

        base_name = self.filename_input.text().strip()
        if not base_name:
            print("Enter a base filename")
            return

        # Reset counter if base name changed
        if base_name != self.last_base_name:
            self.file_counter = 1
            self.last_base_name = base_name

        ascii_text = self.ascii_input.text().strip()
        ascii_value = int(ascii_text) if ascii_text.isdigit() else 32

        rect = self.view.crop_rect

        x1 = int(rect.left())
        y1 = int(rect.top())
        x2 = int(rect.right())
        y2 = int(rect.bottom())

        img = Image.open(self.view.image_path)
        img = img.rotate(-90, expand=True)

        cropped = img.crop((x1, y1, x2, y2))
        resized = cropped.resize(OUTPUT_SIZE, Image.LANCZOS)

        filename = f"{base_name}_{self.file_counter:03d}.png"
        full_path = os.path.join(self.output_dir, filename)

        resized.save(full_path)
        print(f"Saved to {full_path}")

        # Append to CSV
        csv_path = os.path.join(os.path.dirname(self.output_dir), "labels.csv")
        with open(csv_path, "a") as f:
            f.write(f"{filename},{ascii_value}\n")

        print(f"Appended label: {filename},{ascii_value}")

        self.file_counter += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec_())