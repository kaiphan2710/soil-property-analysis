import os
os.environ['QT_API'] = 'pyside6'

# Import GDAL first on Windows/Conda to register Library\bin DLL search paths
try:
    from osgeo import gdal
except ImportError:
    pass

import sys
import numpy as np
import urllib.request
import urllib.error
import json
import io
import matplotlib
matplotlib.use('QtAgg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QRectF, Signal, Slot, QEvent, QTimer
from PySide6.QtGui import QPixmap, QImage, QPen, QColor, QPainter, QIcon, QBrush, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSpinBox, QComboBox, QGroupBox,
    QFormLayout, QRadioButton, QButtonGroup, QMessageBox, QGraphicsView,
    QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem, QSplitter,
    QLineEdit, QPinchGesture, QCheckBox, QDialog, QListWidget, QListWidgetItem,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QInputDialog,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsItem
)
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QGraphicsSimpleTextItem
import re
import base64
import configparser
import math

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hyperspectral_client_config.ini")

def encrypt_key(plain_text: str) -> str:
    key = b"microplastic_secret_salt_123!"
    encoded_bytes = plain_text.encode('utf-8')
    xor_bytes = bytearray(b ^ key[i % len(key)] for i, b in enumerate(encoded_bytes))
    return base64.b64encode(xor_bytes).decode('utf-8')

def decrypt_key(cipher_text: str) -> str:
    try:
        key = b"microplastic_secret_salt_123!"
        xor_bytes = base64.b64decode(cipher_text.encode('utf-8'))
        decoded_bytes = bytearray(b ^ key[i % len(key)] for i, b in enumerate(xor_bytes))
        return decoded_bytes.decode('utf-8')
    except Exception:
        return ""

def exception_hook(exctype, value, tb):
    print("Unhandled exception occurred:", exctype, value, flush=True)
    import traceback
    traceback.print_exception(exctype, value, tb)
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = exception_hook

def create_app_icon():
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Outer circle
    painter.setPen(QPen(QColor("#d2d2d7"), 2))
    painter.setBrush(QBrush(QColor("#ffffff")))
    painter.drawEllipse(4, 4, 56, 56)
    
    # Stylized spectrum bands
    colors = ["#ff3b30", "#ff9500", "#34c759", "#007aff", "#af52de"]
    for i, col in enumerate(colors):
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(col)))
        painter.drawRect(14 + i * 7, 20, 6, 24)
        
    painter.end()
    return QIcon(pixmap)

class MplCanvas(FigureCanvas):
    """
    Embedded Matplotlib canvas to display spectral curves in the GUI.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f5f5f7')
        self.axes = fig.add_subplot(111)
        self.axes.set_facecolor('white')
        self.axes.tick_params(colors='#1d1d1f')
        self.axes.xaxis.label.set_color('#1d1d1f')
        self.axes.yaxis.label.set_color('#1d1d1f')
        self.axes.title.set_color('#1d1d1f')
        self.axes.grid(True, color='#e5e5ea', linestyle='--')
        super().__init__(fig)
        fig.set_layout_engine('tight')
        self.setParent(parent)

class InteractiveGraphicsView(QGraphicsView):
    pixel_clicked = Signal(int, int)  # x, y
    roi_changed = Signal(int, int, int, int, float)  # x, y, w, h, angle
    roi_resized = Signal(int, int)  # w, h
    roi_selected = Signal(int)  # index of selected ROI
    roi_created = Signal(int, int, int, int)  # x, y, w, h
    roi_label_clicked = Signal(int)  # index of label clicked
    interaction_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.viewport().grabGesture(Qt.PinchGesture)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        
        self.pixmap_item = None
        self.rect_item = None
        self.handle_item = None
        self.rot_handle_item = None
        self.rot_line_item = None
        self.roi_items = []
        self.local_rois = []
        self.label_bounds = []
        
        self.interaction_mode = 'fixed_roi'
        self.fixed_w = 512
        self.fixed_h = 512
        self.is_square_locked = False
        
        self.is_drawing = False
        self.is_panning = False
        self.is_resizing = False
        self.is_dragging_roi = False
        self.is_rotating = False
        
        self.last_pan_pos = None
        self.start_point = None
        
        self.roi_pen = QPen(QColor(0, 113, 227, 220), 2)
        self.roi_pen.setStyle(Qt.DashLine)
        self.roi_pen.setCosmetic(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
    def set_image(self, qimage):
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(QRectF(qimage.rect()))
        self.rect_item = None
        self.handle_item = None
        self.rot_handle_item = None
        self.rot_line_item = None
        self.roi_items = []
        self.is_drawing = False
        
    def set_interaction_mode(self, mode, fixed_w=512, fixed_h=512):
        self.interaction_mode = mode
        self.fixed_w = fixed_w
        self.fixed_h = fixed_h
        
        if self.rect_item:
            if self.rect_item.scene():
                self.scene.removeItem(self.rect_item)
            self.rect_item = None
        if self.handle_item:
            if self.handle_item.scene():
                self.scene.removeItem(self.handle_item)
            self.handle_item = None
            
    def zoom(self, factor):
        current_scale = self.transform().m11()
        new_scale = current_scale * factor
        if 0.05 < new_scale < 50.0:
            self.scale(factor, factor)
            self.update_cosmetic_scales()

    def update_cosmetic_scales(self):
        view_scale = self.transform().m11()
        scale = 1.0 / view_scale if view_scale > 0 else 1.0
        
        if self.handle_item:
            self.handle_item.setScale(scale)
        if hasattr(self, 'rot_handle_item') and self.rot_handle_item:
            self.rot_handle_item.setScale(scale)
            
        for idx, bg_item in self.label_bounds:
            try:
                bg_item.setScale(scale)
            except Exception:
                pass

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            angle = event.angleDelta().y()
            factor = 1.15 if angle > 0 else 0.85
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.zoom(factor)
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
            self.zoom(1.15)
            event.accept()
        elif event.key() == Qt.Key_Minus:
            self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
            self.zoom(0.85)
            event.accept()
        else:
            super().keyPressEvent(event)

    def viewportEvent(self, event):
        if event.type() == QEvent.Gesture:
            return self.gestureEvent(event)
        return super().viewportEvent(event)

    def gestureEvent(self, event):
        pinch = event.gesture(Qt.PinchGesture)
        if pinch:
            self.pinchTriggered(pinch)
            return True
        return False

    def pinchTriggered(self, gesture):
        change_flags = gesture.changeFlags()
        if change_flags & QPinchGesture.ScaleFactorChanged:
            factor = gesture.scaleFactor()
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.zoom(factor)

    def update_handle_position(self):
        if self.rect_item and self.handle_item:
            rect = self.rect_item.rect()
            w, h = rect.width(), rect.height()
            
            # Dynamic offset based on box size
            offset = max(10.0, min(w, h) * 0.1)
            
            from PySide6.QtWidgets import QGraphicsEllipseItem
            if isinstance(self.rect_item, QGraphicsEllipseItem):
                # Put the handle at the bottom-right border of the ellipse (45 degrees)
                hx = w * 0.85355
                hy = h * 0.85355
                self.handle_item.setPos(hx, hy)
                
                if hasattr(self, 'rot_handle_item') and self.rot_handle_item:
                    # Place rotation handle further out from the 45-degree border
                    rx = (w / 2.0) + (w / 2.0 + offset) * 0.7071
                    ry = (h / 2.0) + (h / 2.0 + offset) * 0.7071
                    self.rot_handle_item.setPos(rx, ry)
                if hasattr(self, 'rot_line_item') and self.rot_line_item:
                    lx1 = (w / 2.0) + (w / 2.0) * 0.7071
                    ly1 = (h / 2.0) + (h / 2.0) * 0.7071
                    lx2 = (w / 2.0) + (w / 2.0 + offset) * 0.7071
                    ly2 = (h / 2.0) + (h / 2.0 + offset) * 0.7071
                    self.rot_line_item.setLine(lx1, ly1, lx2, ly2)
            else:
                hx = w
                hy = h
                self.handle_item.setPos(hx, hy)
                
                if hasattr(self, 'rot_handle_item') and self.rot_handle_item:
                    # Place rotation handle at (w + offset, h + offset)
                    rx = w + offset
                    ry = h + offset
                    self.rot_handle_item.setPos(rx, ry)
                if hasattr(self, 'rot_line_item') and self.rot_line_item:
                    self.rot_line_item.setLine(w, h, w + offset, h + offset)

    def draw_rois(self, rois, selected_index=-1):
        self.local_rois = rois
        self.selected_index = selected_index
        
        # Remove previously drawn ROI parent items (children auto-removed by Qt)
        for item in self.roi_items:
            try:
                if item.scene() is not None:
                    self.scene.removeItem(item)
            except Exception:
                pass
        self.roi_items = []
        self.label_bounds = []
        
        colors_roi = ["#007aff", "#34c759", "#af52de", "#5856d6", "#5ac8fa"]
        colors_target = ["#ff3b30", "#ff9500", "#ff2d55", "#e5c158", "#e558c1", "#58e5c1", "#c158e5"]
        roi_count = 0
        target_name_to_color = {}
        unique_target_count = 0

        for idx, roi in enumerate(rois):
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            angle = roi.get("angle", 0.0)
            name = roi["name"]
            r_type = roi.get("type", "roi")
            
            if r_type == "target":
                norm_name = name.strip().lower()
                if norm_name not in target_name_to_color:
                    target_name_to_color[norm_name] = colors_target[unique_target_count % len(colors_target)]
                    unique_target_count += 1
                color_str = target_name_to_color[norm_name]
            else:
                color_str = colors_roi[roi_count % len(colors_roi)]
                roi_count += 1
            
            base_color = QColor(color_str)
            
            is_selected = (idx == selected_index)
            if is_selected:
                # Highlight active selected ROI with dashed border
                pen = QPen(base_color, 3.0, Qt.DashLine)
                brush_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 25)
            else:
                pen = QPen(base_color, 1.5, Qt.SolidLine)
                brush_color = QColor(0, 0, 0, 0)
            pen.setCosmetic(True)
                
            # 1. Shape / Bounding box
            shape_type = roi.get("shape_type", "Rectangle")
            if shape_type in ("Ellipse", "Circle"):
                from PySide6.QtWidgets import QGraphicsEllipseItem
                item = QGraphicsEllipseItem(0, 0, w, h)
                item.setTransformOriginPoint(w / 2.0, h / 2.0)
            else:
                item = QGraphicsRectItem(0, 0, w, h)
                item.setTransformOriginPoint(0, 0)
                
            item.setPos(x, y)
            item.setRotation(angle)
            item.setPen(pen)
            item.setBrush(QBrush(brush_color))
            self.scene.addItem(item)
            # Only track the parent item; children are auto-cleaned when parent is removed
            self.roi_items.append(item)
            
            # 2. Text Label (created as children of shape item, NOT via scene.addSimpleText)
            from PySide6.QtGui import QPainterPath, QFontMetrics
            from PySide6.QtWidgets import QGraphicsPathItem
            
            base_font_size = 9
            min_side = min(w, h)
            if min_side < 100:
                font_size = max(5, int(base_font_size * (min_side / 100.0)))
            else:
                font_size = base_font_size

            font = QFont("Segoe UI", font_size, QFont.Bold)
            fm = QFontMetrics(font)
            
            px = max(2, int(5 * (font_size / 9.0)))
            py = max(1, int(2 * (font_size / 9.0)))
            
            # Calculate background dimensions
            text_rect = fm.boundingRect(name)
            bg_w = text_rect.width() + px * 2
            bg_h = fm.height() + py * 2
            
            if shape_type in ("Ellipse", "Circle"):
                bg_x = w * 0.14645
                bg_y = h * 0.14645 - bg_h - 2
            else:
                bg_x = 0
                bg_y = -bg_h - 2
                
            # Create background rect
            bg_color = base_color
            bg_item = QGraphicsRectItem(0, 0, bg_w, bg_h)
            bg_item.setPos(bg_x, bg_y)
            bg_item.setParentItem(item)
            bg_item.setPen(Qt.NoPen)
            bg_item.setBrush(QBrush(bg_color))
            
            # Draw text as path relative to bg_item local origin
            path = QPainterPath()
            ascent = fm.ascent()
            path.addText(px, py + ascent, font, name)
            
            text_item = QGraphicsPathItem(path)
            text_item.setParentItem(bg_item)
            text_item.setBrush(QBrush(QColor("#ffffff")))
            text_item.setPen(Qt.NoPen)
            text_item.setZValue(bg_item.zValue() + 1)
            
            self.label_bounds.append((idx, bg_item))
            
        self.update_cosmetic_scales()

    def mousePressEvent(self, event):
        try:
            self._mousePressEvent(event)
        except Exception as e:
            print("CRITICAL ERROR IN mousePressEvent:", e, flush=True)
            import traceback
            traceback.print_exc()

    def _mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if not self.pixmap_item:
            super().mousePressEvent(event)
            return
            
        scene_pos = self.mapToScene(event.pos())
        
        # 1. Check if clicked on a label
        for idx, bg_item in self.label_bounds:
            local_pt = bg_item.mapFromScene(scene_pos)
            if bg_item.rect().contains(local_pt):
                self.roi_label_clicked.emit(idx)
                event.accept()
                return

        # Helper to check collision with an annotation
        def hit_test(roi):
            rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
            angle = roi.get("angle", 0.0)
            transform = QTransform()
            transform.translate(rx, ry)
            shape_type = roi.get("shape_type", "Rectangle")
            if shape_type in ("Ellipse", "Circle"):
                transform.translate(rw / 2.0, rh / 2.0)
                transform.rotate(angle)
                transform.translate(-rw / 2.0, -rh / 2.0)
            else:
                transform.rotate(angle)
            inv_transform, ok = transform.inverted()
            if ok:
                local_pt = inv_transform.map(scene_pos)
                return QRectF(0, 0, rw, rh).contains(local_pt)
            return False

        # 2. Check if clicked on any target annotation to select it (excluding the currently active/selected target)
        clicked_roi_idx = -1
        if self.local_rois:
            selected_idx = getattr(self, "selected_index", -1)
            for idx, roi in reversed(list(enumerate(self.local_rois))):
                if idx != selected_idx and roi.get("type", "roi") == "target" and hit_test(roi):
                    clicked_roi_idx = idx
                    break

        if clicked_roi_idx != -1:
            self.roi_selected.emit(clicked_roi_idx)

        # 3. Check if clicked on active edit handles or inside active box
        if self.rect_item and self.handle_item:
            if hasattr(self, 'rot_handle_item') and self.rot_handle_item:
                if self.rot_handle_item.sceneBoundingRect().contains(scene_pos):
                    self.is_rotating = True
                    if hasattr(self, 'selected_index') and 0 <= self.selected_index < len(self.roi_items):
                        self.roi_items[self.selected_index].setVisible(False)
                    event.accept()
                    return
            
            if self.handle_item.sceneBoundingRect().contains(scene_pos):
                self.is_resizing = True
                if hasattr(self, 'selected_index') and 0 <= self.selected_index < len(self.roi_items):
                    self.roi_items[self.selected_index].setVisible(False)
                event.accept()
                return
            elif self.rect_item.rect().contains(self.rect_item.mapFromScene(scene_pos)):
                self.is_dragging_roi = True
                if hasattr(self, 'selected_index') and 0 <= self.selected_index < len(self.roi_items):
                    self.roi_items[self.selected_index].setVisible(False)
                self.drag_offset_x = scene_pos.x() - self.rect_item.x()
                self.drag_offset_y = scene_pos.y() - self.rect_item.y()
                event.accept()
                return

        # 4. Check if clicked inside any other Petri Dish ROI bounding box to select it (excluding the currently selected one)
        clicked_roi_idx = -1
        if self.local_rois:
            selected_idx = getattr(self, "selected_index", -1)
            for idx, roi in reversed(list(enumerate(self.local_rois))):
                if idx != selected_idx and roi.get("type", "roi") != "target" and hit_test(roi):
                    clicked_roi_idx = idx
                    break

        if clicked_roi_idx != -1:
            self.roi_selected.emit(clicked_roi_idx)
            if self.rect_item and self.rect_item.rect().contains(self.rect_item.mapFromScene(scene_pos)):
                self.is_dragging_roi = True
                if hasattr(self, 'selected_index') and 0 <= self.selected_index < len(self.roi_items):
                    self.roi_items[self.selected_index].setVisible(False)
                self.drag_offset_x = scene_pos.x() - self.rect_item.x()
                self.drag_offset_y = scene_pos.y() - self.rect_item.y()
                event.accept()
                return


        if self.local_rois:
            self.roi_selected.emit(-1)
        super().mousePressEvent(event)
        
    def mouseDoubleClickEvent(self, event):
        if not self.pixmap_item:
            super().mouseDoubleClickEvent(event)
            return
        scene_pos = self.mapToScene(event.pos())
        img_rect = self.pixmap_item.boundingRect()
        if img_rect.contains(scene_pos):
            x, y = int(scene_pos.x()), int(scene_pos.y())
            self.pixel_clicked.emit(x, y)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        try:
            self._mouseMoveEvent(event)
        except Exception as e:
            print("CRITICAL ERROR IN mouseMoveEvent:", e, flush=True)
            import traceback
            traceback.print_exc()

    def _mouseMoveEvent(self, event):
        if hasattr(self, 'is_panning') and self.is_panning:
            delta = event.pos() - self.last_pan_pos
            self.last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        if not self.pixmap_item:
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        img_rect = self.pixmap_item.boundingRect()
        
        x_clamp = max(0, min(scene_pos.x(), img_rect.width()))
        y_clamp = max(0, min(scene_pos.y(), img_rect.height()))
        
        if self.is_rotating and self.rect_item:
            rect = self.rect_item.rect()
            w, h = rect.width(), rect.height()
            from PySide6.QtWidgets import QGraphicsEllipseItem
            from PySide6.QtCore import QPointF
            
            if isinstance(self.rect_item, QGraphicsEllipseItem):
                cx = self.rect_item.x() + w / 2.0
                cy = self.rect_item.y() + h / 2.0
                vx = scene_pos.x() - cx
                vy = scene_pos.y() - cy
                mouse_angle = math.degrees(math.atan2(vy, vx))
                angle = mouse_angle - 45.0
            else:
                cx = self.rect_item.x()
                cy = self.rect_item.y()
                vx = scene_pos.x() - cx
                vy = scene_pos.y() - cy
                mouse_angle = math.degrees(math.atan2(vy, vx))
                base_angle = math.degrees(math.atan2(h + 20, w + 20))
                angle = mouse_angle - base_angle
                
            self.rect_item.setRotation(angle)
            self.roi_changed.emit(int(self.rect_item.x()), int(self.rect_item.y()), int(w), int(h), angle)
            event.accept()
            return
            
        elif self.is_resizing and self.rect_item:
            local_pos = self.rect_item.mapFromScene(scene_pos)
            
            from PySide6.QtWidgets import QGraphicsEllipseItem
            if isinstance(self.rect_item, QGraphicsEllipseItem):
                new_w = max(10, int(local_pos.x() / 0.85355))
                new_h = max(10, int(local_pos.y() / 0.85355))
            else:
                new_w = max(10, int(local_pos.x()))
                new_h = max(10, int(local_pos.y()))
            
            if self.is_square_locked or (event.modifiers() & Qt.ShiftModifier):
                side = max(new_w, new_h)
                new_w = new_h = side
                
            self.rect_item.setRect(0, 0, new_w, new_h)
            if isinstance(self.rect_item, QGraphicsEllipseItem):
                self.rect_item.setTransformOriginPoint(new_w / 2.0, new_h / 2.0)
            else:
                self.rect_item.setTransformOriginPoint(0, 0)
                
            self.update_handle_position()
            self.roi_resized.emit(new_w, new_h)
            self.roi_changed.emit(int(self.rect_item.x()), int(self.rect_item.y()), new_w, new_h, self.rect_item.rotation())
            event.accept()
            return

        elif self.is_dragging_roi and self.rect_item:
            rect = self.rect_item.rect()
            rw, rh = rect.width(), rect.height()
            nx = scene_pos.x() - self.drag_offset_x
            ny = scene_pos.y() - self.drag_offset_y
            
            self.rect_item.setPos(nx, ny)
            self.update_handle_position()
            self.roi_changed.emit(int(nx), int(ny), int(rw), int(rh), self.rect_item.rotation())
            event.accept()
            return

        if self.rect_item and self.handle_item and not event.buttons():
            if self.handle_item.sceneBoundingRect().contains(scene_pos):
                self.viewport().setCursor(Qt.SizeFDiagCursor)
            elif hasattr(self, 'rot_handle_item') and self.rot_handle_item and self.rot_handle_item.sceneBoundingRect().contains(scene_pos):
                self.viewport().setCursor(Qt.PointingHandCursor)
            elif self.rect_item.rect().contains(self.rect_item.mapFromScene(scene_pos)):
                self.viewport().setCursor(Qt.SizeAllCursor)
            else:
                self.viewport().unsetCursor()
            
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.is_panning = False
            self.unsetCursor()
            event.accept()
            return

        self.is_resizing = False
        self.is_dragging_roi = False
        self.is_rotating = False
        self.is_drawing = False
        
        super().mouseReleaseEvent(event)
        self.interaction_finished.emit()


class RemoteFileDialog(QDialog):
    def __init__(self, parent, current_dir="", list_dir_func=None):
        super().__init__(parent)
        self.setWindowTitle("Remote File Browser")
        self.resize(750, 500)
        self.list_dir_func = list_dir_func
        self.current_dir = current_dir
        self.parent_dir = ""
        self.selected_path = None
        self.raw_items = []

        self.init_ui()
        self.load_directory(self.current_dir)

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Path and Navigation bar
        nav_layout = QHBoxLayout()
        self.btn_up = QPushButton("Up")
        self.btn_up.setFixedWidth(60)
        self.btn_up.clicked.connect(self.navigate_up)
        
        self.txt_path = QLineEdit()
        self.txt_path.setReadOnly(False)
        self.txt_path.returnPressed.connect(self.on_path_entered)
        
        nav_layout.addWidget(self.btn_up)
        nav_layout.addWidget(self.txt_path)
        layout.addLayout(nav_layout)

        # Sorting row
        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Sort By:"))
        self.combo_sort = QComboBox()
        self.combo_sort.addItems([
            "Name (Alphabetical)", 
            "Modification Time (Newest First)", 
            "Modification Time (Oldest First)",
            "Creation Time (Newest First)",
            "Creation Time (Oldest First)"
        ])
        self.combo_sort.currentIndexChanged.connect(self.apply_sorting_and_display)
        sort_layout.addWidget(self.combo_sort)
        sort_layout.addStretch()
        layout.addLayout(sort_layout)
        
        # File list (QTreeWidget for columns layout)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setColumnCount(3)
        self.tree_widget.setHeaderLabels(["Name", "Date Modified", "Date Created"])
        self.tree_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.tree_widget.header().setSectionResizeMode(QHeaderView.Interactive)
        self.tree_widget.header().setStretchLastSection(True)
        self.tree_widget.setColumnWidth(0, 350)
        self.tree_widget.setColumnWidth(1, 160)
        self.tree_widget.setColumnWidth(2, 160)
        layout.addWidget(self.tree_widget)
        
        # Dialog buttons
        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("Select")
        self.btn_select.clicked.connect(self.on_select)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def load_directory(self, path):
        try:
            data = self.list_dir_func(path)
            self.current_dir = data["current_dir"]
            self.txt_path.setText(self.current_dir)
            self.parent_dir = data.get("parent_dir", "")
            
            self.btn_up.setEnabled(bool(self.parent_dir))
            
            self.raw_items = data["items"]
            self.apply_sorting_and_display()
                
        except Exception as e:
            if path != "~" and path != "":
                try:
                    self.load_directory("~")
                    return
                except Exception:
                    pass
            QMessageBox.critical(self, "Error", f"Failed to list directory:\n{str(e)}")

    def on_path_entered(self):
        target = self.txt_path.text().strip()
        if target:
            self.load_directory(target)

    def apply_sorting_and_display(self):
        sort_type = self.combo_sort.currentText()
        
        # Sort rule: directories always first, then by selected option
        if "Name" in sort_type:
            self.raw_items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        elif "Modification Time (Newest" in sort_type:
            self.raw_items.sort(key=lambda x: (not x["is_dir"], -x.get("mtime", 0), x["name"].lower()))
        elif "Modification Time (Oldest" in sort_type:
            self.raw_items.sort(key=lambda x: (not x["is_dir"], x.get("mtime", 0), x["name"].lower()))
        elif "Creation Time (Newest" in sort_type:
            self.raw_items.sort(key=lambda x: (not x["is_dir"], -x.get("ctime", 0), x["name"].lower()))
        elif "Creation Time (Oldest" in sort_type:
            self.raw_items.sort(key=lambda x: (not x["is_dir"], x.get("ctime", 0), x["name"].lower()))

        self.tree_widget.clear()
        
        import datetime
        for item in self.raw_items:
            is_dir = item["is_dir"]
            emoji = "📁 " if is_dir else "📄 "
            display_name = f"{emoji}{item['name']}"
            
            mtime_val = item.get("mtime")
            ctime_val = item.get("ctime")
            
            mtime_str = ""
            if mtime_val:
                try:
                    mtime_str = datetime.datetime.fromtimestamp(mtime_val).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    pass
            
            ctime_str = ""
            if ctime_val:
                try:
                    ctime_str = datetime.datetime.fromtimestamp(ctime_val).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    pass
            
            tree_item = QTreeWidgetItem([display_name, mtime_str, ctime_str])
            tree_item.setData(0, Qt.UserRole, item)
            
            if is_dir:
                font = tree_item.font(0)
                font.setBold(True)
                tree_item.setFont(0, font)
                
            self.tree_widget.addTopLevelItem(tree_item)

    def on_item_double_clicked(self, item):
        item_data = item.data(0, Qt.UserRole)
        if item_data["is_dir"]:
            self.load_directory(item_data["path"])
        else:
            self.selected_path = item_data["path"]
            self.accept()

    def navigate_up(self):
        if self.parent_dir:
            self.load_directory(self.parent_dir)

    def on_select(self):
        current_item = self.tree_widget.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a file first.")
            return
        item_data = current_item.data(0, Qt.UserRole)
        if item_data["is_dir"]:
            QMessageBox.warning(self, "Warning", "Please select a .hdr or .npz file, not a directory.")
            return
        self.selected_path = item_data["path"]
        self.accept()


class MicroplasticClientApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Remote Hyperspectral Data Analyzer (Client)")
        self.setWindowIcon(create_app_icon())
        
        self.metadata = None
        self.current_roi = None  # (x, y, w, h)
        self.is_connected = False
        self.last_remote_dir = ""
        self.rois = []
        self.selected_roi_index = -1
        self.last_target_name = ""
        
        self.init_ui()
        self.apply_light_theme()
        self.set_authenticated(False)
        self.load_config()

    def closeEvent(self, event):
        self.save_config()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            focused = QApplication.focusWidget()
            from PySide6.QtWidgets import QLineEdit, QSpinBox
            if not isinstance(focused, (QLineEdit, QSpinBox)):
                self.delete_selected_roi()
                event.accept()
                return
        super().keyPressEvent(event)

    def load_config(self):
        config = configparser.ConfigParser()
        if os.path.exists(CONFIG_FILE):
            try:
                config.read(CONFIG_FILE, encoding='utf-8')
                if "Connection" in config:
                    conn = config["Connection"]
                    self.txt_server_ip.setText(conn.get("server_ip", "127.0.0.1"))
                    self.txt_server_port.setText(conn.get("server_port", "17474"))
                    enc_key = conn.get("api_key", "")
                    if enc_key:
                        self.txt_api_key.setText(decrypt_key(enc_key))
                    else:
                        self.txt_api_key.setText("microplastic_secret")
                    self.last_remote_dir = conn.get("last_remote_dir", "")
                    
                    # Load default_shape, defaulting to Circle
                    default_shape = conn.get("default_shape", "Circle")
                    self.combo_shape_type.blockSignals(True)
                    self.combo_shape_type.setCurrentText(default_shape)
                    self.combo_shape_type.blockSignals(False)
                    self.view.is_square_locked = (default_shape in ("Square", "Circle"))
            except Exception as e:
                print(f"Error loading config: {e}")

    def save_config(self):
        config = configparser.ConfigParser()
        config["Connection"] = {
            "server_ip": self.txt_server_ip.text().strip(),
            "server_port": self.txt_server_port.text().strip(),
            "api_key": encrypt_key(self.txt_api_key.text().strip()),
            "last_remote_dir": getattr(self, "last_remote_dir", ""),
            "default_shape": self.combo_shape_type.currentText()
        }
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                config.write(f)
        except Exception as e:
            print(f"Error saving config: {e}")
        
    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # 1. Connection Group
        conn_group = QGroupBox("Server Connection")
        conn_layout = QVBoxLayout(conn_group)
        
        ip_port_layout = QHBoxLayout()
        self.txt_server_ip = QLineEdit("127.0.0.1")
        self.txt_server_port = QLineEdit("17474")
        self.txt_server_port.setMaximumWidth(80)
        ip_port_layout.addWidget(QLabel("IP:"))
        ip_port_layout.addWidget(self.txt_server_ip)
        ip_port_layout.addWidget(QLabel("Port:"))
        ip_port_layout.addWidget(self.txt_server_port)
        conn_layout.addLayout(ip_port_layout)
        
        key_layout = QHBoxLayout()
        self.txt_api_key = QLineEdit("microplastic_secret")
        self.txt_api_key.setPlaceholderText("Enter Security Key")
        self.txt_api_key.setEchoMode(QLineEdit.Password)
        
        self.btn_toggle_key = QPushButton("Show")
        self.btn_toggle_key.setFixedWidth(50)
        self.btn_toggle_key.setStyleSheet("background-color: #515154; padding: 2px; font-size: 11px; height: 18px;")
        def toggle_key_echo():
            if self.txt_api_key.echoMode() == QLineEdit.Password:
                self.txt_api_key.setEchoMode(QLineEdit.Normal)
                self.btn_toggle_key.setText("Hide")
            else:
                self.txt_api_key.setEchoMode(QLineEdit.Password)
                self.btn_toggle_key.setText("Show")
        self.btn_toggle_key.clicked.connect(toggle_key_echo)
        
        key_layout.addWidget(QLabel("Key:"))
        key_layout.addWidget(self.txt_api_key)
        key_layout.addWidget(self.btn_toggle_key)
        conn_layout.addLayout(key_layout)
        
        self.btn_connect = QPushButton("Connect & Authenticate")
        self.btn_connect.clicked.connect(self.connect_to_server)
        conn_layout.addWidget(self.btn_connect)
        
        left_layout.addWidget(conn_group)

        # Hook input changes to reset connection state
        self.txt_server_ip.textChanged.connect(self.reset_connection_state)
        self.txt_server_port.textChanged.connect(self.reset_connection_state)
        self.txt_api_key.textChanged.connect(self.reset_connection_state)
 
        # 2. File Control Group
        file_group = QGroupBox("Remote Data Source")
        file_layout = QVBoxLayout(file_group)
        self.txt_hdr_path = QLineEdit("")
        self.txt_hdr_path.setPlaceholderText("Enter server-side absolute path to .hdr file")
        
        path_buttons_layout = QHBoxLayout()
        self.btn_select_remote_path = QPushButton("Browse Remote...")
        self.btn_select_remote_path.clicked.connect(self.browse_remote_path)
        self.btn_select_remote_path.setStyleSheet("background-color: #0071e3; padding: 4px 8px; font-weight: bold;")
        
        self.btn_select_local_path = QPushButton("Browse Local (To get path)")
        self.btn_select_local_path.clicked.connect(self.browse_local_path)
        self.btn_select_local_path.setStyleSheet("background-color: #515154; padding: 4px 8px; font-size: 11px;")
        
        path_buttons_layout.addWidget(self.btn_select_remote_path)
        path_buttons_layout.addWidget(self.btn_select_local_path)
        
        self.lbl_file_info = QLabel("Please connect to the server first.")
        self.lbl_file_info.setStyleSheet("font-size: 11px; color: #515154; margin: 0;")
        self.lbl_file_info.setWordWrap(True)
        
        file_layout.addWidget(self.txt_hdr_path)
        file_layout.addLayout(path_buttons_layout)
        file_layout.addWidget(self.lbl_file_info)
        left_layout.addWidget(file_group)
        
        # 3. RGB Visualizer Group
        from PySide6.QtWidgets import QSlider
        rgb_group = QGroupBox("RGB Visualization")
        rgb_layout = QFormLayout(rgb_group)
        
        # Red row setup
        layout_r = QHBoxLayout()
        self.slider_r = QSlider(Qt.Horizontal)
        self.spin_r = QSpinBox()
        self.spin_r.setFixedWidth(60)
        self.slider_r.valueChanged.connect(self.spin_r.setValue)
        self.spin_r.valueChanged.connect(self.slider_r.setValue)
        layout_r.addWidget(self.slider_r)
        layout_r.addWidget(self.spin_r)
        
        # Green row setup
        layout_g = QHBoxLayout()
        self.slider_g = QSlider(Qt.Horizontal)
        self.spin_g = QSpinBox()
        self.spin_g.setFixedWidth(60)
        self.slider_g.valueChanged.connect(self.spin_g.setValue)
        self.spin_g.valueChanged.connect(self.slider_g.setValue)
        layout_g.addWidget(self.slider_g)
        layout_g.addWidget(self.spin_g)
        
        # Blue row setup
        layout_b = QHBoxLayout()
        self.slider_b = QSlider(Qt.Horizontal)
        self.spin_b = QSpinBox()
        self.spin_b.setFixedWidth(60)
        self.slider_b.valueChanged.connect(self.spin_b.setValue)
        self.spin_b.valueChanged.connect(self.slider_b.setValue)
        layout_b.addWidget(self.slider_b)
        layout_b.addWidget(self.spin_b)
        
        rgb_layout.addRow("Red Band:", layout_r)
        rgb_layout.addRow("Green Band:", layout_g)
        rgb_layout.addRow("Blue Band:", layout_b)
        
        self.btn_apply_rgb = QPushButton("Refresh Visuals")
        self.btn_apply_rgb.clicked.connect(self.update_rgb_image)
        self.btn_apply_rgb.setStyleSheet("background-color: #0071e3; color: white; font-weight: bold;")
        rgb_layout.addRow(self.btn_apply_rgb)
        
        left_layout.addWidget(rgb_group)
        
        # 4. Spectral Plot
        plot_group = QGroupBox("Spectral Curve (Double click image to query)")
        plot_layout = QVBoxLayout(plot_group)
        self.canvas = MplCanvas(self, width=5, height=3, dpi=100)
        plot_layout.addWidget(self.canvas)
        left_layout.addWidget(plot_group)
        
        main_splitter.addWidget(left_widget)
        
        # Right Panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # ROI Control Group
        roi_group = QGroupBox("ROI BBox Annotation Manager")
        roi_layout = QHBoxLayout(roi_group)
        roi_layout.setContentsMargins(10, 5, 10, 5)
        roi_layout.setSpacing(10)
        
        # 1. Width
        roi_layout.addWidget(QLabel("W:"))
        self.spin_roi_w = QSpinBox()
        self.spin_roi_w.setRange(1, 10000)
        self.spin_roi_w.setValue(512)
        self.spin_roi_w.setFixedWidth(90)
        self.spin_roi_w.valueChanged.connect(self.handle_fixed_size_change)
        roi_layout.addWidget(self.spin_roi_w)
        
        # 2. Height
        roi_layout.addWidget(QLabel("H:"))
        self.spin_roi_h = QSpinBox()
        self.spin_roi_h.setRange(1, 10000)
        self.spin_roi_h.setValue(512)
        self.spin_roi_h.setFixedWidth(90)
        self.spin_roi_h.valueChanged.connect(self.handle_fixed_size_change)
        roi_layout.addWidget(self.spin_roi_h)
        
        # 3. Shape Type ComboBox
        roi_layout.addWidget(QLabel("Shape:"))
        self.combo_shape_type = QComboBox()
        self.combo_shape_type.addItems(["Rectangle", "Square", "Ellipse", "Circle"])
        self.combo_shape_type.setCurrentText("Circle")
        self.combo_shape_type.currentTextChanged.connect(self.handle_shape_type_change)
        roi_layout.addWidget(self.combo_shape_type)
        
        roi_layout.addSpacing(10)
        
        # 4. Annotation Type ComboBox
        roi_layout.addWidget(QLabel("Type:"))
        self.combo_roi_type = QComboBox()
        self.combo_roi_type.addItems(["Petri Dish (ROI)", "Microplastic (Target)"])
        self.combo_roi_type.setCurrentText("Petri Dish (ROI)")
        self.combo_roi_type.currentTextChanged.connect(self.handle_roi_type_change)
        roi_layout.addWidget(self.combo_roi_type)

        roi_layout.addSpacing(10)
        
        # 5. Selected ROI Name
        roi_layout.addWidget(QLabel("Name:"))
        self.txt_roi_name = QLineEdit("ROI_1")
        self.txt_roi_name.setFixedWidth(100)
        self.txt_roi_name.returnPressed.connect(self.rename_selected_roi)
        self.spin_roi_w.lineEdit().returnPressed.connect(self.rename_selected_roi)
        self.spin_roi_h.lineEdit().returnPressed.connect(self.rename_selected_roi)
        roi_layout.addWidget(self.txt_roi_name)
        
        # 6. Buttons
        self.btn_rename_roi = QPushButton("Modify Selected")
        self.btn_rename_roi.clicked.connect(self.rename_selected_roi)
        self.btn_rename_roi.setStyleSheet("background-color: #515154; color: white;")
        roi_layout.addWidget(self.btn_rename_roi)
        
        self.btn_add_roi = QPushButton("Add ROI")
        self.btn_add_roi.clicked.connect(self.add_new_roi)
        self.btn_add_roi.setStyleSheet("background-color: #34c759; color: white;")
        roi_layout.addWidget(self.btn_add_roi)
        
        self.btn_delete_roi = QPushButton("Delete Selected")
        self.btn_delete_roi.clicked.connect(self.delete_selected_roi)
        self.btn_delete_roi.setStyleSheet("background-color: #ff3b30; color: white;")
        roi_layout.addWidget(self.btn_delete_roi)
        
        self.btn_save_rois = QPushButton("Save Annotations")
        self.btn_save_rois.clicked.connect(self.save_rois_to_server)
        self.btn_save_rois.setStyleSheet("background-color: #0071e3; color: white;")
        roi_layout.addWidget(self.btn_save_rois)
        
        self.btn_export_all_npz = QPushButton("Export All to NPZ")
        self.btn_export_all_npz.clicked.connect(self.export_all_rois_to_npz)
        self.btn_export_all_npz.setStyleSheet("background-color: #515154; color: white;")
        roi_layout.addWidget(self.btn_export_all_npz)
        
        right_layout.addWidget(roi_group)
        
        self.view = InteractiveGraphicsView()
        self.view.pixel_clicked.connect(self.plot_spectral_curve)
        self.view.roi_changed.connect(self.update_roi_info)
        self.view.roi_resized.connect(self.handle_roi_resized_from_view)
        self.view.roi_selected.connect(self.select_roi_by_index)
        self.view.roi_label_clicked.connect(self.prompt_rename_roi)
        self.view.interaction_finished.connect(lambda: QTimer.singleShot(0, self.draw_all_rois_on_scene))
        right_layout.addWidget(self.view)
        
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([320, 960])


    def apply_light_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f7;
            }
            QWidget {
                color: #1d1d1f;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #d2d2d7;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px 0 4px;
                color: #1d1d1f;
            }
            QPushButton {
                background-color: #0071e3;
                border: none;
                color: white;
                padding: 6px 12px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #147ce5;
            }
            QPushButton:pressed {
                background-color: #0066cc;
            }
            QSpinBox, QComboBox, QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d2d2d7;
                padding: 4px;
                border-radius: 5px;
                color: #1d1d1f;
            }
            QSpinBox:focus, QLineEdit:focus {
                border: 1px solid #0071e3;
            }
            QLabel {
                color: #515154;
            }
            QGraphicsView {
                border: 1px solid #d2d2d7;
                background-color: #eaeaea;
            }
            QSplitter::handle {
                background-color: #d2d2d7;
            }
            QSplitter::handle:hover {
                background-color: #0071e3;
            }
        """)
        
    def browse_local_path(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ENVI Header or NPZ File (To copy path)", "", "ENVI Header or NPZ Files (*.hdr *.npz);;All Files (*)"
        )
        if file_path:
            self.txt_hdr_path.setText(os.path.abspath(file_path))
            self.open_remote_file()
 
    def send_post_request(self, endpoint, data, timeout=30):
        ip = self.txt_server_ip.text().strip()
        port = self.txt_server_port.text().strip()
        key = self.txt_api_key.text().strip()
        if not ip.startswith("http://") and not ip.startswith("https://"):
            base_url = f"http://{ip}"
        else:
            base_url = ip
        url = f"{base_url}:{port}{endpoint}"
        try:
            req_body = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=req_body,
                headers={
                    'Content-Type': 'application/json',
                    'X-API-Key': key
                },
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read(), response.info().get_content_type()
        except urllib.error.URLError as e:
            raise Exception(f"Network error connecting to server: {e.reason}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
 
    def connect_to_server(self):
        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("Connecting...")
        QApplication.processEvents()
        try:
            response_data, _ = self.send_post_request("/connect", {})
            result = json.loads(response_data.decode('utf-8'))
            if result.get("status") == "ok":
                self.is_connected = True
                self.set_authenticated(True)
                self.lbl_file_info.setText("Connected to server. Ready to open file.")
                self.btn_connect.setStyleSheet("background-color: #34c759; color: white;")
                self.btn_connect.setText("Connected ✓")
                self.save_config()
            else:
                raise Exception("Invalid server response")
        except Exception as e:
            self.is_connected = False
            self.set_authenticated(False)
            self.lbl_file_info.setText("Connection failed.")
            self.btn_connect.setStyleSheet("background-color: #ff3b30; color: white;")
            self.btn_connect.setText("Connection Failed")
            QMessageBox.critical(self, "Error", f"Failed to connect to server:\n{str(e)}")
        finally:
            self.btn_connect.setEnabled(True)
 
    def reset_connection_state(self):
        if self.is_connected:
            self.is_connected = False
            self.set_authenticated(False)
            self.lbl_file_info.setText("Please connect to the server first.")
            self.btn_connect.setStyleSheet("")
            self.btn_connect.setText("Connect & Authenticate")
 
    def set_authenticated(self, authenticated):
        self.btn_select_remote_path.setEnabled(authenticated)
        self.txt_hdr_path.setEnabled(authenticated)
        self.btn_select_local_path.setEnabled(authenticated)
        if not authenticated:
            self.set_roi_editing_enabled(False)
 
    def set_roi_editing_enabled(self, enabled):
        self.btn_rename_roi.setEnabled(enabled)
        self.btn_add_roi.setEnabled(enabled)
        self.btn_delete_roi.setEnabled(enabled)
        self.btn_save_rois.setEnabled(enabled)
        self.btn_export_all_npz.setEnabled(enabled)
        self.spin_roi_w.setEnabled(enabled)
        self.spin_roi_h.setEnabled(enabled)
        self.combo_shape_type.setEnabled(enabled)
        self.combo_roi_type.setEnabled(enabled)
        self.txt_roi_name.setEnabled(enabled)

    def remote_list_dir(self, path):
        response_data, _ = self.send_post_request("/list_dir", {"path": path})
        return json.loads(response_data.decode('utf-8'))

    def browse_remote_path(self):
        current_dir = getattr(self, "last_remote_dir", "")
        dialog = RemoteFileDialog(self, current_dir=current_dir, list_dir_func=self.remote_list_dir)
        if dialog.exec():
            if dialog.selected_path:
                self.txt_hdr_path.setText(dialog.selected_path)
                self.last_remote_dir = dialog.current_dir
                self.open_remote_file()
        else:
            self.last_remote_dir = dialog.current_dir
        self.save_config()

    def open_remote_file(self):
        hdr_path = self.txt_hdr_path.text().strip()
        if not hdr_path:
            QMessageBox.warning(self, "Warning", "Please specify a remote .hdr or .npz file path.")
            return

        self.lbl_file_info.setText("Connecting...")
        QApplication.processEvents()

        try:
            # Call /open on the server
            response_data, _ = self.send_post_request("/open", {"hdr_path": hdr_path}, timeout=300)
            self.metadata = json.loads(response_data.decode('utf-8'))
            
            num_bands = self.metadata["bands"]
            self.lbl_file_info.setText(
                f"Connected!\n"
                f"Resolution: {self.metadata['width']} x {self.metadata['height']}\n"
                f"Bands: {num_bands}"
            )
            
            # Setup RGB Sliders and SpinBoxes
            for ctrl in [self.slider_r, self.spin_r, self.slider_g, self.spin_g, self.slider_b, self.spin_b]:
                ctrl.setRange(0, num_bands - 1)
                
            default_rgb = self.metadata["default_rgb"]
            self.spin_r.setValue(default_rgb[0])
            self.spin_g.setValue(default_rgb[1])
            self.spin_b.setValue(default_rgb[2])
            
            is_npz = self.metadata.get("is_npz", False)
            if is_npz:
                self.rois = self.metadata.get("annotations", [])
                self.selected_roi_index = 0 if self.rois else -1
                self.set_roi_editing_enabled(False)
                self.update_rgb_image()
            else:
                self.set_roi_editing_enabled(True)
                self.load_rois_from_server()
                self.update_rgb_image()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open dataset on server:\n{str(e)}")
            self.lbl_file_info.setText("Failed to connect/open file.")
            self.metadata = None

    def update_rgb_image(self):
        if not self.metadata:
            return
            
        r_idx = self.spin_r.value()
        g_idx = self.spin_g.value()
        b_idx = self.spin_b.value()
        
        try:
            # Call /rgb to retrieve the image bytes (jpeg)
            img_bytes, content_type = self.send_post_request("/rgb", {
                "r_band": r_idx,
                "g_band": g_idx,
                "b_band": b_idx
            }, timeout=300)
            
            if "image" not in content_type:
                # Error response, parse JSON
                err_info = json.loads(img_bytes.decode('utf-8'))
                raise Exception(err_info.get("error", "Unknown server error"))
                
            # Convert bytes to QImage
            qimage = QImage.fromData(img_bytes)
            self.view.set_image(qimage)
            self.select_roi_by_index(self.selected_roi_index)
            self.handle_mode_change()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to retrieve RGB preview: {str(e)}")

    def handle_mode_change(self):
        # We always stay in fixed_roi mode now
        pass

    def handle_fixed_size_change(self):
        w = self.spin_roi_w.value()
        h = self.spin_roi_h.value()
        
        if self.view.is_square_locked:
            sender = self.sender()
            if sender == self.spin_roi_w and h != w:
                self.spin_roi_h.blockSignals(True)
                self.spin_roi_h.setValue(w)
                self.spin_roi_h.blockSignals(False)
                h = w
            elif sender == self.spin_roi_h and w != h:
                self.spin_roi_w.blockSignals(True)
                self.spin_roi_w.setValue(h)
                self.spin_roi_w.blockSignals(False)
                w = h
                
        if 0 <= self.selected_roi_index < len(self.rois):
            self.rois[self.selected_roi_index]["w"] = w
            self.rois[self.selected_roi_index]["h"] = h
            if self.view.rect_item:
                self.view.rect_item.setRect(0, 0, w, h)
                self.view.update_handle_position()
            self.draw_all_rois_on_scene()

    def handle_shape_type_change(self):
        shape = self.combo_shape_type.currentText()
        is_locked = (shape in ("Square", "Circle"))
        self.view.is_square_locked = is_locked
        
        if is_locked:
            w = self.spin_roi_w.value()
            self.spin_roi_h.blockSignals(True)
            self.spin_roi_h.setValue(w)
            self.spin_roi_h.blockSignals(False)
            
        if 0 <= self.selected_roi_index < len(self.rois):
            roi = self.rois[self.selected_roi_index]
            roi["shape_type"] = shape
            if is_locked:
                roi["h"] = roi["w"]
            self.select_roi_by_index(self.selected_roi_index)
        self.save_config()

    def handle_roi_type_change(self):
        roi_type_text = self.combo_roi_type.currentText()
        new_type = "roi" if "Petri Dish" in roi_type_text else "target"
        
        if 0 <= self.selected_roi_index < len(self.rois):
            roi = self.rois[self.selected_roi_index]
            if roi.get("type", "roi") != new_type:
                roi["type"] = new_type
                name = roi["name"]
                if new_type == "roi" and name.startswith("Target_"):
                    num_part = name[len("Target_"):]
                    if num_part.isdigit():
                        roi["name"] = f"ROI_{num_part}"
                elif new_type == "target" and name.startswith("ROI_"):
                    if hasattr(self, "last_target_name") and self.last_target_name:
                        roi["name"] = self.last_target_name
                    else:
                        num_part = name[len("ROI_"):]
                        if num_part.isdigit():
                            roi["name"] = f"Target_{num_part}"
                
                self.txt_roi_name.setText(roi["name"])
                self.select_roi_by_index(self.selected_roi_index)

    def handle_roi_resized_from_view(self, w, h):
        self.spin_roi_w.blockSignals(True)
        self.spin_roi_h.blockSignals(True)
        self.spin_roi_w.setValue(w)
        self.spin_roi_h.setValue(h)
        self.spin_roi_w.blockSignals(False)
        self.spin_roi_h.blockSignals(False)

    def update_roi_info(self, x, y, w, h, angle=0.0):
        self.current_roi = (x, y, w, h, angle)
        if 0 <= self.selected_roi_index < len(self.rois):
            self.rois[self.selected_roi_index]["x"] = x
            self.rois[self.selected_roi_index]["y"] = y
            self.rois[self.selected_roi_index]["w"] = w
            self.rois[self.selected_roi_index]["h"] = h
            self.rois[self.selected_roi_index]["angle"] = angle
            # Don't redraw during active interaction — causes C++ crash
            # Redraw is deferred to interaction_finished signal
            if not (self.view.is_dragging_roi or self.view.is_resizing or self.view.is_rotating):
                self.draw_all_rois_on_scene()

    def plot_spectral_curve(self, x, y):
        if not self.metadata:
            return
            
        try:
            # Query /pixel endpoint
            response_data, _ = self.send_post_request("/pixel", {"x": x, "y": y})
            result = json.loads(response_data.decode('utf-8'))
            
            if "error" in result:
                raise Exception(result["error"])
                
            spectrum = result["spectrum"]
            
            # Plot spectrum
            self.canvas.axes.clear()
            
            # Use wavelength values if available in metadata
            wavelengths = self.metadata.get("wavelengths", [])
            if wavelengths and len(wavelengths) == len(spectrum):
                self.canvas.axes.plot(wavelengths, spectrum, color='#0071e3', linewidth=1.5)
                self.canvas.axes.set_xlabel("Wavelength (nm)")
            else:
                self.canvas.axes.plot(spectrum, color='#0071e3', linewidth=1.5)
                self.canvas.axes.set_xlabel("Band Index")
                
            self.canvas.axes.set_title(f"Profile: ({x}, {y})", fontsize=11, fontweight='bold')
            self.canvas.axes.set_ylabel("Digital Number (DN)")
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", f"Failed to extract remote spectral curve: {str(e)}")

    def save_roi_to_npz(self):
        if not self.metadata:
            QMessageBox.warning(self, "Warning", "Please open a dataset first.")
            return
            
        if self.selected_roi_index < 0 or self.selected_roi_index >= len(self.rois):
            QMessageBox.warning(self, "Warning", "Please select an ROI first.")
            return
            
        roi = self.rois[self.selected_roi_index]
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        name = roi["name"]
        roi_type = roi.get("type", "roi")
        shape_type = roi.get("shape_type", "Rectangle")
        angle = roi.get("angle", 0.0)
            
        try:
            # Call /save_roi on server
            # Use 300 seconds timeout for raw data file extraction
            response_data, _ = self.send_post_request("/save_roi", {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "name": name,
                "type": roi_type,
                "shape_type": shape_type,
                "angle": angle,
                "all_rois": self.rois
            }, timeout=300)
            result = json.loads(response_data.decode('utf-8'))
            
            if "error" in result:
                raise Exception(result["error"])
                
            QMessageBox.information(
                self, 
                "Success", 
                f"ROI saved successfully on the server:\n{result['message']}\nShape: {result['shape']}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save ROI on server:\n{str(e)}")

    def load_rois_from_server(self):
        hdr_path = self.txt_hdr_path.text().strip()
        if not hdr_path:
            return
        try:
            response_data, _ = self.send_post_request("/get_rois", {"hdr_path": hdr_path})
            self.rois = json.loads(response_data.decode('utf-8'))
            if self.rois:
                self.selected_roi_index = 0
            else:
                self.selected_roi_index = -1
        except Exception as e:
            self.rois = []
            self.selected_roi_index = -1
            QMessageBox.warning(self, "Error", f"Failed to load ROIs from server:\n{str(e)}")

    def save_rois_to_server(self):
        hdr_path = self.txt_hdr_path.text().strip()
        if not hdr_path:
            QMessageBox.warning(self, "Warning", "Please open a dataset first.")
            return
        try:
            self.send_post_request("/save_rois", {
                "hdr_path": hdr_path,
                "rois": self.rois
            })
            QMessageBox.information(self, "Success", "Annotations saved successfully to JSON on the server!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save ROIs to server:\n{str(e)}")

    def select_roi_by_index(self, index):
        if index < 0 or index >= len(self.rois):
            self.selected_roi_index = -1
            self.txt_roi_name.setText("")
            if self.view.rect_item:
                try:
                    if self.view.rect_item.scene():
                        self.view.scene.removeItem(self.view.rect_item)
                except Exception:
                    pass
                self.view.rect_item = None
            if self.view.handle_item:
                try:
                    if self.view.handle_item.scene():
                        self.view.scene.removeItem(self.view.handle_item)
                except Exception:
                    pass
                self.view.handle_item = None
            if hasattr(self.view, 'rot_handle_item') and self.view.rot_handle_item:
                try:
                    if self.view.rot_handle_item.scene():
                        self.view.scene.removeItem(self.view.rot_handle_item)
                except Exception:
                    pass
                self.view.rot_handle_item = None
            if hasattr(self.view, 'rot_line_item') and self.view.rot_line_item:
                try:
                    if self.view.rot_line_item.scene():
                        self.view.scene.removeItem(self.view.rot_line_item)
                except Exception:
                    pass
                self.view.rot_line_item = None
            self.draw_all_rois_on_scene()
            return
        self.selected_roi_index = index
        selected_roi = self.rois[index]
        self.txt_roi_name.setText(selected_roi["name"])
        
        # Select type
        roi_type = selected_roi.get("type", "roi")
        self.combo_roi_type.blockSignals(True)
        if roi_type == "target":
            self.combo_roi_type.setCurrentText("Microplastic (Target)")
        else:
            self.combo_roi_type.setCurrentText("Petri Dish (ROI)")
        self.combo_roi_type.blockSignals(False)
        
        shape_type = selected_roi.get("shape_type", "Rectangle")
        self.combo_shape_type.blockSignals(True)
        self.combo_shape_type.setCurrentText(shape_type)
        self.combo_shape_type.blockSignals(False)
        
        is_locked = (shape_type in ("Square", "Circle"))
        self.view.is_square_locked = is_locked
        
        self.spin_roi_w.blockSignals(True)
        self.spin_roi_h.blockSignals(True)
        self.spin_roi_w.setValue(selected_roi["w"])
        self.spin_roi_h.setValue(selected_roi["h"])
        self.spin_roi_w.blockSignals(False)
        self.spin_roi_h.blockSignals(False)
        
        self.view.is_drawing = False
        if self.view.rect_item:
            try:
                if self.view.rect_item.scene():
                    self.view.scene.removeItem(self.view.rect_item)
            except Exception:
                pass
        if self.view.handle_item:
            try:
                if self.view.handle_item.scene():
                    self.view.scene.removeItem(self.view.handle_item)
            except Exception:
                pass
            
        colors_roi = ["#007aff", "#34c759", "#af52de", "#5856d6", "#5ac8fa"]
        colors_target = ["#ff3b30", "#ff9500", "#ff2d55", "#e5c158", "#e558c1", "#58e5c1", "#c158e5"]
        if roi_type == "target":
            target_name_to_color = {}
            unique_target_count = 0
            for r in self.rois:
                if r.get("type", "roi") == "target":
                    norm_name = r["name"].strip().lower()
                    if norm_name not in target_name_to_color:
                        target_name_to_color[norm_name] = colors_target[unique_target_count % len(colors_target)]
                        unique_target_count += 1
            norm_selected_name = selected_roi["name"].strip().lower()
            base_color = QColor(target_name_to_color.get(norm_selected_name, colors_target[0]))
        else:
            roi_count = sum(1 for r in self.rois[:index] if r.get("type", "roi") != "target")
            base_color = QColor(colors_roi[roi_count % len(colors_roi)])

        if shape_type in ("Ellipse", "Circle"):
            from PySide6.QtWidgets import QGraphicsEllipseItem
            self.view.rect_item = QGraphicsEllipseItem(0, 0, selected_roi["w"], selected_roi["h"])
            self.view.rect_item.setTransformOriginPoint(selected_roi["w"] / 2.0, selected_roi["h"] / 2.0)
        else:
            self.view.rect_item = QGraphicsRectItem(0, 0, selected_roi["w"], selected_roi["h"])
            self.view.rect_item.setTransformOriginPoint(0, 0)
            
        self.view.rect_item.setPos(selected_roi["x"], selected_roi["y"])
        self.view.rect_item.setRotation(selected_roi.get("angle", 0.0))
        
        # Use matching color dashed pen for active edit box
        active_pen = QPen(base_color, 2, Qt.DashLine)
        active_pen.setCosmetic(True)
        self.view.rect_item.setPen(active_pen)
        self.view.scene.addItem(self.view.rect_item)
        
        self.view.handle_item = QGraphicsRectItem()
        self.view.handle_item.setParentItem(self.view.rect_item)
        h_pen = QPen(base_color, 1.5)
        h_pen.setCosmetic(True)
        self.view.handle_item.setPen(h_pen)
        self.view.handle_item.setBrush(QBrush(QColor("#ffffff")))
        self.view.handle_item.setRect(-5, -5, 10, 10)
        
        from PySide6.QtWidgets import QGraphicsEllipseItem
        self.view.rot_handle_item = QGraphicsEllipseItem()
        self.view.rot_handle_item.setParentItem(self.view.rect_item)
        r_pen = QPen(QColor("#34c759"), 1.5)
        r_pen.setCosmetic(True)
        self.view.rot_handle_item.setPen(r_pen)
        self.view.rot_handle_item.setBrush(QBrush(QColor("#ffffff")))
        self.view.rot_handle_item.setRect(-5, -5, 10, 10)
        
        from PySide6.QtWidgets import QGraphicsLineItem
        self.view.rot_line_item = QGraphicsLineItem()
        self.view.rot_line_item.setParentItem(self.view.rect_item)
        l_pen = QPen(QColor("#34c759"), 1.5, Qt.DashLine)
        l_pen.setCosmetic(True)
        self.view.rot_line_item.setPen(l_pen)
        
        self.view.update_handle_position()
        self.view.update_cosmetic_scales()
        self.draw_all_rois_on_scene()

    def draw_all_rois_on_scene(self):
        self.view.draw_rois(self.rois, self.selected_roi_index)

    def add_new_roi(self):
        if not self.metadata:
            QMessageBox.warning(self, "Warning", "Please open a dataset first.")
            return
            
        roi_type_text = self.combo_roi_type.currentText()
        new_type = "roi" if "Petri Dish" in roi_type_text else "target"
        
        if new_type == "target":
            if hasattr(self, "last_target_name") and self.last_target_name:
                new_name = self.last_target_name
            else:
                base_name = "Target"
                num = 1
                names = {r["name"] for r in self.rois}
                while f"{base_name}_{num}" in names:
                    num += 1
                new_name = f"{base_name}_{num}"
        else:
            base_name = "ROI"
            num = 1
            names = {r["name"] for r in self.rois}
            while f"{base_name}_{num}" in names:
                num += 1
            new_name = f"{base_name}_{num}"
        
        w = self.spin_roi_w.value()
        h = self.spin_roi_h.value()
        
        # Center the new ROI in the current viewport view
        viewport_center = self.view.viewport().rect().center()
        scene_center = self.view.mapToScene(viewport_center)
        x = int(scene_center.x() - w / 2.0)
        y = int(scene_center.y() - h / 2.0)
        
        new_roi = {
            "name": new_name,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "shape_type": self.combo_shape_type.currentText(),
            "angle": 0.0,
            "type": new_type
        }
        self.rois.append(new_roi)
        self.selected_roi_index = len(self.rois) - 1
        
        self.select_roi_by_index(self.selected_roi_index)

    def rename_selected_roi(self):
        if self.selected_roi_index < 0 or self.selected_roi_index >= len(self.rois):
            QMessageBox.warning(self, "Warning", "Please select an annotation to modify.")
            return
        new_name = self.txt_roi_name.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Warning", "Annotation name cannot be empty.")
            return
        roi = self.rois[self.selected_roi_index]
        roi["name"] = new_name
        roi["w"] = self.spin_roi_w.value()
        roi["h"] = self.spin_roi_h.value()
        roi["shape_type"] = self.combo_shape_type.currentText()
        
        roi_type_text = self.combo_roi_type.currentText()
        roi["type"] = "roi" if "Petri Dish" in roi_type_text else "target"
        
        if roi["type"] == "target":
            self.last_target_name = new_name
            
        self.select_roi_by_index(self.selected_roi_index)

    def delete_selected_roi(self):
        if self.selected_roi_index < 0 or self.selected_roi_index >= len(self.rois):
            QMessageBox.warning(self, "Warning", "Please select an ROI to delete.")
            return
        del self.rois[self.selected_roi_index]
        self.selected_roi_index = -1
        
        if self.view.rect_item:
            try:
                if self.view.rect_item.scene():
                    self.view.scene.removeItem(self.view.rect_item)
            except Exception:
                pass
            self.view.rect_item = None
        if self.view.handle_item:
            try:
                if self.view.handle_item.scene():
                    self.view.scene.removeItem(self.view.handle_item)
            except Exception:
                pass
            self.view.handle_item = None
            
        self.select_roi_by_index(-1)

    def handle_roi_created(self, x, y, w, h):
        pass # we don't automatically create on click-drag drawing anymore

    def prompt_rename_roi(self, index):
        if 0 <= index < len(self.rois):
            roi = self.rois[index]
            new_name, ok = QInputDialog.getText(
                self, "Rename ROI", f"Enter new name for '{roi['name']}':", text=roi["name"]
            )
            if ok and new_name.strip():
                roi["name"] = new_name.strip()
                if roi.get("type", "roi") == "target":
                    self.last_target_name = roi["name"]
                if index == self.selected_roi_index:
                    self.txt_roi_name.setText(roi["name"])
                self.draw_all_rois_on_scene()

    def export_all_rois_to_npz(self):
        if not self.metadata or not self.rois:
            QMessageBox.warning(self, "Warning", "No ROIs to export.")
            return
            
        from PySide6.QtWidgets import QProgressDialog
        progress = QProgressDialog("Exporting ROIs to NPZ...", "Cancel", 0, len(self.rois), self)
        progress.setWindowTitle("NPZ Export Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        success_count = 0
        for i, roi in enumerate(self.rois):
            if progress.wasCanceled():
                break
                
            progress.setValue(i)
            progress.setLabelText(f"Exporting ROI '{roi['name']}' ({i+1}/{len(self.rois)})...")
            QApplication.processEvents()
            
            x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
            name = roi["name"]
            roi_type = roi.get("type", "roi")
            try:
                self.send_post_request("/save_roi", {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "name": name,
                    "shape_type": roi.get("shape_type", "Rectangle"),
                    "angle": roi.get("angle", 0.0),
                    "type": roi_type,
                    "all_rois": self.rois
                }, timeout=300)
                success_count += 1
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to export ROI '{name}':\n{str(e)}")
                
        progress.setValue(len(self.rois))
        
        if success_count > 0:
            QMessageBox.information(
                self, 
                "Success", 
                f"Successfully exported {success_count} / {len(self.rois)} ROIs to NPZ files on the server!"
            )

def main():
    app = QApplication(sys.argv)
    window = MicroplasticClientApp()
    window.show()
    QTimer.singleShot(0, window.showMaximized)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

