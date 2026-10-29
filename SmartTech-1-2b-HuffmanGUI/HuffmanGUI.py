import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QTableView, QMessageBox, QLabel, QSplitter
)
from PyQt6.QtCore import QAbstractTableModel, Qt
from PyQt6.QtGui import QPalette, QColor

from Huffman import compress, huffman_decode

class PandasModel(QAbstractTableModel):
    """Bridge between Pandas DataFrame and PyQt's QTableView."""
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])
        return None


class MplCanvas(FigureCanvas):
    """Embedded Matplotlib Canvas for plotting inside PyQt."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Huffman GUI & Data Extractor")
        self.resize(1000, 600)

        self.df = pd.DataFrame()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.btn_load = QPushButton("Load CSV File")
        self.btn_export = QPushButton("Export Selected to Excel")
        self.btn_plot = QPushButton("Plot Selection")
        self.btn_compress = QPushButton("Compress Selection (Huffman)")
        self.btn_sort = QPushButton("Sort by Selection")

        left_layout.addWidget(self.btn_load)
        left_layout.addWidget(self.btn_export)
        left_layout.addWidget(self.btn_plot)
        left_layout.addWidget(self.btn_compress)
        left_layout.addWidget(self.btn_sort)

        self.table_view = QTableView()
        left_layout.addWidget(self.table_view)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("<b>Visualization Area</b>"))
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        right_layout.addWidget(self.canvas)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        main_layout.addWidget(splitter)

        self.btn_load.clicked.connect(self.load_csv)
        self.btn_export.clicked.connect(self.export_selected_to_excel)
        self.btn_plot.clicked.connect(self.plot_selection)
        self.btn_compress.clicked.connect(self.huffman_compress)
        self.btn_sort.clicked.connect(self.sort_by_selection)

        self.sorting = ""

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                model = PandasModel(self.df)
                self.table_view.setModel(model)
                QMessageBox.information(self, "Success", f"Loaded {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read CSV file:\n{str(e)}")

    def get_selected_dataframe(self) -> pd.DataFrame:
        """Extracts highlighted table cells, rows, or columns as a new DataFrame."""
        if self.df.empty:
            return pd.DataFrame()

        selection_model = self.table_view.selectionModel()
        selected_indexes = selection_model.selectedIndexes()

        if not selected_indexes:
            return pd.DataFrame()

        rows = sorted(list(set(index.row() for index in selected_indexes)))
        cols = sorted(list(set(index.column() for index in selected_indexes)))

        return self.df.iloc[rows, cols]

    def export_selected_to_excel(self):
        selected_df = self.get_selected_dataframe()
        if selected_df.empty:
            QMessageBox.warning(self, "Warning", "Please highlight cells/rows in the table first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Excel File", "", "Excel Files (*.xlsx)")
        if file_path:
            try:
                selected_df.to_excel(file_path, index=False)
                QMessageBox.information(self, "Success", "Selected data exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save Excel file:\n{str(e)}")

    def plot_selection(self):
        selected_df = self.get_selected_dataframe()
        if selected_df.empty:
            QMessageBox.warning(self, "Warning", "Select numeric data in the table to plot.")
            return

        numeric_df = selected_df.select_dtypes(include=['number'])
        if numeric_df.empty:
            QMessageBox.warning(self, "Warning", "Selected region contains no numerical values to plot.")
            return

        self.canvas.ax.clear()

        for col in numeric_df.columns:
            self.canvas.ax.plot(numeric_df.index, numeric_df[col], label=str(col))

        self.canvas.ax.legend()
        title = "Plot of Selected Region"
        if (self.sorting):
            title += f" - Sorted by {(self.sorting).capitalize()}"
        self.canvas.ax.set_title(title)
        self.canvas.draw()

    def sort_by_selection(self):
        selected_df = self.get_selected_dataframe()

        if selected_df.empty:
            QMessageBox.warning(self, "Warning", "Please select a column in the table to sort by.")
            return

        if selected_df.shape[1] > 1:
            QMessageBox.warning(self, "Warning", "Please select only ONE column to sort by.")
            return

        sort_column = selected_df.columns[0]
        self.sorting = sort_column
        self.df = self.df.sort_values(by=sort_column, ascending=True).reset_index(drop=True)
        self.table_view.setModel(PandasModel(self.df))

        QMessageBox.information(self, "Success", f"Table sorted by column: '{sort_column}'")

    def huffman_compress(self):
        selected_df = self.get_selected_dataframe()
        if selected_df.empty:
            QMessageBox.warning(self, "Warning", "Select data to compress first.")
            return

        raw_data = selected_df.to_csv(index=False)
        original_byte_size = len(raw_data)
        QMessageBox.information(self, "Huffman", f"Ready to compress {original_byte_size} bytes of extracted text.")

        startTime = time.time()
        encoded_bits, huffman_tree = compress(raw_data)
        elapsedTime = round(time.time() - startTime, 3)

        decoded = huffman_decode(encoded_bits, huffman_tree)
        if (raw_data != decoded):
            QMessageBox.warning(self, "Warning", "Round trip failed.")
            return

        encoded_byte_size = self.get_compressed_byte_size(encoded_bits)
        comp_pct = self.get_compression_percentage(original_byte_size, encoded_byte_size)
        mbps = round(self.get_mbps(original_byte_size, elapsedTime), 2)

        result_text = (
            f"Successfully compressed {original_byte_size} bytes "
            f"to {encoded_byte_size} bytes ({comp_pct}%) in {elapsedTime} seconds. That's {mbps} MB/s."
        )
        
        QMessageBox.information(self, "Huffman", result_text)

    # Hjelpefunksjoner for div info
    def get_compressed_byte_size(self, encoded_bits):
        return len(encoded_bits) / 8

    def get_compression_percentage(self, original_bytes, encoded_bytes):
        return round((encoded_bytes / original_bytes) * 100, 2)

    def get_mbps(self, original_bytes, elapsed):
        original_mb = original_bytes / (10 ** 6)
        return original_mb / elapsed
    
if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyle("Fusion")
    light_palette = QPalette()
    light_palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    light_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(245, 245, 245))
    light_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    light_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    light_palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    light_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    light_palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    light_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    app.setPalette(light_palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())