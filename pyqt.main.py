import sys
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QFormLayout, QLineEdit,
    QTextEdit, QPushButton, QComboBox, QCheckBox, QLabel, QSplitter
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui_backend import run_feature

FEATURE_FIELDS = {
    '因数分解': [('equation', '式')],
    '平方完成': [('equation', '式')],
    '組合せの計算': [('n', 'n'), ('m', 'm')],
    '順列の計算': [('n', 'n'), ('m', 'm')],
    '方程式': [('equation', '方程式')],
    '連立方程式': [('equations', '方程式 (カンマ区切り)')],
    '式の計算': [('expression', '式')],
    'カスタム式の計算': [('expression', '式')],
    '日数の計算': [('date', '日付(YYYY-MM-DD)')],
    'ランダム数の生成': [('start', '開始'), ('end', '終了'), ('samples', 'サンプル数')],
    '統計の計算': [('data', 'データ(カンマ区切り)')],
    '線形回帰の実行': [('data', 'データ(x,y 各行)')],
    '関数のプロット': [('functions', '関数 (カンマ区切り)'), ('x_min', 'x最小'), ('x_max', 'x最大'), ('points', '分割数')],
    'Stars and Bars': [('n', 'n'), ('k', 'k')],
    'ベクトル演算': [('v1', 'ベクトル1'), ('v2', 'ベクトル2'), ('operation', '演算:加算,減算,内積,外積')],
    '行列演算': [('m1', '行列1'), ('m2', '行列2'), ('operation', '演算:加算,減算,乗算,逆行列,行列式')],
    '微分方程式の解法': [('equation', '方程式'), ('dependent', '従属変数')],
    '数列の解析': [('sequence', '数列(カンマ区切り)')],
    '単位変換': [('value', '値'), ('from_unit', '変換元(m,km)'), ('to_unit', '変換先(km,m)')],
    '幾何学的図形の描画と計算': [('shape', '図形(円/三角形)'), ('radius', '半径'), ('vertices', '頂点(x,y 各行)')]
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Math GUI")
        self.inputs = {}

        # left panel widgets
        opt_layout = QFormLayout()
        self.use_fraction = QCheckBox("結果を分数で表示")
        self.use_scientific = QCheckBox("指数表示")
        self.use_scientific.setChecked(True)
        self.input_format = QComboBox(); self.input_format.addItems(["SymPy", "LaTeX"])
        self.output_format = QComboBox(); self.output_format.addItems(["json_sympy", "json_latex", "latex_render", "raw_sympy"])
        opt_layout.addRow(self.use_fraction)
        opt_layout.addRow(self.use_scientific)
        opt_layout.addRow(QLabel("入力形式"), self.input_format)
        opt_layout.addRow(QLabel("出力形式"), self.output_format)

        self.feature_combo = QComboBox()
        self.feature_combo.addItems(FEATURE_FIELDS.keys())
        self.feature_combo.currentTextChanged.connect(self.build_form)

        self.form_layout = QFormLayout()

        self.run_btn = QPushButton("実行")
        self.run_btn.clicked.connect(self.execute)

        left_panel = QWidget()
        left_panel.setMinimumWidth(250)
        left_layout = QVBoxLayout(left_panel)
        left_layout.addLayout(opt_layout)
        left_layout.addWidget(self.feature_combo)
        left_layout.addLayout(self.form_layout)
        left_layout.addWidget(self.run_btn)
        left_layout.addStretch()

        # right panel widgets
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.canvas = FigureCanvas(Figure())
        self.canvas.hide()
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(self.output)
        right_splitter.addWidget(self.canvas)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        self.build_form(self.feature_combo.currentText())

    def build_form(self, name):
        # clear
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.inputs = {}
        for key, label in FEATURE_FIELDS[name]:
            if key in ('operation', 'from_unit', 'to_unit', 'shape'):
                combo = QComboBox()
                if key == 'operation' and 'ベクトル演算' in name:
                    combo.addItems(['加算','減算','内積','外積'])
                elif key == 'operation':
                    combo.addItems(['加算','減算','乗算','逆行列','行列式'])
                elif key == 'from_unit':
                    combo.addItems(['m','km'])
                elif key == 'to_unit':
                    combo.addItems(['km','m'])
                elif key == 'shape':
                    combo.addItems(['円','三角形'])
                self.form_layout.addRow(QLabel(label), combo)
                self.inputs[key] = combo
            elif key == 'vertices':
                text = QTextEdit()
                self.form_layout.addRow(QLabel(label), text)
                self.inputs[key] = text
            else:
                line = QLineEdit()
                self.form_layout.addRow(QLabel(label), line)
                self.inputs[key] = line

    def execute(self):
        name = self.feature_combo.currentText()
        params = {}
        for key, widget in self.inputs.items():
            if isinstance(widget, QLineEdit):
                params[key] = widget.text()
            elif isinstance(widget, QTextEdit):
                params[key] = widget.toPlainText()
            elif isinstance(widget, QComboBox):
                params[key] = widget.currentText()
        opts = {
            'use_fraction': self.use_fraction.isChecked(),
            'use_scientific': self.use_scientific.isChecked(),
            'input_format': self.input_format.currentText(),
            'output_format': self.output_format.currentText()
        }
        try:
            text, fig = run_feature(name, params, opts)
            self.output.setText(text)
            if fig:
                self.canvas.figure = fig
                self.canvas.draw()
                self.canvas.show()
            else:
                self.canvas.hide()
        except Exception as e:
            self.output.setText(f"Error: {e}")
            self.canvas.hide()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec_())
