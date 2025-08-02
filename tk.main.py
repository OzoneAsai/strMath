import tkinter as tk
from tkinter import ttk, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tkinter Math GUI")
        self.inputs = {}

        option_frame = ttk.Frame(self)
        option_frame.pack(fill='x')
        self.use_fraction = tk.BooleanVar()
        self.use_scientific = tk.BooleanVar(value=True)
        ttk.Checkbutton(option_frame, text="結果を分数で表示", variable=self.use_fraction).pack(side='left')
        ttk.Checkbutton(option_frame, text="指数表示", variable=self.use_scientific).pack(side='left')
        ttk.Label(option_frame, text="入力形式").pack(side='left')
        self.input_format = ttk.Combobox(option_frame, values=["SymPy","LaTeX"], width=7)
        self.input_format.current(0)
        self.input_format.pack(side='left')
        ttk.Label(option_frame, text="出力形式").pack(side='left')
        self.output_format = ttk.Combobox(option_frame, values=["json_sympy","json_latex","latex_render","raw_sympy"], width=10)
        self.output_format.current(0)
        self.output_format.pack(side='left')

        self.feature = ttk.Combobox(self, values=list(FEATURE_FIELDS.keys()))
        self.feature.current(0)
        self.feature.pack(fill='x')
        self.feature.bind("<<ComboboxSelected>>", lambda e: self.build_form())

        self.form = ttk.Frame(self)
        self.form.pack(fill='x')

        ttk.Button(self, text="実行", command=self.execute).pack(fill='x')

        self.output = scrolledtext.ScrolledText(self, height=10)
        self.output.pack(fill='both', expand=True)

        self.fig_canvas = None

        self.build_form()

    def build_form(self):
        for child in self.form.winfo_children():
            child.destroy()
        self.inputs = {}
        for key, label in FEATURE_FIELDS[self.feature.get()]:
            ttk.Label(self.form, text=label).pack(anchor='w')
            if key in ('operation', 'from_unit', 'to_unit', 'shape'):
                if key == 'operation' and 'ベクトル演算' in self.feature.get():
                    values = ['加算','減算','内積','外積']
                elif key == 'operation':
                    values = ['加算','減算','乗算','逆行列','行列式']
                elif key == 'from_unit':
                    values = ['m','km']
                elif key == 'to_unit':
                    values = ['km','m']
                elif key == 'shape':
                    values = ['円','三角形']
                cb = ttk.Combobox(self.form, values=values)
                cb.current(0)
                cb.pack(fill='x')
                self.inputs[key] = cb
            elif key == 'vertices':
                txt = scrolledtext.ScrolledText(self.form, height=4)
                txt.pack(fill='x')
                self.inputs[key] = txt
            else:
                entry = ttk.Entry(self.form)
                entry.pack(fill='x')
                self.inputs[key] = entry

    def execute(self):
        name = self.feature.get()
        params = {}
        for key, widget in self.inputs.items():
            if isinstance(widget, ttk.Entry):
                params[key] = widget.get()
            elif isinstance(widget, ttk.Combobox):
                params[key] = widget.get()
            else:
                params[key] = widget.get("1.0", tk.END).strip()
        opts = {
            'use_fraction': self.use_fraction.get(),
            'use_scientific': self.use_scientific.get(),
            'input_format': self.input_format.get(),
            'output_format': self.output_format.get()
        }
        try:
            text, fig = run_feature(name, params, opts)
            self.output.delete('1.0', tk.END)
            self.output.insert(tk.END, text)
            if fig:
                if self.fig_canvas:
                    self.fig_canvas.get_tk_widget().destroy()
                self.fig_canvas = FigureCanvasTkAgg(fig, master=self)
                self.fig_canvas.get_tk_widget().pack(fill='both', expand=True)
                self.fig_canvas.draw()
            elif self.fig_canvas:
                self.fig_canvas.get_tk_widget().destroy()
                self.fig_canvas = None
        except Exception as e:
            self.output.delete('1.0', tk.END)
            self.output.insert(tk.END, f"Error: {e}")
            if self.fig_canvas:
                self.fig_canvas.get_tk_widget().destroy()
                self.fig_canvas = None

if __name__ == '__main__':
    app = App()
    app.geometry('800x600')
    app.mainloop()
