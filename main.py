import streamlit as st
import sympy as sp
from sympy import factor, parse_expr, symbols, Eq, solve, sympify, binomial, factorial, And, solve_univariate_inequality, expand
from fractions import Fraction
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import re
import math

# 汎用的な入力検証とエラーハンドリング
def validate_input(input_data, data_type, help_text=None):
    if help_text:
        st.write(help_text)
    if not isinstance(input_data, data_type):
        return {"error": f"入力が無効です。{data_type}型で入力してください。"}
    return None

# キャッシュを利用して計算の高速化
@st.cache_data
def calculate_combination(n, m):
    result = binomial(n, m)
    return {f"{n}C{m}": result}

@st.cache_data
def calculate_permutation(n, m):
    result = factorial(n) / factorial(n - m)
    return {f"{n}P{m}": result}

@st.cache_data
def factorize_equation(equation):
    x, y, z = symbols('x,y,z')
    try:
        expr = parse_expr(equation)  # 方程式を解析する
        factorized_expr = factor(expr)  # 方程式を因数分解する
        expanded = expand(expr)
        return {"factorizedEq": factorized_expr, "expanded": expanded}
    except Exception as e:
        return f"\n{{エラー: {e}}}"

@st.cache_data
def solve_user_equation(equation_str):
    x, y = symbols('x y')
    try:
        equation = Eq(sympify(equation_str), 0)
        solution = solve(equation, x, rational=True)
        return {"x_values": [str(sol) for sol in solution]}
    except:
        return {"error": "無効な方程式形式です。正しい方程式を入力してください。"}

@st.cache_data
def calculate_expression(expression_str):
    expression = sympify(expression_str)
    result = sp.nsimplify(expression)
    return {"result": str(result)}

@st.cache_data
def calculate_custom_expression(custom_expression_str):
    x, y = symbols('x y')
    custom_expression = sympify(custom_expression_str)
    result = custom_expression.evalf(subs={x: 3, y: 5})
    return {"result": str(result)}

@st.cache_data
def calculate_days_remaining(target_date_str):
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        current_date = datetime.now()
        days_remaining = (target_date - current_date).days
        return {"days_remaining": days_remaining}
    except:
        return {"error": "無効な日付形式です。YYYY-MM-DDの形式で入力してください。"}

@st.cache_data
def generate_random_numbers(start=0, end=1, num_samples=10):
    random_numbers = np.random.uniform(start, end, num_samples).tolist()
    return {"random_numbers": random_numbers}

@st.cache_data
def calculate_statistics(data):
    if not data or not isinstance(data, list):
        return {"error": "データはリスト形式で入力してください。"}
    
    data_array = np.array(data)
    mean = np.mean(data_array)
    median = np.median(data_array)
    variance = np.var(data_array)
    std_deviation = np.std(data_array)
    
    return {
        "mean": mean,
        "median": median,
        "variance": variance,
        "standard_deviation": std_deviation
    }

@st.cache_data
def perform_linear_regression(data):
    if isinstance(data, list):
        data = np.array(data)
    else:
        return {"error": "データはリスト形式で入力してください。"}
    
    if data.shape[1] != 2:
        return {"error": "データは2列で入力してください。"}
    
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    
    model = LinearRegression()
    model.fit(X, y)
    
    coef = model.coef_[0]
    intercept = model.intercept_
    equation = f"y = {coef:.2f}x + {intercept:.2f}"
    
    return {"coefficients": coef, "intercept": intercept, "equation": equation}

def plot_functions(functions, x_range, x_label='X', y_label='Y'):
    x_values = np.linspace(x_range[0], x_range[1], x_range[2])
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for func in functions:
        y_values = eval_func(func, x_values)
        ax.plot(x_values, y_values, label=func)
    
    ax.grid(visible=None, which='major', axis='both')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    st.pyplot(fig)

def eval_func(func, x_values):
    y_values = []
    for x in x_values:
        y_values.append(eval(func.replace("^", "**").replace("x", str(x))))
    return y_values

@st.cache_data
def solve_univariate_inequalityA(inequality_str, x):
    inequality = eval(inequality_str)
    solution = solve_univariate_inequality(inequality, x)
    return solution

@st.cache_data
def find_common_region(inequality1_str, inequality2_str):
    x = symbols('x')
    inequality1 = eval(inequality1_str)
    inequality2 = eval(inequality2_str)
    common_region = And(inequality1, inequality2)
    solution = solve_univariate_inequality(common_region, x)
    return solution

@st.cache_data
def solve_system_of_inequalities(inequality1_str, inequality2_str):
    x = symbols('x')
    inequality1 = eval(inequality1_str)
    inequality2 = eval(inequality2_str)
    solution = solve((inequality1, inequality2), x)
    return solution

@st.cache_data
def complete_the_square(equation_str):
    x = symbols('x')
    try:
        equation = sympify(equation_str)
        completed_square = sp.complete_the_square(equation, x)
        return {"completed_square": completed_square}
    except Exception as e:
        return {"error": f"無効な方程式形式です。エラー: {e}"}

@st.cache_data
def solve_system_of_equations(equations_str):
    try:
        variables = sorted(set(re.findall(r'[a-zA-Z]+', equations_str)))
        symbols_dict = symbols(' '.join(variables))
        symbols_tuple = tuple(symbols_dict)
        equations = [Eq(sympify(eq.split('=')[0].strip()), sympify(eq.split('=')[1].strip())) for eq in equations_str.split(',')]
        solutions = solve(equations, symbols_tuple, dict=True)
        formatted_solutions = [{str(k): v for k, v in solution.items()} for solution in solutions]
        return formatted_solutions
    except Exception as e:
        return {"error": f"無効な方程式形式です。エラー: {e}"}

@st.cache_data
def stars_and_bars(n, k):
    try:
        if n < 0 or k <= 0:
            raise ValueError("nは0以上、kは1以上である必要があります。")
        result = math.comb(n + k - 1, k - 1)
        return {"Stars and Bars": result}
    except Exception as e:
        return {"error": f"計算エラー: {str(e)}"}

# ベクトル演算
@st.cache_data
def vector_operations(v1, v2, operation):
    try:
        v1 = np.array(v1)
        v2 = np.array(v2)
        if operation == "加算":
            result = np.add(v1, v2)
        elif operation == "減算":
            result = np.subtract(v1, v2)
        elif operation == "内積":
            result = np.dot(v1, v2)
        elif operation == "外積":
            result = np.cross(v1, v2)
        return {"result": result.tolist()}
    except Exception as e:
        return {"error": f"計算エラー: {str(e)}"}

# 行列演算
@st.cache_data
def matrix_operations(m1, m2, operation):
    try:
        m1 = np.array(m1)
        m2 = np.array(m2)
        if operation == "加算":
            result = np.add(m1, m2)
        elif operation == "減算":
            result = np.subtract(m1, m2)
        elif operation == "乗算":
            result = np.matmul(m1, m2)
        elif operation == "逆行列":
            result = np.linalg.inv(m1)
        elif operation == "行列式":
            result = np.linalg.det(m1)
        return {"result": result.tolist()}
    except Exception as e:
        return {"error": f"計算エラー: {str(e)}"}

# 微分方程式の解法
@st.cache_data
def solve_differential_equation(equation_str, dependent_var):
    try:
        equation = sympify(equation_str)
        solution = sp.dsolve(equation, dependent_var)
        return {"solution": str(solution)}
    except Exception as e:
        return {"error": f"解法エラー: {str(e)}"}

# 確率分布の可視化
def plot_probability_distribution(distribution, params):
    try:
        x = np.linspace(params["x_min"], params["x_max"], 1000)
        if distribution == "正規分布":
            y = sns.norm.pdf(x, loc=params["mean"], scale=params["std_dev"])
        elif distribution == "二項分布":
            y = sns.binom.pmf(x, n=params["n"], p=params["p"])
        elif distribution == "ポアソン分布":
            y = sns.poisson.pmf(x, mu=params["lambda"])
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel("x")
        ax.set_ylabel("確率")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"プロットエラー: {str(e)}")

# 数列の解析
@st.cache_data
def sequence_analysis(sequence):
    try:
        sequence = np.array(sequence)
        difference = np.diff(sequence)
        is_arithmetic = np.all(difference == difference[0])
        return {"difference": difference.tolist(), "is_arithmetic": is_arithmetic}
    except Exception as e:
        return {"error": f"解析エラー: {str(e)}"}

# 単位変換
@st.cache_data
def unit_conversion(value, from_unit, to_unit):
    try:
        # Example: length conversion
        conversion_factors = {
            "m_to_km": 0.001,
            "km_to_m": 1000,
            # Add more units as needed
        }
        key = f"{from_unit}_to_{to_unit}"
        if key in conversion_factors:
            result = value * conversion_factors[key]
            return {"converted_value": result}
        else:
            return {"error": "変換できません"}
    except Exception as e:
        return {"error": f"変換エラー: {str(e)}"}

# 幾何学的図形の描画と計算
def draw_geometric_shape(shape, params):
    try:
        fig, ax = plt.subplots()
        if shape == "円":
            circle = plt.Circle((0, 0), radius=params["半径"], fill=False)
            ax.add_artist(circle)
            ax.set_xlim(-params["半径"] * 1.2, params["半径"] * 1.2)
            ax.set_ylim(-params["半径"] * 1.2, params["半径"] * 1.2)
        elif shape == "三角形":
            triangle = plt.Polygon(params["頂点"], fill=False)
            ax.add_artist(triangle)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        ax.set_aspect("equal", "box")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"描画エラー: {str(e)}")

def main():
    st.title("Streamlit App")

    selected_function = st.sidebar.selectbox("機能を選択", [
        "因数分解", "方程式", "連立方程式", "式の計算", "カスタム式の計算", "日数の計算", 
        "組合せの計算", "順列の計算", "不等式", "共通部分", "連立不等式", "ランダム数の生成", 
        "統計の計算", "線形回帰の実行", "関数のプロット", "平方完成", "Stars and Bars",
        "ベクトル演算", "行列演算", "微分方程式の解法", "確率分布の可視化", "数列の解析", 
        "単位変換", "幾何学的図形の描画と計算"
    ])

    # 動的なヘルプ表示
    st.sidebar.markdown("### 使用方法")
    
    if selected_function == "因数分解":
        st.sidebar.markdown("因数分解する式を入力してください。例: `x**2 - 4`")
        equation = st.text_input("因数分解する方程式を入力してください")
        if st.button("因数分解する"):
            response = factorize_equation(equation)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "平方完成":
        st.sidebar.markdown("平方完成を行いたい式を入力してください。例: `x**2 + 4*x + 4`")
        equation = st.text_input("平方完成する方程式を入力してください")
        if st.button("平方完成する"):
            response = complete_the_square(equation)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "組合せの計算":
        st.sidebar.markdown("組合せの計算を行います。nCkのnとkを入力してください。")
        n = st.number_input("nを入力してください", value=0, min_value=0, step=1)
        m = st.number_input("mを入力してください", value=0, min_value=0, step=1)
        if st.button("計算する"):
            response = calculate_combination(n, m)
            st.json(response)

    elif selected_function == "順列の計算":
        st.sidebar.markdown("順列の計算を行います。nPkのnとkを入力してください。")
        n = st.number_input("nを入力してください", value=0, min_value=0, step=1)
        m = st.number_input("mを入力してください", value=0, min_value=0, step=1)
        if st.button("計算する"):
            response = calculate_permutation(n, m)
            st.json(response)

    elif selected_function == "方程式":
        st.sidebar.markdown("方程式を解く場合、=0の形式で入力してください。例: `2*x + 3 = 0`")
        equation = st.text_input("方程式を入力してください")
        if st.button("解く"):
            response = solve_user_equation(equation)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "連立方程式":
        st.sidebar.markdown("方程式をカンマで区切って入力してください。例: `2*x + y = 10, 3*x - y = 5`")
        equations_str = st.text_input("連立方程式を入力してください")
        if st.button("解く"):
            response = solve_system_of_equations(equations_str)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "式の計算":
        st.sidebar.markdown("計算したい式を入力してください。例: `2 + 3*4`")
        expression = st.text_input("式を入力してください")
        if st.button("計算する"):
            response = calculate_expression(expression)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "カスタム式の計算":
        st.sidebar.markdown("カスタム式を入力してください。例: `x**2 + y**2` (x=3, y=5)")
        custom_expression = st.text_input("カスタム式を入力してください")
        if st.button("計算する"):
            response = calculate_custom_expression(custom_expression)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "日数の計算":
        st.sidebar.markdown("目標日を入力してください。形式: YYYY-MM-DD")
        target_date = st.text_input("目標日を入力してください (YYYY-MM-DD)")
        if st.button("計算する"):
            response = calculate_days_remaining(target_date)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "ランダム数の生成":
        st.sidebar.markdown("ランダム数を生成する範囲とサンプル数を入力してください。")
        start = st.number_input("開始", value=0.0)
        end = st.number_input("終了", value=1.0)
        num_samples = st.number_input("サンプル数", value=10, step=1)
        if st.button("生成する"):
            response = generate_random_numbers(start, end, num_samples)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "統計の計算":
        st.sidebar.markdown("データをカンマ区切りで入力してください。例: `1,2,3,4,5`")
        data = st.text_area("データを入力してください（カンマ区切り）")
        if st.button("計算する"):
            data_list = [float(x.strip()) for x in data.split(",") if x.strip()]
            response = calculate_statistics(data_list)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "線形回帰の実行":
        st.sidebar.markdown("データポイントをカンマ区切りで入力してください。例: `[1,1], [2,2]`")
        data = st.text_area("データポイントを入力してください（カンマ区切り）")
        if st.button("回帰を実行する"):
            data_list = [list(map(float, point.strip("[] ").split(","))) for point in data.split("],") if point.strip()]
            response = perform_linear_regression(data_list)
            if "error" in response:
                st.error(response["error"])
            else:
                st.write("回帰式:", response["equation"])
                st.write("回帰係数:", response["coefficients"])
                st.write("切片:", response["intercept"])

    elif selected_function == "関数のプロット":
        st.sidebar.markdown("Python構文で関数を入力してください。複数の関数をカンマで区切ってください。例: `x+1, x*x+1`")
        functions = st.text_input("関数を入力してください")
        x_min = st.number_input("X軸の最小値", value=-10.0)
        x_max = st.number_input("X軸の最大値", value=10.0)
        num_points = st.number_input("X軸の点の数", min_value=10, max_value=10000, value=1000)
        x_range = (x_min, x_max, num_points)
        x_label = st.text_input("Xラベル", value="X")
        y_label = st.text_input("Yラベル", value="Y")
        if st.button("プロットする"):
            function_list = [x.strip() for x in functions.split(",") if x.strip()]
            plot_functions(function_list, x_range, x_label, y_label)

    elif selected_function == "不等式":
        st.sidebar.markdown("解きたい不等式を入力してください。例: `x**2 - 4 > 0`")
        inequality_str = st.text_input("不等式を入力してください")
        if st.button("解く"):
            try:
                solution = solve_univariate_inequalityA(inequality_str, symbols('x'))
                st.write("解:", solution)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
    
    elif selected_function == "共通部分":
        st.sidebar.markdown("2つの不等式を入力してください。例: `x > 1, x < 5`")
        inequality1_str = st.text_input("1つ目の不等式を入力してください")
        inequality2_str = st.text_input("2つ目の不等式を入力してください")
        if st.button("共通部分を求める"):
            try:
                solution = find_common_region(inequality1_str, inequality2_str)
                st.write("共通部分:", solution)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

    elif selected_function == "連立不等式":
        st.sidebar.markdown("2つの不等式を入力してください。例: `x > 1, x < 5`")
        inequality1_str = st.text_input("1つ目の不等式を入力してください")
        inequality2_str = st.text_input("2つ目の不等式を入力してください")
        if st.button("解く"):
            try:
                solution = solve_system_of_inequalities(inequality1_str, inequality2_str)
                st.write("解:", solution)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

    elif selected_function == "Stars and Bars":
        st.sidebar.markdown("n個のものをk個のグループに分ける方法を計算します。")
        n = st.number_input("nを入力してください（分けたいものの数）", value=0, min_value=0, step=1)
        k = st.number_input("kを入力してください（グループの数）", value=1, min_value=1, step=1)
        if st.button("計算する"):
            response = stars_and_bars(n, k)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "ベクトル演算":
        st.sidebar.markdown("2つのベクトルを操作します。")
        v1 = st.text_area("ベクトル1を入力（カンマ区切り）", "1,2,3")
        v2 = st.text_area("ベクトル2を入力（カンマ区切り）", "4,5,6")
        operation = st.selectbox("演算を選択", ["加算", "減算", "内積", "外積"])
        if st.button("計算する"):
            v1 = list(map(float, v1.split(",")))
            v2 = list(map(float, v2.split(",")))
            response = vector_operations(v1, v2, operation)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "行列演算":
        st.sidebar.markdown("2つの行列を操作します。")
        m1 = st.text_area("行列1を入力（行ごとにセミコロン区切り）", "1,2;3,4")
        m2 = st.text_area("行列2を入力（行ごとにセミコロン区切り）", "5,6;7,8")
        operation = st.selectbox("演算を選択", ["加算", "減算", "乗算", "逆行列", "行列式"])
        if st.button("計算する"):
            m1 = [list(map(float, row.split(","))) for row in m1.split(";")]
            m2 = [list(map(float, row.split(","))) for row in m2.split(";")]
            response = matrix_operations(m1, m2, operation)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "微分方程式の解法":
        st.sidebar.markdown("常微分方程式 (ODE) を解きます。")
        equation = st.text_input("方程式を入力してください")
        dependent_var = st.text_input("従属変数を入力してください", "y")
        if st.button("解く"):
            response = solve_differential_equation(equation, symbols(dependent_var))
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "確率分布の可視化":
        st.sidebar.markdown("確率分布を可視化します。")
        distribution = st.selectbox("分布を選択", ["正規分布", "二項分布", "ポアソン分布"])
        params = {}
        params["x_min"] = st.number_input("xの最小値", value=-10.0)
        params["x_max"] = st.number_input("xの最大値", value=10.0)
        if distribution == "正規分布":
            params["mean"] = st.number_input("平均", value=0.0)
            params["std_dev"] = st.number_input("標準偏差", value=1.0)
        elif distribution == "二項分布":
            params["n"] = st.number_input("試行回数", value=10)
            params["p"] = st.number_input("成功確率", value=0.5)
        elif distribution == "ポアソン分布":
            params["lambda"] = st.number_input("λ（平均発生率）", value=1.0)
        if st.button("プロットする"):
            plot_probability_distribution(distribution, params)

    elif selected_function == "数列の解析":
        st.sidebar.markdown("数列を解析します。")
        sequence = st.text_area("数列を入力（カンマ区切り）", "1,3,5,7,9")
        if st.button("解析する"):
            sequence = list(map(int, sequence.split(",")))
            response = sequence_analysis(sequence)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "単位変換":
        st.sidebar.markdown("物理単位を変換します。")
        value = st.number_input("値を入力", value=1.0)
        from_unit = st.selectbox("変換元の単位", ["m", "km"])
        to_unit = st.selectbox("変換先の単位", ["km", "m"])
        if st.button("変換する"):
            response = unit_conversion(value, from_unit, to_unit)
            if "error" in response:
                st.error(response["error"])
            else:
                st.json(response)

    elif selected_function == "幾何学的図形の描画と計算":
        st.sidebar.markdown("幾何学的図形を描画します。")
        shape = st.selectbox("図形を選択", ["円", "三角形"])
        params = {}
        if shape == "円":
            params["半径"] = st.number_input("半径", value=1.0)
        elif shape == "三角形":
            params["頂点"] = [
                (st.number_input("頂点1 (x)", value=0.0), st.number_input("頂点1 (y)", value=0.0)),
                (st.number_input("頂点2 (x)", value=1.0), st.number_input("頂点2 (y)", value=0.0)),
                (st.number_input("頂点3 (x)", value=0.5), st.number_input("頂点3 (y)", value=1.0))
            ]
        if st.button("描画する"):
            draw_geometric_shape(shape, params)

if __name__ == "__main__":
    main()
