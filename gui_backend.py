import json
from typing import Any, Dict, Tuple
from main import (
    factorize_equation,
    complete_the_square,
    calculate_combination,
    calculate_permutation,
    solve_user_equation,
    solve_system_of_equations,
    calculate_expression,
    calculate_custom_expression,
    calculate_days_remaining,
    generate_random_numbers,
    calculate_statistics,
    perform_linear_regression,
    stars_and_bars,
    vector_operations,
    matrix_operations,
    solve_differential_equation,
    sequence_analysis,
    unit_conversion,
    parse_math_input,
    format_sympy_output
)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import symbols

SCIENTIFIC_DIGITS = 50

# plotting helpers that return matplotlib figure objects

def plot_functions_backend(funcs, x_min, x_max, points):
    x_values = np.linspace(x_min, x_max, points)
    fig, ax = plt.subplots()
    for func in funcs:
        y_values = []
        for x in x_values:
            y_values.append(eval(func.replace('^', '**').replace('x', str(x))))
        ax.plot(x_values, y_values, label=func)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return fig

def plot_probability_distribution_backend(distribution, params):
    x = np.linspace(params['x_min'], params['x_max'], 1000)
    if distribution == '正規分布':
        y = sns.norm.pdf(x, loc=params['mean'], scale=params['std_dev'])
    elif distribution == '二項分布':
        y = sns.binom.pmf(x, n=params['n'], p=params['p'])
    elif distribution == 'ポアソン分布':
        y = sns.poisson.pmf(x, mu=params['lambda'])
    else:
        raise ValueError('未知の分布です')
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('Probability')
    return fig

def draw_geometric_shape_backend(shape, params):
    fig, ax = plt.subplots()
    if shape == '円':
        r = params['半径']
        circle = plt.Circle((0,0), radius=r, fill=False)
        ax.add_artist(circle)
        ax.set_xlim(-r*1.2, r*1.2)
        ax.set_ylim(-r*1.2, r*1.2)
    elif shape == '三角形':
        tri = plt.Polygon(params['頂点'], fill=False)
        ax.add_artist(tri)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
    ax.set_aspect('equal','box')
    return fig

# format output for GUI display

def format_response(response: Any, use_fraction: bool, output_format: str, use_scientific: bool) -> str:
    formatted = format_sympy_output(response, use_fraction, output_format, use_scientific)
    if isinstance(formatted, (dict, list)):
        return json.dumps(formatted, ensure_ascii=False, indent=2)
    return str(formatted)

# execution dispatcher

def run_feature(name: str, params: Dict[str, Any], opts: Dict[str, Any]) -> Tuple[str, Any]:
    """Run a feature and return (text, figure)."""
    fig = None
    if name == '因数分解':
        res = factorize_equation(params['equation'], opts['input_format'])
    elif name == '平方完成':
        res = complete_the_square(params['equation'], opts['input_format'])
    elif name == '組合せの計算':
        res = calculate_combination(int(params['n']), int(params['m']))
    elif name == '順列の計算':
        res = calculate_permutation(int(params['n']), int(params['m']))
    elif name == '方程式':
        res = solve_user_equation(params['equation'], opts['input_format'])
    elif name == '連立方程式':
        res = solve_system_of_equations(params['equations'], opts['input_format'])
    elif name == '式の計算':
        res = calculate_expression(params['expression'], opts['input_format'])
    elif name == 'カスタム式の計算':
        res = calculate_custom_expression(params['expression'], opts['input_format'])
    elif name == '日数の計算':
        res = calculate_days_remaining(params['date'])
    elif name == 'ランダム数の生成':
        res = generate_random_numbers(float(params['start']), float(params['end']), int(params['samples']))
    elif name == '統計の計算':
        data = [float(x.strip()) for x in params['data'].split(',') if x.strip()]
        res = calculate_statistics(data)
    elif name == '線形回帰の実行':
        rows = [line.split(',') for line in params['data'].strip().split('\n') if line.strip()]
        data = [[float(a), float(b)] for a,b in rows]
        res = perform_linear_regression(data)
    elif name == '関数のプロット':
        funcs = [f.strip() for f in params['functions'].split(',') if f.strip()]
        fig = plot_functions_backend(funcs, float(params['x_min']), float(params['x_max']), int(params['points']))
        res = {'status': 'plotted'}
    elif name == 'Stars and Bars':
        res = stars_and_bars(int(params['n']), int(params['k']))
    elif name == 'ベクトル演算':
        v1 = [float(x) for x in params['v1'].split(',')]
        v2 = [float(x) for x in params['v2'].split(',')]
        res = vector_operations(v1, v2, params['operation'])
    elif name == '行列演算':
        m1 = [list(map(float, row.split(','))) for row in params['m1'].split(';')]
        m2 = [list(map(float, row.split(','))) for row in params['m2'].split(';')]
        res = matrix_operations(m1, m2, params['operation'])
    elif name == '微分方程式の解法':
        dep = symbols(params['dependent'])
        res = solve_differential_equation(params['equation'], dep, opts['input_format'])
    elif name == '数列の解析':
        seq = [float(x.strip()) for x in params['sequence'].split(',') if x.strip()]
        res = sequence_analysis(seq)
    elif name == '単位変換':
        res = unit_conversion(float(params['value']), params['from_unit'], params['to_unit'])
    elif name == '幾何学的図形の描画と計算':
        shape = params['shape']
        if shape == '円':
            fig = draw_geometric_shape_backend(shape, {'半径': float(params['radius'])})
        elif shape == '三角形':
            pts = []
            for line in params['vertices'].strip().split('\n'):
                x,y = map(float, line.split(','))
                pts.append((x,y))
            fig = draw_geometric_shape_backend(shape, {'頂点': pts})
        res = {'status': 'plotted'}
    else:
        raise ValueError(f'未対応の機能: {name}')

    text = format_response(res, opts['use_fraction'], opts['output_format'], opts['use_scientific'])
    return text, fig
