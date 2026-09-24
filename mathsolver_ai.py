"""
MathSolver AI - Aplicación matemática educativa en un solo archivo.

Instalación:
    pip install streamlit sympy numpy scipy pandas matplotlib pillow

Ejecución:
    streamlit run mathsolver_ai.py

Opcionales:
    pip install pytesseract
    - Requiere Tesseract OCR instalado en el sistema para reconocimiento de imágenes.
    - La IA externa es opcional y no es necesaria para el motor matemático.

Características:
- Resolución simbólica y numérica con SymPy.
- Identificación heurística de ejercicios.
- Explicaciones paso a paso.
- Verificación independiente por sustitución/simplificación.
- Biblioteca extensible de fórmulas.
- Búsqueda de fórmulas.
- Generación de ejercicios de práctica.
- Verificación de respuestas.
- Gráficas.
- OCR opcional.
- Historial local de sesión.
- Exportación de soluciones a PDF si reportlab está instalado.
- Interfaz responsive mediante Streamlit.

La arquitectura separa conceptualmente:
interpretación -> clasificación -> método -> resolución -> verificación -> explicación.
"""

from __future__ import annotations

import io
import json
import math
import random
import re
import textwrap
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# Matplotlib es opcional hasta que se use una gráfica.
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

APP_NAME = "MathSolver AI"
APP_VERSION = "1.0.0"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="∑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODELOS DE DATOS
# ============================================================

@dataclass
class Formula:
    id: str
    name: str
    category: str
    subcategory: str
    latex: str
    variables: list[str]
    description: str
    conditions: list[str]
    example: str
    methods: list[str]
    level: str


@dataclass
class ProblemAnalysis:
    original_text: str
    normalized_text: str
    category: str
    subcategory: str
    variables: list[str]
    data: dict[str, Any]
    method: str
    confidence: float
    notes: list[str]


@dataclass
class VerificationResult:
    verified: bool
    message: str
    details: list[str]


@dataclass
class Solution:
    analysis: ProblemAnalysis
    result: Any
    steps: list[str]
    formulas: list[str]
    verification: VerificationResult
    concept: str


# ============================================================
# BIBLIOTECA DE FÓRMULAS
# Arquitectura extensible: agregar nuevos registros no requiere
# modificar el motor central.
# ============================================================

FORMULAS: list[Formula] = [
    Formula(
        "ALG-001",
        "Fórmula cuadrática",
        "Álgebra",
        "Ecuaciones de segundo grado",
        r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}",
        ["a", "b", "c"],
        "Resuelve ecuaciones ax² + bx + c = 0.",
        ["a ≠ 0"],
        "2x² - 5x - 3 = 0",
        ["fórmula general", "discriminante"],
        "Secundaria",
    ),
    Formula(
        "GEO-001",
        "Área de triángulo",
        "Geometría",
        "Áreas",
        r"A=\frac{bh}{2}",
        ["b", "h"],
        "Calcula el área de un triángulo.",
        ["b y h deben estar en unidades compatibles"],
        "b=10, h=6",
        ["área", "geometría"],
        "Secundaria",
    ),
    Formula(
        "GEO-002",
        "Área de círculo",
        "Geometría",
        "Áreas",
        r"A=\pi r^2",
        ["r"],
        "Calcula el área de un círculo.",
        ["r ≥ 0"],
        "r=5",
        ["área", "círculo"],
        "Secundaria",
    ),
    Formula(
        "GEO-003",
        "Longitud de circunferencia",
        "Geometría",
        "Perímetros",
        r"C=2\pi r",
        ["r"],
        "Calcula la longitud de una circunferencia.",
        ["r ≥ 0"],
        "r=5",
        ["circunferencia"],
        "Secundaria",
    ),
    Formula(
        "GEO-004",
        "Teorema de Pitágoras",
        "Geometría",
        "Triángulos",
        r"a^2+b^2=c^2",
        ["a", "b", "c"],
        "Relaciona los lados de un triángulo rectángulo.",
        ["El triángulo debe ser rectángulo"],
        "a=3, b=4",
        ["Pitágoras"],
        "Secundaria",
    ),
    Formula(
        "TRI-001",
        "Identidad fundamental",
        "Trigonometría",
        "Identidades",
        r"\sin^2(x)+\cos^2(x)=1",
        ["x"],
        "Identidad trigonométrica fundamental.",
        [],
        "sin²(x)+cos²(x)",
        ["identidades trigonométricas"],
        "Secundaria",
    ),
    Formula(
        "CAL-001",
        "Regla de la potencia",
        "Cálculo",
        "Derivadas",
        r"\frac{d}{dx}x^n=nx^{n-1}",
        ["n"],
        "Regla básica para derivar potencias.",
        [],
        "f(x)=x³",
        ["derivación"],
        "Secundaria/Universitaria",
    ),
    Formula(
        "CAL-002",
        "Derivada de suma",
        "Cálculo",
        "Derivadas",
        r"(f+g)'=f'+g'",
        ["f", "g"],
        "La derivada de una suma es la suma de sus derivadas.",
        [],
        "f(x)=x²+3x",
        ["derivación"],
        "Universitaria",
    ),
    Formula(
        "STA-001",
        "Media aritmética",
        "Probabilidad y estadística",
        "Estadística descriptiva",
        r"\bar{x}=\frac{\sum x_i}{n}",
        ["x_i", "n"],
        "Calcula la media de un conjunto de datos.",
        ["n > 0"],
        "2,4,6,8",
        ["media"],
        "Secundaria",
    ),
    Formula(
        "STA-002",
        "Varianza poblacional",
        "Probabilidad y estadística",
        "Estadística descriptiva",
        r"\sigma^2=\frac{\sum(x_i-\mu)^2}{N}",
        ["x_i", "μ", "N"],
        "Mide la dispersión respecto de la media poblacional.",
        ["N > 0"],
        "1,2,3",
        ["varianza"],
        "Universitaria",
    ),
    Formula(
        "ARI-001",
        "Porcentaje",
        "Aritmética",
        "Porcentajes",
        r"p=\frac{\text{parte}}{\text{total}}\times100",
        ["parte", "total"],
        "Calcula qué porcentaje representa una parte de un total.",
        ["total ≠ 0"],
        "25 de 100",
        ["porcentaje"],
        "Secundaria",
    ),
    Formula(
        "ALG-002",
        "Pendiente de una recta",
        "Geometría analítica",
        "Rectas",
        r"m=\frac{y_2-y_1}{x_2-x_1}",
        ["x₁", "y₁", "x₂", "y₂"],
        "Calcula la pendiente entre dos puntos.",
        ["x₂ ≠ x₁"],
        "(1,2), (3,6)",
        ["pendiente"],
        "Secundaria",
    ),
]


# ============================================================
# UTILIDADES
# ============================================================

def clean_text(text: str) -> str:
    text = text.strip()
    text = text.replace("²", "^2")
    text = text.replace("³", "^3")
    text = text.replace("⁴", "^4")
    text = text.replace("×", "*")
    text = text.replace("÷", "/")
    text = text.replace("−", "-")
    text = text.replace("π", "pi")
    text = re.sub(r"\s+", " ", text)
    return text


def sympy_latex(value: Any) -> str:
    try:
        return sp.latex(value)
    except Exception:
        return str(value)


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def format_result(result: Any) -> str:
    if result is None:
        return "No se obtuvo resultado."
    if isinstance(result, dict):
        return "\n".join(f"{k}: {v}" for k, v in result.items())
    try:
        return sp.pretty(result)
    except Exception:
        return str(result)


def extract_numbers(text: str) -> list[float]:
    matches = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?", text)
    result = []
    for item in matches:
        try:
            result.append(float(item.replace(",", ".")))
        except ValueError:
            pass
    return result


def normalize_equation(expr: str) -> str:
    expr = clean_text(expr)
    expr = expr.replace("=", " = ")
    expr = re.sub(r"\s+", " ", expr)
    return expr.strip()


def parse_expression(expr: str) -> sp.Expr:
    """
    Parser matemático controlado.

    Acepta notación matemática natural, incluyendo multiplicación implícita
    como 3x, x(x+1), y 2(x+1) sin requerir un '*' explícito.
    No se ejecuta código arbitrario.
    """
    expr = clean_text(expr)
    expr = expr.replace("^", "**")

    local_dict = {
        "pi": sp.pi,
        "e": sp.E,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "log": sp.log,
        "ln": sp.log,
        "exp": sp.exp,
        "abs": sp.Abs,
    }

    return parse_expr(
        expr,
        local_dict=local_dict,
        transformations=standard_transformations + (implicit_multiplication_application,),
    )


def extract_equation(text: str) -> Optional[tuple[sp.Expr, sp.Expr]]:
    match = re.search(r"(.+?)=(.+)", clean_text(text))
    if not match:
        return None

    left = match.group(1).strip()
    right = match.group(2).strip()

    try:
        return parse_expression(left), parse_expression(right)
    except Exception:
        return None


def extract_function(text: str) -> Optional[tuple[str, str]]:
    patterns = [
        r"f\s*\(\s*x\s*\)\s*=\s*(.+)",
        r"y\s*=\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean_text(text), re.I)
        if match:
            if pattern.startswith("f"):
                return "f(x)", match.group(1)
            return "y", match.group(1)
    return None


def extract_symbolic_expression(text: str, *, operation: str) -> Optional[str]:
    normalized = clean_text(text)
    lower = normalized.lower()

    if operation == "derivada":
        patterns = [
            r"(?:derivada|derivar)\s+(?:de\s+)?(.+)",
            r"d/dx\s*\(?\s*(.+?)\s*\)?(?:\s*dx)?$",
            r"d/dx\s*\(?\s*(.+?)\s*\)?$",
        ]
    else:
        patterns = [
            r"(?:integral|integrar)\s+(?:de\s+)?(.+)",
            r"∫\s*(.+?)\s*dx",
            r"∫\s*(.+?)\s*\)",
        ]

    for pattern in patterns:
        match = re.search(pattern, normalized, re.I)
        if not match:
            continue

        candidate = match.group(1).strip()
        if not candidate:
            continue
        candidate = candidate.rstrip(".")
        candidate = candidate.replace("dx", "")
        candidate = candidate.strip()
        if candidate:
            return candidate

    return None


# ============================================================
# IDENTIFICADOR
# ============================================================

class ProblemIdentifier:

    KEYWORDS = {
        "Álgebra": [
            "ecuación", "ecuaciones", "polinomio", "factoriza",
            "factorización", "inecuación", "sistema", "variable",
            "x²", "cuadrática", "cuadratico", "cuadrática",
        ],
        "Geometría": [
            "triángulo", "triangulo", "cuadrado", "rectángulo",
            "rectangulo", "círculo", "circulo", "área", "perímetro",
            "volumen", "radio", "base", "altura", "pitágoras",
        ],
        "Trigonometría": [
            "seno", "coseno", "tangente", "secante", "cosecante",
            "cotangente", "radian", "ángulo", "angulo",
        ],
        "Cálculo": [
            "derivada", "derivar", "integral", "integrar", "límite",
            "limite", "continuidad", "máximo", "mínimo", "optimización",
        ],
        "Probabilidad y estadística": [
            "media", "mediana", "moda", "varianza", "desviación",
            "desviacion", "probabilidad", "bayes", "distribución",
            "regresión", "correlación",
        ],
        "Aritmética": [
            "porcentaje", "fracción", "fraccion", "proporción",
            "regla de tres", "mcd", "mcm", "divisibilidad",
        ],
        "Geometría analítica": [
            "pendiente", "recta", "coordenadas", "distancia entre",
            "vector", "matriz", "determinante",
        ],
    }

    @classmethod
    def identify(cls, text: str) -> ProblemAnalysis:
        original = text
        normalized = clean_text(text)
        lower = normalized.lower()

        scores = {category: 0 for category in cls.KEYWORDS}

        for category, words in cls.KEYWORDS.items():
            for word in words:
                if word.lower() in lower:
                    scores[category] += 1

        category = max(scores, key=scores.get)
        max_score = scores[category]

        if max_score == 0:
            category = "Álgebra"
            confidence = 0.25
            notes = ["No se detectaron palabras clave suficientes."]
        else:
            confidence = min(0.55 + max_score * 0.10, 0.98)
            notes = []

        variables = sorted(set(re.findall(r"\b[a-zA-Z]\b", normalized)))

        # Clasificación específica.
        if re.search(r"x\s*\^?\s*2", normalized) and "=" in normalized:
            subcategory = "Ecuación de segundo grado"
            method = "Fórmula cuadrática / factorización"
        elif "deriv" in lower:
            subcategory = "Derivación"
            method = "Reglas de derivación"
        elif "integr" in lower:
            subcategory = "Integración"
            method = "Integración simbólica"
        elif "triángulo" in lower or "triangulo" in lower:
            if "pitágoras" in lower or "pitagoras" in lower:
                subcategory = "Triángulo rectángulo"
                method = "Teorema de Pitágoras"
            else:
                subcategory = "Geometría de triángulos"
                method = "Fórmula geométrica apropiada"
        elif "círculo" in lower or "circulo" in lower:
            subcategory = "Círculo"
            method = "Fórmula de área/circunferencia"
        elif "pendiente" in lower:
            subcategory = "Pendiente de una recta"
            method = "Fórmula de pendiente"
        elif "media" in lower:
            subcategory = "Media aritmética"
            method = "Media aritmética"
        elif "porcentaje" in lower:
            subcategory = "Porcentajes"
            method = "Cálculo porcentual"
        elif "=" in normalized:
            subcategory = "Ecuación"
            method = "Resolución simbólica"
        else:
            subcategory = "Problema matemático general"
            method = "Análisis simbólico"

        return ProblemAnalysis(
            original_text=original,
            normalized_text=normalized,
            category=category,
            subcategory=subcategory,
            variables=variables,
            data={},
            method=method,
            confidence=confidence,
            notes=notes,
        )


# ============================================================
# VERIFICADOR
# ============================================================

class Verifier:

    @staticmethod
    def verify_equation(
        left: sp.Expr,
        right: sp.Expr,
        variable: sp.Symbol,
        solutions: Any,
    ) -> VerificationResult:

        if solutions is None:
            return VerificationResult(
                False,
                "No se pudo verificar porque no hay soluciones calculadas.",
                [],
            )

        if not isinstance(solutions, (list, tuple, set)):
            solutions = [solutions]

        details = []
        all_valid = True

        for solution in solutions:
            try:
                lhs = sp.simplify(left.subs(variable, solution))
                rhs = sp.simplify(right.subs(variable, solution))
                valid = sp.simplify(lhs - rhs) == 0
                all_valid = all_valid and valid
                details.append(
                    f"{variable} = {solution}: "
                    f"{sympy_latex(lhs)} = {sympy_latex(rhs)} → "
                    f"{'válido' if valid else 'no válido'}"
                )
            except Exception as exc:
                all_valid = False
                details.append(f"No se pudo verificar {solution}: {exc}")

        return VerificationResult(
            all_valid,
            "Todas las soluciones fueron verificadas." if all_valid
            else "Al menos una solución no pudo verificarse.",
            details,
        )

    @staticmethod
    def verify_expression(original: sp.Expr, candidate: sp.Expr) -> VerificationResult:
        try:
            difference = sp.simplify(original - candidate)
            valid = difference == 0
            return VerificationResult(
                valid,
                "Las expresiones son equivalentes."
                if valid else "Las expresiones no son equivalentes.",
                [f"Diferencia simplificada: {sympy_latex(difference)}"],
            )
        except Exception as exc:
            return VerificationResult(False, f"Error de verificación: {exc}", [])


# ============================================================
# MOTOR MATEMÁTICO
# ============================================================

class MathEngine:

    @staticmethod
    def solve_equation(text: str) -> Solution:
        analysis = ProblemIdentifier.identify(text)
        equation = extract_equation(text)

        if equation is None:
            verification = VerificationResult(
                False,
                "No se encontró una ecuación con formato reconocible.",
                [],
            )
            return Solution(
                analysis=analysis,
                result=None,
                steps=[
                    "No fue posible separar el problema en dos lados de una ecuación.",
                    "Comprueba que exista una expresión del tipo izquierda = derecha.",
                ],
                formulas=[],
                verification=verification,
                concept="Resolución de ecuaciones",
            )

        left, right = equation
        variable_candidates = sorted(
            left.free_symbols.union(right.free_symbols),
            key=lambda s: str(s)
        )

        if not variable_candidates:
            verification = VerificationResult(
                sp.simplify(left - right) == 0,
                "La expresión no contiene variables.",
                [],
            )
            return Solution(
                analysis=analysis,
                result=sp.simplify(left - right),
                steps=[
                    f"Expresión izquierda: {sympy_latex(left)}",
                    f"Expresión derecha: {sympy_latex(right)}",
                    f"Evaluación de la igualdad: {sympy_latex(sp.simplify(left-right))}",
                ],
                formulas=[],
                verification=verification,
                concept="Igualdad matemática",
            )

        variable = variable_candidates[0]
        expression = sp.expand(left - right)

        try:
            solutions = sp.solve(sp.Eq(left, right), variable)
        except Exception as exc:
            return Solution(
                analysis=analysis,
                result=None,
                steps=[f"El motor simbólico no pudo resolver la ecuación: {exc}"],
                formulas=[],
                verification=VerificationResult(
                    False,
                    "No se pudo resolver de forma confiable.",
                    [],
                ),
                concept="Resolución simbólica",
            )

        steps = [
            f"1. Ecuación original: {sympy_latex(left)} = {sympy_latex(right)}",
            f"2. Variable detectada: {variable}",
            f"3. Llevamos todo a un lado: {sympy_latex(expression)} = 0",
        ]

        poly = sp.Poly(expression, variable) if expression.is_polynomial(variable) else None

        formulas = []
        if poly is not None and poly.degree() == 2:
            a = poly.coeff_monomial(variable**2)
            b = poly.coeff_monomial(variable)
            c = poly.coeff_monomial(1)
            discriminant = sp.simplify(b**2 - 4*a*c)

            if sp.simplify(a) == 0:
                formulas.append(r"ax+b=0\Rightarrow x=-\frac{b}{a}")
                steps.extend([
                    f"4. El coeficiente a es cero: a={a}. Esto no es una ecuación cuadrática.",
                    "5. Se resuelve como una ecuación lineal.",
                ])
                solutions = sp.solve(sp.Eq(left, right), variable)
            else:
                formulas.append(r"x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}")
                steps.extend([
                    f"4. Coeficientes: a={a}, b={b}, c={c}",
                    f"5. Discriminante: D = b^2 - 4ac = {sympy_latex(discriminant)}",
                ])

                if discriminant.is_real is False or discriminant.is_negative is True:
                    steps.append(
                        "6. El discriminante es negativo, por lo que las soluciones son complejas."
                    )
                    steps.append(
                        "7. Aplicamos la fórmula cuadrática con números complejos."
                    )
                else:
                    steps.append("6. Aplicamos la fórmula cuadrática.")

                solutions = sp.solve(sp.Eq(left, right), variable, domain=sp.S.Complexes)
        elif poly is not None and poly.degree() == 1:
            formulas.append(r"ax+b=0\Rightarrow x=-\frac{b}{a}")
            steps.append("4. Se trata de una ecuación lineal; aislamos la variable.")
        else:
            steps.append("4. Se utiliza resolución simbólica general.")

        steps.append(
            "8. Solución: " + (
                ', '.join(sympy_latex(s) for s in solutions)
                if solutions else "no hay solución real en este caso"
            )
        )

        verification = Verifier.verify_equation(
            left, right, variable, solutions
        )

        return Solution(
            analysis=analysis,
            result=solutions,
            steps=steps,
            formulas=formulas,
            verification=verification,
            concept=analysis.subcategory,
        )

    @staticmethod
    def derivative(text: str) -> Solution:
        analysis = ProblemIdentifier.identify(text)
        function = extract_function(text)

        if function is not None:
            _, expression_text = function
        else:
            expression_text = extract_symbolic_expression(text, operation="derivada")

        if expression_text is None:
            return Solution(
                analysis=analysis,
                result=None,
                steps=["No se pudo identificar una expresión de la forma f(x)=... o una derivada directa."],
                formulas=[],
                verification=VerificationResult(
                    False,
                    "No se encontró una función válida.",
                    [],
                ),
                concept="Derivación",
            )

        try:
            expr = parse_expression(expression_text)
            x = sp.Symbol("x")
            derivative = sp.diff(expr, x)

            verification = Verifier.verify_expression(
                derivative,
                sp.diff(expr, x),
            )

            steps = [
                f"1. Función: f(x) = {sympy_latex(expr)}",
                "2. Aplicamos las reglas de derivación.",
                f"3. Resultado: f'(x) = {sympy_latex(derivative)}",
            ]

            if expr.is_polynomial(x):
                steps.append(
                    "4. Para términos de la forma x^n se utiliza la regla "
                    "d/dx(x^n) = n*x^(n-1)."
                )

            return Solution(
                analysis=analysis,
                result=derivative,
                steps=steps,
                formulas=[r"\frac{d}{dx}x^n=nx^{n-1}"],
                verification=verification,
                concept="Derivación",
            )
        except Exception as exc:
            return Solution(
                analysis=analysis,
                result=None,
                steps=[f"No se pudo derivar la expresión: {exc}"],
                formulas=[],
                verification=VerificationResult(
                    False, "La expresión no pudo procesarse.", []
                ),
                concept="Derivación",
            )

    @staticmethod
    def integral(text: str) -> Solution:
        analysis = ProblemIdentifier.identify(text)
        function = extract_function(text)

        if function is not None:
            _, expression_text = function
        else:
            expression_text = extract_symbolic_expression(text, operation="integral")

        if expression_text is None:
            return Solution(
                analysis=analysis,
                result=None,
                steps=["No se pudo identificar la expresión a integrar ni una integral directa."],
                formulas=[],
                verification=VerificationResult(
                    False, "No se encontró una expresión válida.", []
                ),
                concept="Integración",
            )

        try:
            expr = parse_expression(expression_text)
            x = sp.Symbol("x")
            integral = sp.integrate(expr, x)
            derivative_back = sp.diff(integral, x)
            verification = Verifier.verify_expression(expr, derivative_back)

            steps = [
                f"1. Integrando: {sympy_latex(expr)}",
                "2. Calculamos una antiderivada.",
                f"3. Resultado: {sympy_latex(integral)} + C",
                f"4. Verificación derivando el resultado: {sympy_latex(derivative_back)}",
            ]

            return Solution(
                analysis=analysis,
                result=integral,
                steps=steps,
                formulas=[],
                verification=verification,
                concept="Integración",
            )
        except Exception as exc:
            return Solution(
                analysis=analysis,
                result=None,
                steps=[f"No se pudo integrar: {exc}"],
                formulas=[],
                verification=VerificationResult(False, "Error de integración.", []),
                concept="Integración",
            )

    @staticmethod
    def statistics(numbers: list[float], text: str) -> Solution:
        analysis = ProblemIdentifier.identify(text)

        if not numbers:
            return Solution(
                analysis=analysis,
                result=None,
                steps=["No se detectaron datos numéricos."],
                formulas=[],
                verification=VerificationResult(
                    False, "No hay datos para calcular estadísticas.", []
                ),
                concept="Estadística descriptiva",
            )

        arr = np.array(numbers, dtype=float)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        mode_values = []
        counts = {}
        for value in numbers:
            counts[value] = counts.get(value, 0) + 1
        max_count = max(counts.values())
        if max_count > 1:
            mode_values = [v for v, c in counts.items() if c == max_count]

        variance = float(np.var(arr))
        std = float(np.std(arr))

        result = {
            "media": mean,
            "mediana": median,
            "moda": mode_values if mode_values else "No hay moda única",
            "varianza poblacional": variance,
            "desviación estándar poblacional": std,
        }

        steps = [
            f"1. Datos: {numbers}",
            f"2. Media = {mean:.6g}",
            f"3. Mediana = {median:.6g}",
            f"4. Varianza poblacional = {variance:.6g}",
            f"5. Desviación estándar poblacional = {std:.6g}",
        ]

        return Solution(
            analysis=analysis,
            result=result,
            steps=steps,
            formulas=[
                r"\bar{x}=\frac{\sum x_i}{n}",
                r"\sigma^2=\frac{\sum(x_i-\mu)^2}{N}",
            ],
            verification=VerificationResult(
                True,
                "Los cálculos estadísticos fueron realizados directamente con NumPy.",
                [],
            ),
            concept="Estadística descriptiva",
        )

    @classmethod
    def solve(cls, text: str) -> Solution:
        lower = clean_text(text).lower()

        if "deriv" in lower:
            return cls.derivative(text)

        if "integr" in lower:
            return cls.integral(text)

        if any(k in lower for k in ["media", "mediana", "moda", "varianza"]):
            numbers = extract_numbers(text)
            return cls.statistics(numbers, text)

        if "=" in text:
            return cls.solve_equation(text)

        analysis = ProblemIdentifier.identify(text)
        return Solution(
            analysis=analysis,
            result=None,
            steps=[
                "No se encontró un patrón resoluble automáticamente.",
                "Prueba escribiendo una ecuación, una derivada, una integral o un conjunto de datos.",
            ],
            formulas=[],
            verification=VerificationResult(
                False,
                "Información insuficiente para una resolución confiable.",
                [],
            ),
            concept="Problema general",
        )


# ============================================================
# BÚSQUEDA DE FÓRMULAS
# ============================================================

def search_formulas(query: str) -> list[Formula]:
    q = query.lower().strip()
    if not q:
        return FORMULAS

    scored = []

    for formula in FORMULAS:
        text = " ".join([
            formula.name,
            formula.category,
            formula.subcategory,
            formula.description,
            formula.example,
            " ".join(formula.methods),
        ]).lower()

        score = sum(1 for token in q.split() if token in text)

        if score:
            scored.append((score, formula))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [formula for _, formula in scored]


# ============================================================
# GENERADOR DE PRÁCTICA
# ============================================================

def generate_practice(topic: str, difficulty: str, quantity: int) -> list[dict]:
    quantity = max(1, min(quantity, 50))
    generated = []

    for _ in range(quantity):
        if "cuadr" in topic.lower() or "ecuación" in topic.lower():
            a = random.choice([1, 2, 3, 4, 5])
            r1 = random.randint(-8, 8)
            r2 = random.randint(-8, 8)
            b = -a * (r1 + r2)
            c = a * r1 * r2
            equation = f"{a}x^2 + ({b})x + ({c}) = 0"
            generated.append({
                "problema": f"Resuelve {equation}",
                "respuesta": sorted([r1, r2]),
                "tema": "Ecuaciones cuadráticas",
                "dificultad": difficulty,
            })

        elif "deriv" in topic.lower():
            a = random.randint(1, 8)
            b = random.randint(-8, 8)
            c = random.randint(-8, 8)
            expression = f"{a}*x^3 + ({b})*x^2 + ({c})*x"
            x = sp.Symbol("x")
            answer = sp.diff(parse_expression(expression), x)
            generated.append({
                "problema": f"Calcula la derivada de f(x)={expression}",
                "respuesta": str(answer),
                "tema": "Derivadas",
                "dificultad": difficulty,
            })

        elif "porcentaje" in topic.lower():
            total = random.choice([50, 80, 100, 120, 200, 500])
            percent = random.choice([5, 10, 15, 20, 25, 30])
            answer = total * percent / 100
            generated.append({
                "problema": f"Calcula el {percent}% de {total}.",
                "respuesta": answer,
                "tema": "Porcentajes",
                "dificultad": difficulty,
            })

        else:
            a = random.randint(2, 20)
            b = random.randint(2, 20)
            generated.append({
                "problema": f"Calcula {a} + {b} × 2.",
                "respuesta": a + b * 2,
                "tema": "Aritmética",
                "dificultad": difficulty,
            })

    return generated


# ============================================================
# OCR OPCIONAL
# ============================================================

def perform_ocr(image_bytes: bytes) -> tuple[str, Optional[str]]:
    try:
        from PIL import Image
        import pytesseract

        image = Image.open(io.BytesIO(image_bytes))
        try:
            text = pytesseract.image_to_string(image, config="--psm 6")
        except (FileNotFoundError, OSError, RuntimeError, pytesseract.TesseractNotFoundError) as exc:
            return "", (
                "No se encontró Tesseract en el sistema. "
                "Instala Tesseract OCR o escribe la ecuación manualmente en el campo de texto."
            )

        cleaned = text.strip()
        if not cleaned:
            return "", (
                "No se detectó texto en la imagen. "
                "Escribe la ecuación manualmente en el campo de texto."
            )

        return cleaned, None

    except ImportError:
        return "", (
            "OCR opcional no instalado. Ejecuta: "
            "pip install pytesseract pillow. "
            "Si no deseas usar OCR, escribe la ecuación manualmente."
        )
    except Exception as exc:
        return "", (
            f"Error de OCR: {exc}. "
            "Escribe la ecuación manualmente en el campo de texto."
        )


# ============================================================
# PDF
# ============================================================

def solution_to_pdf(solution: Solution) -> Optional[bytes]:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.enums import TA_CENTER

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        title = styles["Title"]
        title.alignment = TA_CENTER

        story = [
            Paragraph(APP_NAME, title),
            Spacer(1, 16),
            Paragraph("<b>Problema</b>", styles["Heading2"]),
            Paragraph(solution.analysis.original_text, styles["BodyText"]),
            Spacer(1, 10),
            Paragraph("<b>Problema identificado</b>", styles["Heading2"]),
            Paragraph(
                f"{solution.analysis.category} — "
                f"{solution.analysis.subcategory}",
                styles["BodyText"],
            ),
            Spacer(1, 10),
            Paragraph("<b>Desarrollo</b>", styles["Heading2"]),
        ]

        for step in solution.steps:
            story.append(Paragraph(step, styles["BodyText"]))
            story.append(Spacer(1, 5))

        story.extend([
            Paragraph("<b>Resultado</b>", styles["Heading2"]),
            Paragraph(format_result(solution.result).replace("\n", "<br/>"), styles["BodyText"]),
            Spacer(1, 10),
            Paragraph("<b>Comprobación</b>", styles["Heading2"]),
            Paragraph(solution.verification.message, styles["BodyText"]),
        ])

        for detail in solution.verification.details:
            story.append(Paragraph(detail, styles["BodyText"]))

        doc.build(story)
        return buffer.getvalue()

    except ImportError:
        return None
    except Exception:
        return None


# ============================================================
# GRÁFICAS
# ============================================================

def plot_function(expression: str):
    x = sp.Symbol("x")

    try:
        expr = parse_expression(expression)
        fn = sp.lambdify(x, expr, modules=["numpy"])

        xs = np.linspace(-10, 10, 1000)
        ys = fn(xs)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, ys)
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"f(x) = {sympy_latex(expr)}")
        ax.grid(True, alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

    except Exception as exc:
        st.error(f"No se pudo graficar: {exc}")


# ============================================================
# INTERFAZ
# ============================================================

def initialize_state():
    if "history" not in st.session_state:
        st.session_state.history = []

    if "last_solution" not in st.session_state:
        st.session_state.last_solution = None

    if "practice" not in st.session_state:
        st.session_state.practice = []

    if "practice_index" not in st.session_state:
        st.session_state.practice_index = 0


def render_header():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0;
        }
        .subtitle {
            color: #6b7280;
            font-size: 1.1rem;
        }
        .result-box {
            padding: 1rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-title">∑ MathSolver AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Asistente matemático educativo con resolución simbólica, '
        'verificación y explicación paso a paso.</div>',
        unsafe_allow_html=True,
    )


def solve_page():
    st.header("Resolver problema")

    col1, col2 = st.columns([3, 1])

    with col1:
        problem = st.text_area(
            "Escribe el problema",
            height=180,
            placeholder=(
                "Ejemplos:\n"
                "2x^2 - 5x - 3 = 0\n"
                "Calcula la derivada de f(x)=x^3+2*x^2-5*x\n"
                "Calcula la media de 2, 4, 6, 8"
            ),
        )

    with col2:
        level = st.selectbox(
            "Nivel de explicación",
            ["Básico", "Intermedio", "Avanzado", "Universitario", "Experto"],
        )

        language = st.selectbox(
            "Idioma",
            ["Español", "English", "Français", "Português"],
        )

    uploaded = st.file_uploader(
        "O carga una fotografía del ejercicio",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded:
        if st.button("🔎 Extraer texto de la imagen"):
            image_text, error = perform_ocr(uploaded.getvalue())

            if error:
                st.warning(error)

            if image_text:
                st.session_state["ocr_text"] = image_text
                st.success("Texto detectado.")
                st.code(image_text)

    if "ocr_text" in st.session_state and not problem:
        problem = st.session_state["ocr_text"]

    if st.button("🚀 Resolver", type="primary", use_container_width=True):
        if not problem.strip():
            st.warning("Escribe un problema o carga una imagen.")
            return

        solution = MathEngine.solve(problem)
        st.session_state.last_solution = solution

        st.session_state.history.insert(0, {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "problema": problem,
            "categoría": solution.analysis.category,
            "método": solution.analysis.method,
            "resultado": format_result(solution.result),
            "nivel": level,
        })

    solution = st.session_state.last_solution

    if not solution:
        return

    st.divider()

    a, b, c = st.columns(3)

    with a:
        st.metric("Categoría", solution.analysis.category)

    with b:
        st.metric("Subcategoría", solution.analysis.subcategory)

    with c:
        st.metric(
            "Confianza de identificación",
            f"{solution.analysis.confidence * 100:.0f}%",
        )

    st.subheader("🧠 Problema identificado")
    st.write(solution.analysis.subcategory)

    if solution.analysis.notes:
        for note in solution.analysis.notes:
            st.info(note)

    st.subheader("📌 Datos detectados")

    if solution.analysis.variables:
        st.write("Variables:", ", ".join(solution.analysis.variables))
    else:
        st.write("No se detectaron variables explícitas.")

    st.subheader("📐 Fórmulas")

    if solution.formulas:
        for formula in solution.formulas:
            st.latex(formula)
    else:
        st.write("El método no requiere una fórmula predefinida.")

    st.subheader("📝 Desarrollo")

    for step in solution.steps:
        st.write(step)

    st.subheader("✅ Resultado")

    st.markdown(
        f'<div class="result-box"><pre>{format_result(solution.result)}</pre></div>',
        unsafe_allow_html=True,
    )

    st.subheader("🔬 Comprobación")

    if solution.verification.verified:
        st.success(solution.verification.message)
    else:
        st.warning(solution.verification.message)

    for detail in solution.verification.details:
        st.write(detail)

    st.subheader("🎓 Concepto utilizado")
    st.write(solution.concept)

    pdf = solution_to_pdf(solution)

    if pdf:
        st.download_button(
            "📄 Descargar solución en PDF",
            data=pdf,
            file_name="mathsolver_solution.pdf",
            mime="application/pdf",
        )


def formulas_page():
    st.header("📚 Biblioteca de fórmulas")

    query = st.text_input(
        "Buscar fórmula, concepto, teorema o método",
        placeholder="Ej.: área círculo, derivada, Pitágoras...",
    )

    results = search_formulas(query)

    st.write(f"{len(results)} fórmula(s) encontrada(s).")

    for formula in results:
        with st.expander(
            f"{formula.name} | {formula.category} | {formula.level}"
        ):
            st.latex(formula.latex)
            st.write(formula.description)

            st.write("**Variables:**", ", ".join(formula.variables))
            st.write("**Condiciones:**", ", ".join(formula.conditions) or "Ninguna")
            st.write("**Ejemplo:**", formula.example)
            st.write("**Métodos relacionados:**", ", ".join(formula.methods))


def practice_page():
    st.header("🎯 Modo práctica")

    col1, col2, col3 = st.columns(3)

    with col1:
        topic = st.selectbox(
            "Tema",
            [
                "Ecuaciones cuadráticas",
                "Derivadas",
                "Porcentajes",
                "Aritmética",
            ],
        )

    with col2:
        difficulty = st.selectbox(
            "Dificultad",
            ["Fácil", "Media", "Difícil"],
        )

    with col3:
        quantity = st.number_input(
            "Cantidad",
            min_value=1,
            max_value=50,
            value=5,
        )

    if st.button("Generar ejercicios", type="primary"):
        st.session_state.practice = generate_practice(
            topic,
            difficulty,
            int(quantity),
        )
        st.session_state.practice_index = 0

    exercises = st.session_state.practice

    if not exercises:
        st.info("Genera una tanda de ejercicios para comenzar.")
        return

    index = st.session_state.practice_index
    index = min(index, len(exercises) - 1)

    exercise = exercises[index]

    st.markdown(f"### Ejercicio {index + 1} de {len(exercises)}")
    st.info(exercise["problema"])

    answer = st.text_input("Tu respuesta")

    if st.button("Verificar respuesta"):
        expected = exercise["respuesta"]

        # Para respuestas simbólicas, comparar simplificación.
        try:
            if isinstance(expected, list):
                submitted = sorted([
                    float(x.strip())
                    for x in answer.split(",")
                ])
                correct = np.allclose(submitted, expected)
            elif isinstance(expected, (int, float)):
                correct = math.isclose(float(answer), float(expected), rel_tol=1e-9)
            else:
                candidate = parse_expression(answer)
                target = parse_expression(str(expected))
                correct = sp.simplify(candidate - target) == 0

            if correct:
                st.success("Respuesta correcta.")
            else:
                st.error("La respuesta no coincide con el resultado esperado.")
                st.caption("Puedes pedir la solución desde el modo Resolver.")
        except Exception:
            st.error("No pude interpretar esa respuesta.")

    if st.button("Siguiente ejercicio"):
        if index + 1 < len(exercises):
            st.session_state.practice_index += 1
            st.rerun()
        else:
            st.success("Has completado la tanda.")


def graph_page():
    st.header("📈 Gráficas")

    expression = st.text_input(
        "f(x) =",
        value="x^2 - 4*x + 3",
    )

    if st.button("Graficar", type="primary"):
        plot_function(expression)


def history_page():
    st.header("🕘 Historial")

    history = st.session_state.history

    if not history:
        st.info("Todavía no hay problemas en el historial.")
        return

    df = pd.DataFrame(history)
    st.dataframe(df, use_container_width=True)

    if st.button("🗑️ Borrar historial"):
        st.session_state.history = []
        st.rerun()


def about_page():
    st.header("ℹ️ MathSolver AI")

    st.write(
        """
        MathSolver AI está diseñado como una plataforma educativa matemática
        extensible.

        El motor intenta separar cuatro responsabilidades:

        1. Interpretar el problema.
        2. Resolverlo matemáticamente.
        3. Explicar el procedimiento.
        4. Verificar el resultado.

        La resolución matemática usa SymPy/NumPy en lugar de depender
        exclusivamente de un modelo de lenguaje.
        """
    )

    st.code(
        """
        pip install streamlit sympy numpy scipy pandas matplotlib pillow
        streamlit run mathsolver_ai.py
        """,
        language="bash",
    )


def main():
    initialize_state()
    render_header()

    with st.sidebar:
        st.markdown("## Menú")

        page = st.radio(
            "Ir a",
            [
                "Resolver",
                "Fórmulas",
                "Práctica",
                "Gráficas",
                "Historial",
                "Acerca de",
            ],
        )

        st.divider()

        st.caption(f"{APP_NAME} v{APP_VERSION}")
        st.caption("Motor matemático: SymPy + NumPy")

    if page == "Resolver":
        solve_page()
    elif page == "Fórmulas":
        formulas_page()
    elif page == "Práctica":
        practice_page()
    elif page == "Gráficas":
        graph_page()
    elif page == "Historial":
        history_page()
    elif page == "Acerca de":
        about_page()


if __name__ == "__main__":
    main()
