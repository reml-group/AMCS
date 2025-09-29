
import re
import multiprocessing
from math import isclose
from typing import Union

from sympy import simplify, N, sympify
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex


def is_digit(s):
    try:
        float(str(s).replace(",", ""))
        return True
    except ValueError:
        return False


def math_equal(prediction: Union[bool, float, str],
                reference: Union[float, str],
                include_percentage: bool = True,
                is_close_val: bool = True,
                use_timeout: bool = True,
                timeout_duration: int = 3
                ) -> bool:
    try:
        pred_str = re.sub(r"\s+", "", str(prediction).strip())
        ref_str  = re.sub(r"\s+", "", str(reference).strip())

        if pred_str == ref_str:
            return True

        eq = False
        if use_timeout:
            eq = call_with_timeout(symbolic_equal_process, pred_str, ref_str, timeout=timeout_duration)

        if eq or (not use_timeout and symbolic_equal(pred_str, ref_str)) or (use_timeout and symbolic_equal(pred_str, ref_str)):
            return True

        if is_digit(pred_str) and is_digit(ref_str):
            pred_num = float(pred_str.replace(",", ""))
            ref_num  = float(ref_str.replace(",", ""))
            candidates = [ref_num]
            if include_percentage:
                candidates.extend([ref_num / 100, ref_num * 100])
            for val in candidates:
                if is_close_val and isclose(val, pred_num, rel_tol=1e-4):
                    return True
                if not is_close_val and val == pred_num:
                    return True
        return False
    except Exception:
        return False


def symbolic_equal(a: str, b: str) -> bool:
 
    def _parse(expr_str: str):
        s = expr_str.strip()
        s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)', s)
        s = s.replace(r"\pi", "pi")
        if s.startswith('$$') and s.endswith('$$'):
            s = s[2:-2]
        elif s.startswith('$') and s.endswith('$'):
            s = s[1:-1]
        elif s.startswith(r'\\(') and s.endswith(r'\\)'):
            s = s[2:-2]
        for parser in (parse_latex, parse_expr):
            try:
                return parser(s)
            except Exception:
                continue
        try:
            return sympify(s)
        except Exception:
            return s

    a_val = _parse(a)
    b_val = _parse(b)
    try:
        if isclose(N(a_val), N(b_val), rel_tol=1e-3):
            return True
    except Exception:
        pass
    try:
        if simplify(a_val - b_val) == 0:
            return True
    except Exception:
        pass
    return False


def symbolic_equal_process(a: str, b: str, output_queue: multiprocessing.Queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout: int = 3, **kwargs) -> bool:
    output_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=func, args=(*args, output_queue), kwargs=kwargs)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
    if output_queue.empty():
        return False
    return output_queue.get()


# Self-tests
if __name__ == "__main__":
    tests = [
        ('\\frac{1}{8}', '\\frac{1}{8}'),
        ('0.5', '\\frac{1}{2}'),
        ('64 - 16\\pi', '64-16\\pi'),
        ('1+sqrt(2)', 'sqrt(2)+1'),
        ('x^2+2x+1', 'x^2 + 1 + 2*x'),
    ]
    for p, r in tests:
        print(f"{p} == {r}? ->", math_equal(p, r))

