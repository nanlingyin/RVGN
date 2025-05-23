import re
import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
from sympy.core.sympify import SympifyError
from typing import Dict, List, Any, Optional, Tuple

class MathValidator:
    """数学推理验证器 - 使用SymPy检查数学推理中的错误"""
    
    def __init__(self):
        """初始化数学验证器"""
        # 配置SymPy解析器的转换选项，使其更宽松地处理输入
        self.transformations = (
            standard_transformations +
            (implicit_multiplication_application, convert_xor)
        )
        
    def validate_expressions(self, text: str) -> List[Dict]:
        """
        验证文本中的数学表达式和等式
        
        参数:
            text: 包含数学推理的文本
            
        返回:
            验证结果列表
        """
        results = []
        
        # 提取和验证不同类型的数学表达式
        results.extend(self._validate_equations(text))
        results.extend(self._validate_derivatives(text))
        results.extend(self._validate_integrals(text))
        
        return results
    
    def _validate_equations(self, text: str) -> List[Dict]:
        """验证文本中的等式"""
        results = []
        
        # 正则表达式模式匹配各种等式格式
        equation_pattern = r'([^=<>]+?)\s*(=|==)\s*([^=\n\r]+)'
        equations = re.finditer(equation_pattern, text)
        
        for eq_match in equations:
            left_side = eq_match.group(1).strip()
            right_side = eq_match.group(3).strip()
            
            # 尝试使用SymPy进行符号验证
            validation_result = self._check_equation_validity(left_side, right_side)
            
            results.append({
                "equation": f"{left_side} = {right_side}",
                "location": (eq_match.start(), eq_match.end()),
                "is_checked": True,
                "is_valid": validation_result[0],
                "note": validation_result[1]
            })
                
        return results
    
    def _validate_derivatives(self, text: str) -> List[Dict]:
        """验证文本中的导数计算"""
        results = []
        
        # 正则表达式匹配导数表达式
        # 匹配形如 "d/dx(x^2) = 2x" 或 "导数 d(x^3)/dx = 3x^2" 的表达式
        derivative_pattern = r'(?:d|∂)(?:\s*?[/]?\s*?d([a-zA-Z])|\(\s*([a-zA-Z])\s*\))(?:\s*\(\s*([^)]+)\s*\)|\s*([^=]+))\s*=\s*([^=\n\r]+)'
        derivatives = re.finditer(derivative_pattern, text)
        
        for deriv_match in derivatives:
            # 提取变量和表达式
            var = deriv_match.group(1) or deriv_match.group(2)
            expr = deriv_match.group(3) or deriv_match.group(4)
            result = deriv_match.group(5)
            
            validation_result = self._check_derivative_validity(expr, var, result)
            
            results.append({
                "equation": f"d/d{var}({expr}) = {result}",
                "location": (deriv_match.start(), deriv_match.end()),
                "is_checked": True,
                "is_valid": validation_result[0],
                "note": validation_result[1]
            })
                
        return results
    
    def _validate_integrals(self, text: str) -> List[Dict]:
        """验证文本中的积分计算"""
        results = []
        
        # 正则表达式匹配积分表达式
        # 匹配形如 "∫x^2 dx = x^3/3 + C" 或 "积分 x^2 dx = x^3/3 + C" 的表达式
        integral_pattern = r'(?:∫|积分)(?:\s*([^=\n\r]+?)\s*d([a-zA-Z]))\s*=\s*([^=\n\r]+)'
        integrals = re.finditer(integral_pattern, text)
        
        for int_match in integrals:
            expr = int_match.group(1).strip()
            var = int_match.group(2)
            result = int_match.group(3)
            
            validation_result = self._check_integral_validity(expr, var, result)
            
            results.append({
                "equation": f"∫{expr} d{var} = {result}",
                "location": (int_match.start(), int_match.end()),
                "is_checked": True,
                "is_valid": validation_result[0],
                "note": validation_result[1]
            })
                
        return results
        
    def _check_equation_validity(self, left_side: str, right_side: str) -> Tuple[bool, str]:
        """
        使用SymPy检查等式的有效性
        
        参数:
            left_side: 等式左侧字符串
            right_side: 等式右侧字符串
            
        返回:
            (is_valid, message): 验证结果和消息
        """
        try:
            # 尝试解析两边的表达式
            left_expr = parse_expr(left_side, transformations=self.transformations)
            right_expr = parse_expr(right_side, transformations=self.transformations)
            
            # 检查两边是否相等
            difference = sympy.simplify(left_expr - right_expr)
            is_valid = difference == 0
            
            if is_valid:
                return True, "等式有效：左右两边在符号上相等"
            else:
                return False, f"等式无效：左右两边不相等，差值为 {difference}"
                
        except SympifyError as e:
            return False, f"无法解析表达式：{str(e)}"
        except Exception as e:
            return False, f"验证过程出错：{str(e)}"
    
    def _check_derivative_validity(self, expr_str: str, var: str, result_str: str) -> Tuple[bool, str]:
        """
        验证导数计算是否正确
        """
        try:
            # 解析表达式和结果
            expr = parse_expr(expr_str, transformations=self.transformations)
            result = parse_expr(result_str, transformations=self.transformations)
            
            # 计算导数
            var_sym = sympy.Symbol(var)
            derivative = sympy.diff(expr, var_sym)
            
            # 检查计算结果是否与给定结果相等
            difference = sympy.simplify(derivative - result)
            is_valid = difference == 0
            
            if is_valid:
                return True, "导数计算正确"
            else:
                return False, f"导数计算错误，正确结果为 {derivative}"
                
        except SympifyError as e:
            return False, f"无法解析表达式：{str(e)}"
        except Exception as e:
            return False, f"验证过程出错：{str(e)}"
    
    def _check_integral_validity(self, expr_str: str, var: str, result_str: str) -> Tuple[bool, str]:
        """
        验证积分计算是否正确
        """
        try:
            # 解析表达式和结果
            expr = parse_expr(expr_str, transformations=self.transformations)
            
            # 处理结果中可能存在的常数项C
            result_str = re.sub(r'[+\-]\s*C', '', result_str)
            result = parse_expr(result_str, transformations=self.transformations)
            
            # 计算积分的导数，应与原表达式相等
            var_sym = sympy.Symbol(var)
            derivative_of_result = sympy.diff(result, var_sym)
            
            # 比较原表达式与结果的导数
            difference = sympy.simplify(expr - derivative_of_result)
            is_valid = difference == 0
            
            if is_valid:
                return True, "积分计算正确"
            else:
                correct_integral = sympy.integrate(expr, var_sym)
                return False, f"积分计算错误，正确结果为 {correct_integral} + C"
                
        except SympifyError as e:
            return False, f"无法解析表达式：{str(e)}"
        except Exception as e:
            return False, f"验证过程出错：{str(e)}"