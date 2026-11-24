"""
This module provides basic calculator functions:

- Addition

- Subtraction

- Multiplication

- Division

"""

def sum(x, y):
    """
    Calculate the sum of two numbers.
    
    Args:
        x (int or float): the first number
        y (int or float): the second number
    
    Returns:
        int or float: the sum of x and y
    """
    return x + y

def subtract(x, y):
    """
    Calculate the subtraction of y from x.
    
    Args:
        x (int or float): the minuend
        y (int or float): the subtraend
    
    Returns:
        int or float: the difference between x and y
    """
    return x - y

def multiply(x, y):
    
    """
    Calculate the product of two numbers.
    
    Args:
        x (int or float): the multiplicand
        y (int or float): the multiplier
        
    Returns:
        int or float: the product of x and y
    """
    return x * y

def divide(x, y):
    """
    Calculate the quotient of two numbers.
    
    Args:
        x (int or float): the dividend
        y (int or float): the divisor
        
    Returns:
        int or float: the quotient of x and y
    Raises:
        ZeroDivisionError: error when the divisor is 0
    """
    return x / y
