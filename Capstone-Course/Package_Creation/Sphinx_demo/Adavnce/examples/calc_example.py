"""
Calculator Example
##################

This is some example code using functions from the calculator module.

"""
import os
import sys
sys.path.insert(0, os.path.abspath("../src"))
from calculator import sum, subtract, multiply, divide

# Example numbers for operations
number1 = 8
number2 = 2

# Using the sum function to add two numbers
result_addition = sum(number1, number2)
print(f"The sum of {number1} and {number2} is {result_addition}.")

# Using the subtract function to subtract the second number from the first
result_subtraction = subtract(number1, number2)
print(f"The result of subtracting {number2} from {number1} is {result_subtraction}.")

# Using the multiply function to multiply two numbers
result_multiplication = multiply(number1, number2)
print(f"The product of {number1} and {number2} is {result_multiplication}.")

# Using the divide function to divide the first number by the second
try:
    result_division = divide(number1, number2)
    print(f"The quotient of {number1} divided by {number2} is {result_division}.")
except ZeroDivisionError:
    print("Cannot divide by zero.")

# Attempt to divide by zero to demonstrate error handling
try:
    zero_division_test = divide(number1, 0)
except ZeroDivisionError:
    print("An error occurred: cannot divide by zero.")