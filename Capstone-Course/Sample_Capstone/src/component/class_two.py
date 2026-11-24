# -*- coding: utf-8 -*-
"""
Author: Your Name
Date: 2023-11-18
Version: 1.0
"""


class Person:
    def __init__(self, name, age) -> object:
        """

        :rtype: object
        :param name:
        :param age:
        """
        self.name = name
        self.age = age

    def __str__(self):
        """

        :rtype: object
        """
        return f"{self.name}({self.age})"
