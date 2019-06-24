# -*- coding: utf-8 -*-

class CustomException(Exception):
    def __init__(self, err="CustomException"):
        Exception.__init__(self, err)
