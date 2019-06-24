# -*- coding: utf-8 -*-

from 异常.CustomException import CustomException


class ThreadResult:
    def __init__(self, result, err):
        self.result = result
        self.err = err

    def get_result(self):
        if self.result:
            return self.result
        else:
            raise CustomException(err="threadResultException")
