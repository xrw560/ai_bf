# -*- coding: utf-8 -*-
from 进程线程.threading.MyThread import MyThread
from 进程线程.threading.ThreadResult import ThreadResult
from 异常.CustomException import CustomException
import time

def cus_print(args):
    try:
        raise CustomException(err="custom")
        print("hello world")
        time.sleep(3)
    except Exception as e:
        print(e)
        return ThreadResult(None, e)
    else:
        return ThreadResult(args, None)


thread_li = []
for i in range(10):
    t = MyThread(cus_print, args=("zhang" + str(i),))
    thread_li.append(t)
    t.start()

for t in thread_li:
    t.join()
    print(t.get_result().get_result())
    break
