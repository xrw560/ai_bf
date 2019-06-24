# -*- conding:utf-8 -*-

import time
import threading

# 单线程
''' '''
# def music(name, loop):
#     for i in range(loop):
#         print('listen music %s %s' % (name, time.ctime()))
#         time.sleep(1)
#
#
# def movie(name, loop):
#     for i in range(loop):
#         print('look movie %s %s' % (name, time.ctime()))
#         time.sleep(1)
#
#
# if __name__ == '__main__':
#     music('爱的故事上集', 3)
#     movie('晓生克的救赎', 4)
#     print('end time %s' % time.ctime())

# 多线程
''' '''
# def music(name,loop):
#     for i in range(loop):
#         print('listen music %s %s %s'% (name, time.ctime(), threading.Thread.getName(t1)))
#         time.sleep(1)
#
# def movie(name,loop):
#     for i in range(loop):
#         print('look movie %s %s %s'% (name, time.ctime(), threading.Thread.getName(t2)))
#         time.sleep(1)
#
#
# #1.创建多线程
# t1 = threading.Thread(target= music, args=('爱的故事上集', 4))
# t1.setName('musicThread')
# t2 = threading.Thread(target= movie, args=('肖生克的救赎', 4), name = 'movieThread')
#
# if __name__ == '__main__':
#     # #4.守护主线程，主线程结束杀死子线程
#     t1.setDaemon(True)
#     t2.setDaemon(True)
#
#     #2.启动线程
#     t1.start()
#     t2.start()
#     # print('end time %s'%time.ctime())
#
#     # print(t1.ident)
#     # print(t2.ident)
#
#     #3.join 可以对主线程进行阻塞，等所有的子线程运行结束在运行主线程
#     # t1.join()
#     # t2.join()
#
#     print('主线程：%s'%time.ctime())



# 加锁
balance = 0


def change(n):
    global balance
    balance += n
    balance -= n


# def run_thread(n):
#     for i in range(1000000):
#         change(n)

lock = threading.Lock()  # 获取线程锁
#
#
# def run_thread(n):
#     for i in range(1000000):
#         # 获取锁
#         lock.acquire()
#         try:
#             change(n)
#         finally:
#             # 释放锁
#             lock.release()
#
#
# t1 = threading.Thread(target=run_thread, args=(4,))
# t2 = threading.Thread(target=run_thread, args=(8,))
#
# t1.start()
# t2.start()
# t1.join()
# t2.join()
# print(balance)
