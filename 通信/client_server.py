# -*- coding: utf-8 -*-

import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1",10001))
server.listen(50)
while True:
    data,addr = server.accept()
    info = b'OK'
    while True:
        buffer = data.recv(2)
        if buffer == b'':
            info = b'Bad request'
            break
    data.send(info)
    data.close()


