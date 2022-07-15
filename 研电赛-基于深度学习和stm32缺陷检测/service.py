import socket
import sys
import time
serversocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# host=socket.gethostname()#获取本地主机名
host = "192.168.4.2"
port=8088
#绑定端口号
serversocket.bind((host,port))
#设置最大连接数
serversocket.listen(5)

while True:
    flag=1
    print('服务器启动，监听客户端链接')
    clientsocket,addr=serversocket.accept()
    print('链接地址：%s' % str(addr))
    while True:
        try:
            data=clientsocket.recv(5*1024)
        except Exception:
            print('断开的客户端：',addr)
            break
        print(data)
        # print('客户端发送内容：',data.decode('utf-8'))
        reply="test"
        if not reply:
            break
        clientsocket.send(reply.encode('utf-8'))
    clientsocket.close()
serversocket.closel()