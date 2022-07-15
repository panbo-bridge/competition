import socket
import sys


s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# host="192.168.1.102"
host=socket.gethostname()#获取本地主机名
print(host)
port=8087
s.connect((host,port))
while True:
    data= input('>>').strip()
    if not data:
        break
    s.send(data.encode('utf-8'))
    msg=s.recv(1024)
    if not msg:
        break
    print(msg.decode('utf-8'))
s.close()