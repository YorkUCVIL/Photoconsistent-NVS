import socket
import sys

ip = sys.argv[1]
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((ip, 0))
addr = s.getsockname()
print(addr[1],end='')
s.close()
