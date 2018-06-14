import datetime
import serial
import socket
import time
import sys

SLEEP_TIME = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

def send_camera_command(s):
    print('Sending to camera:', s)
    sock.sendto(bytes(s, 'ascii'), ('255.255.255.255', 8000))

ser = serial.Serial('/dev/ttyACM0', 9600)

def send_arduino_command(s):
    print('Sending to Arduino:', s)
    ser.write(bytes(s, 'ascii'))
    ser.flush()

photo_id = '{now.minute:02}{now.second:02}'.format(now=datetime.datetime.utcnow())
print('Photo filename:', photo_id)

connected = False
while not connected:
    send_arduino_command('C')
    while ser.in_waiting > 0:
        if ser.read() == b'K':
            connected = True
    time.sleep(0.5)
print('Connected')

for i in range(1, 3 + 1):
    send_arduino_command(str(i))
    time.sleep(SLEEP_TIME / 2)
    send_camera_command('S' + photo_id)
    time.sleep(SLEEP_TIME / 2)

time.sleep(SLEEP_TIME)
send_arduino_command('0')
