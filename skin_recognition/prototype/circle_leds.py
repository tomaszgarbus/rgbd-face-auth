import itertools
import serial
import sys
import time

SLEEP_TIME = 2

ser = serial.Serial('/dev/ttyACM0', 9600)

def send_arduino_command(s):
    print('Sending to Arduino:', s)
    ser.write(bytes(s, 'ascii'))
    ser.flush()

connected = False
while not connected:
    send_arduino_command('C')
    while ser.in_waiting > 0:
        if ser.read() == b'K':
            connected = True
    time.sleep(0.5)
print('Connected')

try:
    for i in itertools.cycle(range(1, 3 + 1)):
        send_arduino_command(str(i))
        time.sleep(SLEEP_TIME)
except KeyboardInterrupt:
    send_arduino_command('0')
