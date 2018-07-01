import serial
import time
import sys

if len(sys.argv) < 2 or sys.argv[1] not in ['1', '2', '3']:
    print('You need to give an argument which is an integer in {1,2,3}')
    sys.exit(1)

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

send_arduino_command(sys.argv[1])
