import ultrasonic
import RPi.GPIO as GPIO
import time
import warnings

GPIO.setwarnings(False)

TRIG,ECHO = ultrasonic.ultrasonic_setup()
i = 0

while i != 10:
    distance = ultrasonic.ultrasonic_read(TRIG,ECHO)
    print(distance,"cm")
    i += 1
    time.sleep(1)

ultrasonic.ultrasonic_cleanup()