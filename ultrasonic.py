from threading import Thread
import RPi.GPIO as GPIO
import time

class Ultrasonic:
    def __init__(self, TRIG, ECHO):
        GPIO.setmode(GPIO.BCM)
        
        self.TRIG = TRIG
        self.ECHO = ECHO

        GPIO.setup(self.TRIG,GPIO.OUT)
        GPIO.setup(self.ECHO,GPIO.IN)

        GPIO.output(TRIG, False)
        
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def read(self):
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        while GPIO.input(ECHO)==0:
            pulse_start = time.time()

        while GPIO.input(ECHO)==1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
    
        distance = pulse_duration * 17150

        distance = round(distance, 2)
        
        return distance
    
    def stop(self):
        self.stopped = True