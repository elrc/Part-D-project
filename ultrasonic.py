import RPi.GPIO as GPIO
import time

def ultrasonic_setup():
    
    GPIO.setmode(GPIO.BCM)
    
    TRIG = 23 
    ECHO = 24
    
    GPIO.setup(TRIG,GPIO.OUT)
    GPIO.setup(ECHO,GPIO.IN)
    
    GPIO.output(TRIG, False)
    time.sleep(2)
    
    return (TRIG,ECHO)

def ultrasonic_read(TRIG,ECHO):
    
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    
    while GPIO.input(ECHO)==0:
        pulse_start = time.time()
    
    while GPIO.input(ECHO)==1:
        pulse_end = time.time()
    
    pulse_duration = pulse_end - pulse_start
    
    distance = pulse_duration * 171.5
    
    distance = round(distance, 4)
    
    return (distance)

def ultrasonic_cleanup():
    
    GPIO.cleanup()
    
    return (None)