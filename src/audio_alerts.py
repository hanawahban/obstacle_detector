import pyttsx3
import time

engine = pyttsx3.init()
last_spoken = 0
COOLDOWN = 3 

def speak(message):
    global last_spoken
    current_time = time.time()

    if current_time - last_spoken > COOLDOWN:
        engine.say(message)
        engine.runAndWait()
        last_spoken = current_time
