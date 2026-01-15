import pyttsx3
import time

engine = pyttsx3.init()

engine.setProperty('rate', 160) 
engine.setProperty('volume', 1.0) 

last_spoken = 0
last_message = ""
COOLDOWN = 2.5 
REPEAT_COOLDOWN = 5 

def speak(message):
    global last_spoken, last_message
    current_time = time.time()
    
    # Determine which cooldown to use
    if message == last_message:
        required_cooldown = REPEAT_COOLDOWN
    else:
        required_cooldown = COOLDOWN

    if current_time - last_spoken > required_cooldown:
        engine.say(message)
        engine.runAndWait()
        last_spoken = current_time
        last_message = message
        return True
    
    return False

def speak_immediate(message):
    global last_spoken, last_message
    engine.say(message)
    engine.runAndWait()
    last_spoken = time.time()
    last_message = message