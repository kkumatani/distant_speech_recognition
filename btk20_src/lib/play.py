# 
#                        Beamforming Toolkit
#                               (btk)
# 
#   Module:  btk.play
#   Purpose: Sound playback.
#   Author:  Fabian Jakobs

import pygame
import time

def initPlayback(sampling, stereo):
    """
    Initialize PyGame.
    """
    pygame.mixer.init(sampling, sampling, stereo)

# def playData(data, sampling = 16000.0, stereo=0):
def playData(data, sampling = 22050.0, stereo=0):
    global gSampleRate, gStereo
    """
    play the data

    The data needs to be a NumPy array.
    Note: You will have to install pygame (SDL wrappers)
    """
    if sampling != gSampleRate or stereo != gStereo:
            deinitPlayback()
            gSampleRate = sampling
            gStereo     = stereo
            initPlayback(sampling = gSampleRate, stereo = gStereo)

    snd     = pygame.sndarray.make_sound(data)
    channel = snd.play()
    while channel.get_busy():
        time.sleep(0.1)

def deinitPlayback():
    pygame.mixer.quit()

global gSampleRate, gStereo
# gSampleRate = 16000.0
gSampleRate = 22050.0
gStereo     = 0
initPlayback(sampling = gSampleRate, stereo = gStereo)
