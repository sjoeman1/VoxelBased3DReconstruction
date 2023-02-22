import cv2 as cv
import numpy as np


def compose_background(frames):
    average = np.zeros(frames[0].shape, dtype=np.float32)
    sd = np.zeros(frames[0].shape, dtype=np.float32)
    for frame in frames:
        average += frame
        sd += (frame - average)**2
    average /= len(frames)
    sd = np.sqrt(sd / len(frames))
    return np.uint8(average), np.uint8(sd)

def getFrames(cam):
    frames = []
    totalFrames = cam.get(cv.CAP_PROP_FRAME_COUNT)
    _, img = cam.read()
    for i in np.linspace(0, totalFrames, num= 30, dtype= int, endpoint=False):
        cam.set(cv.CAP_PROP_POS_FRAMES, i)
        _, img = cam.read()
        frames.append(img)
    return np.array(frames)

def main():
    for i in range(4):
        video = cv.VideoCapture('data/cam' + str(i+1) + '/background.avi')
        frames = getFrames(video)
        background, sd = compose_background(frames)
        cv.imwrite('data/cam' +  str(i+1) + '/background.png', background)
        cv.imwrite('data/cam' +  str(i+1) + '/sd.png', sd)

if __name__ == '__main__':
        main()

