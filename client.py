#grpc-related
import grpc
import proto_sample_pb2, proto_sample_pb2_grpc

import os
import cv2
import argparse

#options 10651
def opt():
    #object/instance
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='localhost:', help='ip address (default: localhost)')
    parser.add_argument('--port', type=str, default='10651', help='port address (default: 10651)')
    return parser.parse_args()

def main():
    args = opt()

    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        _, frame = webcam.read() # read image: boolean, pixel save frame
        print(type(frame))
        cv2.imshow('window', frame) #show image on pc
        key = cv2.waitKey(33) # 33ms show image - speed - frame per sec
        # change user key input to ASCII
        if key == ord('q'): # change q to ASCII
            break

if __name__ == '__main__':
    main()
