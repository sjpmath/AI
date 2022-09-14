#grpc-related
import grpc
import proto_sample_pb2, proto_sample_pb2_grpc
from calibration import Calibration

import os
import cv2
import argparse


class Message:

    """

        Types of user feedback
        key = message name
        loc = location of message on frame
        color = color of message
        text = text shown on frame

    """

    m_dict = {
        'absent': {
            'loc': (100,300),
            'color': (255,0,0),
            'text': 'Pause'
        },
        'error': {
            'loc': (100,300),
            'color': (0,255,0),
            'text':'Cannot connect to server'
        },
        'too_close': {
            'loc': (100,300),
            'color': (0,255,0),
            'text':'Your face is too close to the screen'
        }
    }

def userfeedback(distance, frame, thresdistance):
    """

        Arguments:
            distance (int) = distance between face and screen
            frame (np.ndarray) = webcam image
            thresdistance (int) = threshold distance calibrated by user

        Returns feedback for user based on distance between face and screen

    """
    keyname = ''
    if distance ==0: #error
        keyname = 'absent'
        # feature for pausing video
    elif distance==-1: #server error
        keyname = 'error'
    elif distance < thresdistance: #too close
        keyname = 'too_close'
        os.system('say face is too close') # text to speech
    frame = cv2.putText(frame, 'Detected distance: {}'.format(distance), \
        (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2) # put text in image
    if keyname!='':
        frame = cv2.putText(frame, Message.m_dict[keyname]['text'], \
            Message.m_dict[keyname]['loc'], cv2.FONT_HERSHEY_SIMPLEX, 1.0, \
            Message.m_dict[keyname]['color'], 2)
    return frame

def grpc_request(ip, port, frame):
    if ip[-1]!=':':
        ip+=':'
    print('request: client -> server : {}{}'.format(ip,port))
    with grpc.insecure_channel(ip+port) as channel: #store info as channel
        stub = proto_sample_pb2_grpc.AI_OnlineClassMonitorStub(channel)
        response = stub.process(
            proto_sample_pb2.InferenceRequest(
                img_bytes = bytes(frame),
                width = frame.shape[1],
                height = frame.shape[0],
                channel = frame.shape[2]
            )
        )
        return response

#options 10651
def opt():
    #object/instance
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, default='localhost:', \
        help='ip address (default: localhost)')
    parser.add_argument('--port', type=str, default='10651', \
        help='port address (default: 10651)')
    return parser.parse_args()

def main():
    args = opt()

    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cal = Calibration(webcam, grpc_request)
    calibratedDistance = cal(args.ip, args.port)


    while True:
        _, frame = webcam.read() # read image: boolean, pixel save frame
        response = grpc_request(args.ip, args.port, frame)
        frame = userfeedback(response.distance, frame, calibratedDistance)
        frame = cv2.putText(frame, 'Calibrated distance: {}'.format(calibratedDistance), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2) # put text in imag
        cv2.imshow('window', frame) #show image on pc
        key = cv2.waitKey(330) # 33ms show image - speed - frame per sec
        # change user key input to ASCII
        if key == ord('q'): # change q to ASCII
            break
        elif key==ord('s'):
            cv2.imwrite('shot.png', frame)

if __name__ == '__main__':
    main()
