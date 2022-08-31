#grpc-related
import grpc
import proto_sample_pb2, proto_sample_pb2_grpc

import os
import cv2
import argparse

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

        response = grpc_request(args.ip, args.port, frame)
        frame = cv2.putText(frame, str(response.distance), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        print(response.distance)

        cv2.imshow('window', frame) #show image on pc
        key = cv2.waitKey(33) # 33ms show image - speed - frame per sec
        # change user key input to ASCII
        if key == ord('q'): # change q to ASCII
            break

if __name__ == '__main__':
    main()
