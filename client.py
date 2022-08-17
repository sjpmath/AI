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
    print(args)

if __name__ == '__main__':
    main()
