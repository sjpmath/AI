import grpc
import proto_sample_pb2, proto_sample_pb2_grpc

from concurrent import futures
import os
import cv2
import numpy as np
import argparse

from datetime import datetime
from pytz import timezone
import logging

import ai_module as ai

class OnlineClassMonitor_Model(proto_sample_pb2_grpc.AI_OnlineClassMonitor):
    def __init__(self, args):
        super().__init__()

        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir) # mkdir
        #create logger obj
        self.logger = logging.getLogger(__class__.__name__) # name of class
        self.logger.setLevel(logging.INFO) # classification of log - information/warnung/critical error
        log_file_path = os.path.join(args.logdir, __class__.__name__+'.log') # logdir/classname.log

        self.stream_handler = logging.StreamHandler()
        self.file_handler = logging.FileHandler(log_file_path)
        #add handlers
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)

        self.fmt = '%Y-%m-%d %H:%M:%S' # year month day hour min sec
        time_info = datetime.now(timezone('Asia/Seoul')).strftime(self.fmt)

        self.model = ai.GazeTracker()


        self.logger.info('{} - Server ready'.format(time_info))

    def process(self, input, context):

        image = np.array(list(input.img_bytes))
        image = image.reshape((input.height, input.width, input.channel))
        image = np.array(image, dtype=np.uint8) # create image



        response = proto_sample_pb2.InferenceReply()
        try:
            result = self.model(image)
            print(result.face_distance)
        except Exception as e:
            self.logger.info('{0} - Error Occurred: {1}'.format(
                datetime.now(timezone('Asia/Seoul')).strftime(self.fmt), repr(e)
            ))
            response.distance=-1
        except KeyboardInterrupt:
            print("Terminate server")
        else:
            response.distance = result.face_distance
        finally:
            self.logger.info('{0} - Successful image process'.format(
                datetime.now(timezone('Asia/Seoul')).strftime(self.fmt)
            ))
            return response


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='10651', help='Port number,default=50051')
    parser.add_argument('--num_worker', type=int, default=8, help='the number of workers')
    parser.add_argument('--logdir', type=str, default='./service_log', help='directory for logs')
    return parser.parse_args()

def main():
    args = opt()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.num_worker))
    proto_sample_pb2_grpc.add_AI_OnlineClassMonitorServicer_to_server(
        OnlineClassMonitor_Model(args), server # register class to server
    )
    server.add_insecure_port('[::]:%s' % (args.port))
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    main()
