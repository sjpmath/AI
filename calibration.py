import cv2

class Calibration:
    def __init__(self, webcam, grpc):
        self.webcam = webcam
        self.grpc = grpc
    def __call__(self, ip, port):
        while True:
            _,frame = self.webcam.read()
            response = self.grpc(ip, port, frame)
            frame = cv2.putText(frame, 'Detected distance: {}'.format(response.distance), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.imshow('window', frame) #show image on pc
            key = cv2.waitKey(33) # 33ms show image - speed - frame per sec
            # change user key input to ASCII
            if key == ord('q'): # change q to ASCII
                break
            elif key == ord('c'):
                return response.distance
