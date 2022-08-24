import cv2

from .uiux import CalibrationUI

class Calibration:
    @staticmethod
    def process(webcam, tracker, debug=False):
        cv2.namedWindow("Calibration", cv2.WINDOW_AUTOSIZE)
        
        calibrator = None; calib_dict = None
        is_being_calibrated = False
        
        # TODO : parse from args
        calibration_cnt = 20

        while True:
            _, frame_bgr = webcam.read()

            if calibrator is None:
                H, W, _ = frame_bgr.shape
                calibrator = CalibrationUI(W, H)
                calib_dict = calibrator.calib_dict
        
            # inference 
            eval_result = tracker(frame_bgr)

            if eval_result.eye_pitch is None:
                continue

            if debug:
                vis_img = tracker.get_visualizer_image()        
                vis_img = cv2.flip(vis_img, 1)
            else:
                vis_img = calibrator.template
            
            # calibration 
            if is_being_calibrated: 
                circle_size = calibration_cnt
                vis_img = calibrator.center_circle(vis_img, circle_size)
                calibrator.calib_dict = eval_result

                calibration_cnt -= 1
                if calibration_cnt == 3: # calibration finish
                    break

            else:
                vis_img = calibrator.center_circle(vis_img)
           
            cv2.imshow('Calibration', vis_img)
            key = cv2.waitKey(33)
            if key == ord('q'):
                exit()
            elif key == ord('c') and not is_being_calibrated:
                is_being_calibrated = True
                calibrator.update_template()
                
        cv2.destroyAllWindows()

        calib_dict = calibrator.calib_dict
        for k in calib_dict:
            if k != 'cnt':
                calib_dict[k] /= calib_dict['cnt']
                if debug:
                    print('Calibrated {}: {:.2f}'.format(k, calib_dict[k]))

        return calib_dict