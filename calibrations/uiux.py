import math

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import textwrap


class UIUX:
    kor_font = ImageFont.truetype('fonts/AppleGothic.ttf', 20)

    @staticmethod
    def get_font(size=20):
        return ImageFont.truetype('fonts/AppleGothic.ttf', size)

    @staticmethod
    def linedashed(image, pt1, pt2, color, dashlen=4, ratio=3, arrow=True):
        image = Image.fromarray(image.astype(np.uint8))
        draw = ImageDraw.Draw(image)

        x0, y0 = pt1
        x1, y1 = pt2


        dx=x1-x0 # delta x
        dy=y1-y0 # delta y

        # TODO : fix coords bug
        assert dy > 0
        
        # check whether we can avoid sqrt
        if dy==0: len=dx
        elif dx==0: len=dy
        else: len=math.sqrt(dx*dx+dy*dy) # length of line
        xa=dx/len # x add for 1px line length
        ya=dy/len # y add for 1px line length
        step=dashlen*ratio # step to the next dash
        a0=0
        last_pt = []
        while a0<len:
            a1=a0+dashlen
            if a1>len: a1=len
            draw.line((x0+xa*a0, y0+ya*a0, x0+xa*a1, y0+ya*a1), fill =color, width=7)
            last_pt.append( (x0+xa*a1, y0+ya*a1) )
            a0+=step 

        image = np.array(image)

        if arrow:
            # TODO : fix coords bug
            #last_pt = tuple(map(int, last_pt[-2]))
            #image = cv2.arrowedLine(image, last_pt, pt2, color=color, thickness=1)
            last_pt = tuple(map(int, last_pt[1]))
            #! image = cv2.arrowedLine(image, last_pt, pt1, color=color, thickness=3)
            

        return image


    @staticmethod
    def draw_border(img, pt1, pt2, color, thickness, r, d):
        x1,y1 = pt1
        x2,y2 = pt2
        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)        
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
        
        # top horizontal line (optional)

        # bottom horizontal line (optional)

        return img

    @staticmethod
    def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):
        #  corners:
        #  p1 - p2
        #  |     |
        #  p4 - p3

        bottom_right = (bottom_right[1], bottom_right[0])

        p1 = top_left
        p2 = (bottom_right[1], top_left[1])
        p3 = (bottom_right[1], bottom_right[0])
        p4 = (top_left[0], bottom_right[0])

        height = abs(bottom_right[0] - top_left[1])

        if radius > 1:
            radius = 1

        corner_radius = int(radius * (height/2))

        if thickness < 0:
            #big rect
            top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
            bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

            top_left_rect_left = (p1[0], p1[1] + corner_radius)
            bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

            top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
            bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

            all_rects = [
            [top_left_main_rect, bottom_right_main_rect], 
            [top_left_rect_left, bottom_right_rect_left], 
            [top_left_rect_right, bottom_right_rect_right]]

            [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

        # draw straight lines
        cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
        cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
        cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
        cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

        # draw arcs
        cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
        cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
        cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
        cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

        return src

    @staticmethod
    def putText(image, message, origin, size=20, kor=True, center=True):
        if kor:
            _, MAX_W, _ = image.shape

            image = Image.fromarray(image.astype(np.uint8))
            draw = ImageDraw.Draw(image)

            font = UIUX.get_font(size)

            if center:
                w, h = draw.textsize(message, font=font)
                draw.text(((MAX_W- w) / 2, origin[1]-10), message, font=font)
                #draw.text(origin, message, font=font, anchor="mm") #! not working
            else:
                draw.text(origin, message, font=font)

            image = np.array(image)

        else:
            raise NotImplementedError

        return image

    '''
    @staticmethod
    def getCenterLocation(width, height, message, x=None, y=None):
        font = cv2.FONT_HERSHEY_SIMPLEX
        textSize = cv2.getTextSize(message, font, 1, 2)[0]

        textX = (width - textSize[0]) // 2
        textY = (height + textSize[1]) // 2

        if x is not None:
            textX = x
        if y is not None:
            textY = y

        return textX, textY
    '''

    @staticmethod
    def get_calibration_image(width, height, guide=True):
        img = np.zeros((height, width, 3))
        
        if not guide:
            return img 

        # title
        message1 = '시선 추적 설정'
        img = UIUX.putText(img, message1, (width//2, height//5), size=40)
       
        # guide
        message2 = '머리를 고정시킨채로 붉은색 점을 응시해주세요'
        img = UIUX.putText(img, message2, (width//2, height-height//5), size=20)
        
        # yellow round-rect
        # TODO : automatically calcuate  rect size 
        center_pt = (width//2, height-height//5)
        round_rect_height = 80
        round_rect_width = int(width / 2.5)

        pt1 = (center_pt[0] - round_rect_width//2, center_pt[1] - round_rect_height//2)
        pt2 = (center_pt[0] + round_rect_width//2, center_pt[1] + round_rect_height//2)
        img = UIUX.draw_border(img, pt1, pt2, (0,255,255), thickness=6, r=5, d=10)

        # dashed arrowed line
        pt1 = (center_pt[0], center_pt[1]-30)
        pt2 = (width//2, height//2 + 20)
        img = UIUX.linedashed(img, pt2, pt1, (0,187, 255))

        return img


class CalibrationUI:
    def __init__(self, width, height, **kwargs):
        self.width = width
        self.height = height

        # calibration tempplate
        self._guide_template = UIUX.get_calibration_image(self.width, self.height)
        self._calib_template = UIUX.get_calibration_image(self.width, self.height, guide=False)

        self._template = self._guide_template

        self._build_calibration_dict()
        #self._build_hyper_params(**kwargs)

    def _build_calibration_dict(self):
        self._calib_dict = {
            'pitch' : 0,
            'yaw' : 0,
            'x' : 0,
            'y' : 0,
            'cnt' : 0
        }

    def center_circle(self, image, radius=20):
        return cv2.circle(image, (self.width//2, self.height//2), radius, (0,0,255), -1)

    @property
    def calib_dict(self):
        return self._calib_dict

    @property
    def template(self):
        return self._template.copy()
    
    def update_template(self):
        self._template = self._calib_template

    @calib_dict.setter
    def calib_dict(self, eval_result):
        self._calib_dict['pitch'] += eval_result.eye_pitch
        self._calib_dict['yaw'] += eval_result.eye_yaw
        self._calib_dict['cnt'] += 1

    

        
    
        



