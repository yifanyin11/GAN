import os
import cv2
import numpy as np


def draw_annotation(box_img, bbox=[], kp=[], win_name="Img with Annotation"):
    color_kp = [0, 0, 255]
    color_bbox = [0, 127, 255]
    box_img_s = box_img.copy()
    for d in kp:
        d = [int(pt) for pt in d]
        cv2.circle(box_img_s, tuple(d), radius=2, color=color_kp, thickness=-1)
    for i in range(int(len(bbox)/4)):
        cv2.rectangle(box_img_s, tuple(bbox[0:2]), tuple(bbox[2:]), color_bbox, thickness=2)
    cv2.imshow(win_name, box_img_s)

def refine_keypoints(img, kp, win=5):
    # img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    kp = cv2.cornerSubPix(gray, np.array(kp).astype(np.float32), (win,win), (-1,-1), criteria)
    return kp

def resize_bg_img(path, size):
    h = size[0]
    w = size[1]
    dirlist = os.listdir(path)
    print(dirlist)
    cwd = os.getcwd()
  
    for imgfile in dirlist:
        os.chdir(path)
        img = cv2.imread(imgfile)
        height, width = img.shape[:2]
        # crop images    
        if size[1]*height/(size[0]*width)<0.7 or size[1]*height/(size[0]*width)>1.3:
            if height>h and width>w:
                img = img[int((height-h)/2):int((height+h)/2), int((width-w)/2):int((width+w)/2)]
            else:
                if height/width > h/w:
                    # fix width
                    height_new = h/w*width
                    img = img[int((height-height_new)/2):int((height+height_new)/2), :]
                else:
                    width_new = w/h*height
                    img = img[:, int((width-width_new)/2):int((width+width_new)/2)]
        img = cv2.resize(img, (w, h), cv2.INTER_AREA)
        cv2.imwrite(os.path.join(path, imgfile), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),)
    os.chdir(cwd)

def delete_files(dir, common_str):
    fileList = os.listdir(dir)
    for d in fileList:
        if common_str in d:
            os.remove(os.path.join(dir, d))
            
