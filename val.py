import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
from ultralytics import YOLO
if __name__ == '__main__':
    #model = YOLO('runs/best.pt')
    #model = RTDETR('runs/best.pt')
    #model = RTDETR('E:\\papers\\PWD\\exp-Vis-yolov8-faster150\\weights\\last.pt')
    model = YOLO("E:\\projects\\pytorch\\UCBCL\\ultralytics\\runs\\train52\\weights\\best.pt")
    #model = YOLO("E:\\papers\\PWD\\exp-Vis-yolov8-faster141\\weights\\best891.pt")
    model.val(data="cfg/datasets/UCB_new.yaml",
              #data = 'cfg/datasets/tree-small.yaml',
              split='test',
              #task='test',
              imgsz=640,
              batch=8,
              iou=0.7,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )