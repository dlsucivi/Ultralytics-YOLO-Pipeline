from ultralytics import YOLO

if __name__ == '__main__':
    # SETTINGS
    MODEL = 'yolov8l'
    NAME = 'CamoCrops_yolov8l-final'
    DATA = 'camocrops.yaml'
    EPOCHS = 250
    BATCH = 32
    IMGSZ = 300
    LR0 = 0.01
    LRF = 0.0001
    MOMENTUM = 0.9
    OPTIMIZER = 'SGD'
    SAVE_PERIOD = 5
    VAL = False

    model = YOLO(MODEL + '.pt')

    # Train the model
    train_results = model.train(data=DATA,
                                epochs=EPOCHS,
                                batch=BATCH,
                                imgsz=IMGSZ,
                                save_period=SAVE_PERIOD,
                                device=0,
                                name=NAME,
                                optimizer=OPTIMIZER,
                                lr0=LR0,
                                lrf=LRF,
                                momentum=MOMENTUM,
                                val=VAL)
