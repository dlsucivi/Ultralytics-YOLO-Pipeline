from ultralytics import YOLO


def write_results(name,
                  metrics):
    with open('runs/detect/' + name + '/ap50.txt', 'a') as file:
        for ap50 in metrics.box.ap50:
            file.write(str(ap50) + '\n')
        file.write('\n')
        file.write(str(metrics.box.map50))

    with open('runs/detect/' + name + '/maps.txt', 'a') as file:
        for maps in metrics.box.maps:
            file.write(str(maps) + '\n')
        file.write('\n')
        file.write(str(metrics.box.map))


if __name__ == '__main__':
    version = 'CamoCrops_yolov8l-final'

    start_e = 5
    end_e = 250
    interval = 5

    imgsz = 300
    batch = 1
    save_json = True
    conf = 0.01
    iou = 0.5
    max_det = 50

    path = 'runs/detect/' + version + '/weights/'

    for epoch in range(start_e, end_e, interval):
        weight_file = path + 'epoch' + str(epoch) + '.pt'
        model = YOLO(weight_file)
        name = version + '_epoch_' + str(epoch)
        metrics = model.val(imgsz=imgsz,
                            batch=batch,
                            save_json=save_json,
                            conf=conf,
                            iou=iou,
                            max_det=max_det,
                            name=name)

        write_results(name, metrics)

    weight_file = path + 'last.pt'
    model = YOLO(weight_file)
    name = version + '_epoch_' + str(end_e)
    metrics = model.val(imgsz=imgsz,
                        batch=batch,
                        save_json=save_json,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        name=name)

    write_results(name, metrics)
