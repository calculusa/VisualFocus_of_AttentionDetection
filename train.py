from ultralytics import YOLO


def main():
    model = YOLO("yolo26s.pt")

    model.train(
        data="/home/lunet/cowz2/Documents/VisualFocus_of_AttentionDetection/Dataset/widerface_yolo26_subset/data.yaml",
        epochs=100,
        imgsz=640,
        device=[0, 1],
        batch=16,
        workers=8,
        project="runs/vfoa_yolo26_subset_3c_14092026",
        name="yolo26s_img640_ep100_subset_3c_14092026",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()



# from ultralytics import YOLO


# def main():
#     last_pt = (
#         "/home/lunet/cowz2/Documents/"
#         "VisualFocus_of_AttentionDetection/yolo26/"
#         "runs/detect/runs/vfoa_yolo26/"
#         "yolo26s_img640_ep100/weights/last.pt"
#     )

#     model = YOLO(last_pt)

#     model.train(
#         resume=True,
#         device="2,3",
#     )


# if __name__ == "__main__":
#     main()