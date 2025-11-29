from ultralytics import YOLO

# 加载模型
model = YOLO("yolo11n.pt")  # 预训练的YOLO11n模型

# 对图像列表进行批量推理
results = model(["image1.jpg", "image2.jpg"])  # 返回Results对象列表

# 处理结果列表
for result in results:
    boxes = result.boxes  # 边界框输出的Boxes对象
    masks = result.masks  # 分割掩码输出的Masks对象
    keypoints = result.keypoints  # 姿态输出的Keypoints对象
    probs = result.probs  # 分类输出的Probs对象
    obb = result.obb  # OBB输出的Oriented boxes对象
    result.show()  # 显示到屏幕
    result.save(filename="result.jpg")  # 保存到磁盘