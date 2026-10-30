TWO-CLASS YOLO11 ROAD DAMAGE EXPERIMENT

Copy these two Python files into your project root:
C:\MyWork\RoadDamageDetection

Expected source dataset:
road_damage_data\images\train, val, test
road_damage_data\labels\train, val, test

Run in this order from the project folder:
1) python prepare_2class_dataset.py
2) python train_2class_640.py

The preparation script creates a NEW dataset folder named road_damage_2class and
writes data_2class.yaml. It does not modify the original road_damage_data.

Class mapping:
Original 0 pothole -> new 0 pothole
Original 1 crack -> removed
Original 2 manhole -> new 1 manhole

Images that contain only cracks are skipped. Images containing potholes/manholes
are kept, with crack annotations removed. The existing train/validation/test split
is preserved. Do not resume the old 3-class checkpoint for this experiment.
