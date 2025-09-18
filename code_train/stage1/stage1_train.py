import os
from ultralytics import YOLO
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

warnings.filterwarnings("ignore")



def train_stage1():
    """第一阶段训练：仅定位面单位置"""
    model = YOLO("ultralytics/cfg/models/11/yolo11n-CBAM.yaml")

    model.train(
        data=r"F:\envs_workstation\yolo11\code_train\stage1\stage1_dataset.yaml",
        augment=True,
        imgsz=800,
        epochs=100,
        batch=16,
        lr0=0.001,
        workers=8,
        optimizer='AdamW',
        single_cls=True,  # 单类别：仅定位
        device='cuda',
        amp=True,
        name='stage1_localization',
        # nc=1,  # 覆盖为1个类别
        # patience=20
    )


if __name__ == '__main__':
    train_stage1()