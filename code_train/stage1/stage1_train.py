import sys
import os

# 设置 PYTHONPATH，保证优先使用本地修改后的 ultralytics
ultralytics_path = r"F:\envs_workstation\yolo11\ultralytics"
if ultralytics_path not in sys.path:
    sys.path.insert(0, ultralytics_path)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore")

# 然后再导入 YOLO


from ultralytics import YOLO


def train_stage1():
    """第一阶段训练：仅定位面单位置"""
    model = YOLO(r"F:\envs_workstation\yolo11\code_train\stage1\verify\base\CBAM_CA\weights\last.pt")    # pt模型文件/自定义的模型结构文件.yaml
    model.train(
        #resume=True,                        # 接着.last模型继续训练需要设置的参数
        data=r"F:\envs_workstation\yolo11\code_train\stage1\stage1_datasets.yaml",          # 数据文件
        scale='n' ,                 #指定版本号
        augment=True,
        imgsz=800,
        epochs=50,
        batch=8,   # 原16
        lr0=0.001,
        workers=8,  # 原8
        optimizer='AdamW',
        single_cls=True,  # 单类别：仅定位
        device='cuda',
        amp=True,
        # name='stage1_localization',
        # nc=1,  # 覆盖为1个类别
        project=r"F:\envs_workstation\yolo11\code_train\stage1\verify\base",    # 训练输出文件的根目录
        name="little_target",                                                         # 拼接根目录后的最后输出文件夹
        exist_ok=False

        # 早停设置
        # patience=10  验证集指标连续10个epoch没有改善就停止
        # save_period=5  每5个epoch保存，便于回滚
        # plots=True 生成训练曲线图，便于观察过拟合

    )


if __name__ == '__main__':
    train_stage1()