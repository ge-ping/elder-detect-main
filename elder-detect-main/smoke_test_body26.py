"""RTMPose body-26 环境冒烟测试。

用法（在 stgcn 环境下）：
    python smoke_test_body26.py                # 自动下载一张示例图跑通
    python smoke_test_body26.py 你的图片.jpg   # 用自己的图
预期：打印 keypoints shape (26, 2) 或 (26, 3)，无报错即环境 OK。
注意：本仓库设计去面部 5 点后 V=21，那是后处理；模型原生输出 26 点，此处先验证 26 点推理链路。
"""
import sys
import urllib.request

from mmpose.apis import MMPoseInferencer

img = sys.argv[1] if len(sys.argv) > 1 else None
if img is None:
    url = "https://github.com/open-mmlab/mmpose/raw/main/tests/data/coco/000000000785.jpg"
    img = "smoke_sample.jpg"
    print("下载示例图 ->", img)
    urllib.request.urlretrieve(url, img)

inferencer = MMPoseInferencer(pose2d="body26")  # alias 自动加载 RTMDet+RTMPose
for item in inferencer(img):
    pred = item["predictions"][0]
    kpts = pred["keypoints"]
    scores = pred["keypoint_scores"]
    print("keypoints shape:", kpts.shape, "| first 3:", kpts[:3])
    print("scores shape:", scores.shape)
    print("OK: body26 推理通过")
