"""
MediaPipe Pose 模型下载脚本

从 Google 官方 CDN 下载预训练的 PoseLandmarker 模型文件。
模型文件较大（~5-15MB），不会被 git 提交（已在 .gitignore 中排除）。

运行方式:
    python model/mediapipe_pose/download_models.py          # 下载全部3个版本
    python model/mediapipe_pose/download_models.py --lite   # 仅下载 lite
    python model/mediapipe_pose/download_models.py --full   # 仅下载 full
    python model/mediapipe_pose/download_models.py --heavy  # 仅下载 heavy
"""

import os
import sys
import urllib.request
import argparse
from pathlib import Path

# MediaPipe Pose Landmarker 模型 URL（Google 官方）
MODELS = {
    "lite": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "file": "pose_landmarker_lite.task",
        "desc": "轻量版 — CPU实时 ~15ms/帧，适合边缘设备（论文4/8推荐）",
    },
    "full": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "file": "pose_landmarker_full.task",
        "desc": "完整版 — 精度更高，GPU ~8ms/帧，适合服务器端",
    },
    "heavy": {
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        "file": "pose_landmarker_heavy.task",
        "desc": "重量版 — 最高精度，适合离线批处理",
    },
}

# 下载到脚本所在目录
OUTPUT_DIR = Path(__file__).resolve().parent


def download_file(url: str, dest: Path) -> bool:
    """下载单个文件，显示进度"""
    print(f"  下载中: {url}")
    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                bar_len = 30
                filled = int(bar_len * downloaded // total_size)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stdout.write(f"\r  [{bar}] {pct}% ({downloaded//1024}/{total_size//1024} KB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, str(dest), _progress)
        print()  # 换行
        return True
    except Exception as e:
        print(f"\n  下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="下载 MediaPipe Pose Landmarker 模型文件"
    )
    parser.add_argument("--lite", action="store_true", help="仅下载 lite（轻量版）")
    parser.add_argument("--full", action="store_true", help="仅下载 full（完整版）")
    parser.add_argument("--heavy", action="store_true", help="仅下载 heavy（重量版）")
    args = parser.parse_args()

    # 决定下载哪些版本
    versions = ["lite", "full", "heavy"]
    if any([args.lite, args.full, args.heavy]):
        versions = []
        if args.lite:
            versions.append("lite")
        if args.full:
            versions.append("full")
        if args.heavy:
            versions.append("heavy")

    print(f"模型保存目录: {OUTPUT_DIR}")
    print(f"待下载: {', '.join(versions)} ({len(versions)} 个)\n")

    for ver in versions:
        info = MODELS[ver]
        dest = OUTPUT_DIR / info["file"]
        print(f"[{ver.upper()}] {info['desc']}")

        if dest.exists():
            size_kb = dest.stat().st_size // 1024
            print(f"  ✓ 已存在 ({size_kb} KB)，跳过")
            continue

        success = download_file(info["url"], dest)
        if success:
            size_kb = dest.stat().st_size // 1024
            print(f"  ✓ 下载完成 ({size_kb} KB)")
        print()

    print("完成。")
    print(f"\n提示: 模型文件已加入 .gitignore，不会被提交到 git。")
    print(f"如需在 CI/新环境部署，请重新运行此脚本下载。")


if __name__ == "__main__":
    main()
