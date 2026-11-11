"""启动纵向个人基线 Web 演示。"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web_demo.app import create_app  # noqa: E402


if __name__ == "__main__":
    print("演示地址：http://127.0.0.1:5000")
    print("未安装 MediaPipe .task 模型时，上传后会明确进入示例回退模式。")
    create_app().run(host="127.0.0.1", port=5000, debug=False)
