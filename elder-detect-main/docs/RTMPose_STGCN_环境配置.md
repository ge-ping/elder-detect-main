# RTMPose + ST-GCN 环境配置（阶段 0 落地指令）

> 配套任务清单：`docs/RTMPose_STGCN_开发任务清单.md`（阶段 0）
> 分支：`feature/rtmpose-stgcn-fall-risk`
> 作用：本文件给出**可直接复制执行**的安装命令；由你在本地环境运行，**不在此仓库内执行**。
> 关键约定：本构建用 **RTMPose body-26（去面部 5 点 → V=21）**；跟踪用 **MMPose PoseTracker = RTMDet 检测框 + RTMPose 关键点 + ByteTrack**，故**必须同时装 mmdet 与 RTMPose**。

> 🔴 **【2026-07-15 实测·路线裁决】Windows 原生装 mmcv 不可行 → 推荐改 WSL2 Ubuntu！**
> 逐项 curl 探明：
> 1. **mmcv 在 Windows 无任何预编译 wheel**——PyPI 的 2.1.0/2.2.0 仅 `tar.gz` 源码包；OpenMMLab 从不发 Windows wheel。
> 2. **OpenMMLab 仅给 Linux(manylinux) 编译 mmcv**，最高只到 **torch2.4**；你原计划的 `cu121/torch2.5` 目录 **404 不存在**。
> 3. Windows 源码编译 mmcv 需 VS2022+CUDA 工具链，且卡在 `ModuleNotFoundError: No module named 'pkg_resources'`（mmcv 2.x 老式 setup 坑）。
> → 纯 Windows 只剩"源码编译 mmcv"，极痛苦。
> ✅ **推荐 WSL2 Ubuntu（Linux 子系统）**：openmmlab `cu121/torch2.1.0` 有 **mmcv-2.1.0 的 Linux manylinux wheel**，pip 直装；且匹配 mmpose 1.3.2 锁定的 `mmdet<3.3 → mmcv<2.2`。
> **唯一 pip 直装预编译组合**：`torch==2.1.0+cu121` + `mmcv==2.1.0` + `mmdet==3.2.0` + `mmpose==1.3.2`（torch 由 2.5.1 降到 2.1.0，ST-GCN 所需算子全支持）。完整命令见 **§1.5**。
> Windows 原生 torch 2.5.1 那条命令（步骤 2 备选）**已无后续 mmcv 可配**，仅留作记录，勿用于本构建。

> ⚠️ **【强制】OpenMMLab 栈必须装在独立环境（如 `stgcn`），禁止装进现有 `elder`！**
> `elder` 是**现有 MediaPipe+LSTM 跌倒系统**的运行环境（含 `mediapipe`），且已混装 `d2l / datasets / selenium / transformers`。混装 OpenMMLab 会：
> 1. **搞坏现有系统**：`mim install mmpose` 的依赖解析可能把 **numpy 顶到 ≥2.0**，而 MediaPipe 的 wheel **锁 `numpy<2.0`** → 现有 `demo.py` / MediaPipe 推理当场报废（验证：`pip show mediapipe` 看 Requires 行的 numpy 上限）。
> 2. **环境更乱**：OpenMMLab 对 torch/mmcv 版本极敏感，与 `elder` 里已有包互相拉扯（你已见到那堆 conflict 警告）。
> **正确做法**：见步骤 1 新建 `stgcn`，OpenMMLab 全装那里；`elder` 保持原样不动。若已误在 `elder` 装了 `openmim` 之类，**别继续装 mmcv/mmdet/mmpose**，直接切到 `stgcn` 重来即可（`openmim` 只是下载器，无害）。

---

## 0. 前置要求

- NVIDIA 驱动满足 CUDA 最低要求（Ampere 30 系 / A100 用 CUDA 11；不确定就 CUDA 11.8）。
- 包管理：推荐 **conda**（或 venv）。Python 建议 **3.10 / 3.11**（mmcv 预编译包覆盖最全；3.12 可能可用但需确认 find-url）。
- 版本对应关系（OpenMMLab）：`mmdet 3.x ↔ mmpose 1.x ↔ mmcv 2.x`。装错主版本会直接 import 失败。

---

## 1. 建虚拟环境（独立，隔离现有 `elder`）

> ⚠️ 若你当前命令行前缀是 `(elder)`，**先 `conda deactivate` 退出来**，绝不在 `elder` 内执行下面的安装。新建的 `stgcn` 是给 RTMPose+ST-GCN 专用的，和 `elder`（MediaPipe+LSTM）互不干扰。

```bash
conda create -n stgcn python=3.11 -y
conda activate stgcn
# 或 venv:
# python3.11 -m venv .venv && source .venv/bin/activate
```

---

## 1.5 推荐：WSL2 Ubuntu 路线（完整命令）

> 为什么 torch 降到 **2.1.0**：openmmlab 仅 `cu121/torch2.1` 提供 `mmcv-2.1.0` 的 Linux wheel（mmcv<2.2，恰好满足 mmpose 1.3.2 锁定的 `mmdet<3.3→mmcv<2.2`）；torch2.3/2.4 目录只有 mmcv 2.2.0（被 mmdet 拒），torch2.5 目录 404。这是唯一能 pip 直装预编译 wheel 的组合。

**① Windows 侧：把 Ubuntu 装到 D 盘（不占 C 盘）**
> C 盘空间紧张时，用官方 rootfs 直装到 D:，避免 WSL 默认把数据盘 vhdx 写进 `C:\Users\...\AppData\Local\Packages`。本机实测 C 仅剩 ~43G、D 剩 ~100G，故全部落 D:。
```cmd
# 1) 管理员 PowerShell：下载 Ubuntu 22.04 WSL rootfs 到 D:（约几百 MB，不碰 C:）
#    官方源 cloud-images.ubuntu.com 在国境极慢 → 用清华镜像（已验证 200）
Invoke-WebRequest -Uri "https://mirrors.tuna.tsinghua.edu.cn/ubuntu-cloud-images/wsl/jammy/current/ubuntu-jammy-wsl-amd64-ubuntu22.04lts.rootfs.tar.gz" -OutFile "D:\ubuntu-jammy.rootfs.tar.gz"

# 2) 导入为名为 Ubuntu 的 WSL2 发行版，数据盘落在 D:\wsl\ubuntu（vhdx 在 D:）
wsl --import Ubuntu D:\wsl\ubuntu D:\ubuntu-jammy.rootfs.tar.gz --version 2

# 3) 进 Ubuntu（import 后默认以 root 登录，够用；要普通用户见末尾附注）
wsl -d Ubuntu

# 4) 导入成功后删掉临时 tarball（可选）
# del D:\ubuntu-jammy.rootfs.tar.gz
```
> 附注 A（可选·建普通用户）：`wsl -d Ubuntu` 进 root 后执行
> `useradd -m -s /bin/bash yourname && echo "yourname ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/yourname`
> 再在 Windows 侧写 `D:\wsl\ubuntu\etc\wsl.conf`（或 Ubuntu 内 `/etc/wsl.conf`）：`[user]\ndefault=yourname`，重进即默认该用户。
> 附注 B（回收 C 盘）：旧的 Windows 版 `stgcn` 环境（`C:\Users\R9000P\.conda\envs\stgcn`，残留 torch 2.5.1+cpu）已废弃，确认 WSL2 路线跑通后可 `conda env remove -n stgcn` 释放 C 盘空间。

**② Ubuntu 侧：换清华源 + 装 python3.10-venv 包**（venv 创建见 ③）
> ⚠️ **关键坑**：别硬编码源文件路径！本 WSL 镜像的真实源文件是 **`/etc/apt/sources.list`**（老式格式，已被 `apt update` 验证走 `archive.ubuntu.com`），**不是** `ubuntu.sources`（该路径在此镜像不存在，sed 会报 "No such file"）。用 `grep -rl` 定位真实文件再替换，避免路径猜错。
> ⚠️ 用 **`http://`** 而非 `https://`：最小 rootfs 可能缺 `ca-certificates`/`apt-transport-https`，https 源会报证书错或 "Method https has died"。清华同时支持 http。
> 💡 **用系统默认 python3.10**（Ubuntu 22.04 自带）。注意：`apt install python3-venv` 在 universe 里会顺带装 python3.11 并只给 `python3.11-venv`，但 `python3`=3.10，故 venv 要**显式装 `python3.10-venv`**。torch 2.1.0+cu121 有 cp310 轮子，3.10 够用。
> 用 `sed` 一行替换（**避免 heredoc 粘贴被吞/EOF 粘连** 这种坑）。
```bash
# 1) 若之前把 sources.list 写脏了，先还原成干净原版（.bak 是改之前备份）
sudo cp /etc/apt/sources.list.bak /etc/apt/sources.list
# 2) grep 定位所有含官方源的 apt 文件（不硬编码路径），整体换成清华（http/https 都覆盖）
sudo sed -i 's#http://archive.ubuntu.com/ubuntu/#http://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g; s#https://archive.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g; s#http://security.ubuntu.com/ubuntu/#http://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g; s#https://security.ubuntu.com/ubuntu/#https://mirrors.tuna.tsinghua.edu.cn/ubuntu/#g' $(grep -rl 'archive.ubuntu.com' /etc/apt/)
# 3) 验证（应无输出 = 已全部替换）
grep -rh 'ubuntu.com' /etc/apt/ || echo "已全部替换为清华 ✅"
# 2) 更新索引（应秒回清华）+ 装 python3.10 工具链（要一次 sudo 密码）
sudo apt update
sudo apt install -y python3.10-venv python3.10-dev python3-pip git
```

> 🔴 **【2026-07-16 实测踩坑·WSL 内 shell 是 `sh`(dash)，venv 用绝对路径调用】**
> 本 WSL 镜像默认登录 shell 是 **`sh`（dash），不是 bash**，且系统 **只有 `python3`、没有 `python`**。已实测会连环爆：
> 1. `source ~/stgcn/bin/activate` → `-sh: source: not found`（sh 无 `source`）。→ **venv 没激活**。
> 2. venv 没激活 → `pip install` 全落进**用户全局目录** `~/.local/lib/python3.10/site-packages/`（提示 "Defaulting to user installation"）——**这不隔离**，会污染该 WSL 内所有 python3 项目，不符合环境隔离要求。
> 3. 末尾 `python -c ...` → `python: not found`（系统无 `python`）。
> ✅ **正确做法：建 venv 但不用 `source` 激活，一律用 venv 内的绝对路径 `~/stgcn/bin/pip`、`~/stgcn/bin/python`、`~/stgcn/bin/mim`。** 这样 dash 无 `source` 也不影响，且完全隔离。
> 💡 **venv 内 torch 重装会命中 pip 缓存**（`~/.cache/pip` 里已有刚下的 2.2GB wheel），**秒装不重下**。装前可先把误装进用户目录的清掉：`python3 -m pip uninstall -y torch torchvision torchaudio triton`（可选，回收空间）。
> 💡 venv 里**有** `python`（venv 创建时会建软链），故 `~/stgcn/bin/python` 可用；venv 装了 mim 后 `~/stgcn/bin/mim` 也在，无需改 PATH。

**③ Ubuntu 侧：建 venv（不激活，用绝对路径）**
```bash
# 若之前建过残缺 venv，先删重建
rm -rf ~/stgcn
python3 -m venv ~/stgcn
# 升级 pip + pip 走清华（写进 venv）
~/stgcn/bin/pip install --upgrade pip
~/stgcn/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**④ 步骤 2：torch 2.1.0+cu121（阿里云直链，命中缓存秒装；清华 pytorch-wheels 无 2.1.0）**
> ⚠️ 阿里云是**扁平目录、非标准索引页**，故只能给**完整文件 URL** 直装，不能当 `--index-url`。URL 里的 `%2B` 必须保留（即 `+` 的转义），且整条用双引号包住。
```bash
~/stgcn/bin/pip install "https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl" "https://mirrors.aliyun.com/pytorch-wheels/cu121/torchvision-0.16.0%2Bcu121-cp310-cp310-linux_x86_64.whl" "https://mirrors.aliyun.com/pytorch-wheels/cu121/torchaudio-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl" --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
~/stgcn/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
> 期望 `2.1.0+cu121 True`（WSL2 用宿主 GPU，需 Win11+最新 NVIDIA 驱动；若 False 先查驱动，或先用 CPU 跑通验证）。第 2 次装若命中缓存会**秒装**，若又开始下 2.2GB 说明缓存没命中。

**⑤ 步骤 3：mmcv 2.1.0（关键！openmmlab `cu121/torch2.1.0` 有 Linux manylinux wheel）**
> ⚠️ 路径是 **`torch2.1.0`**（多一个 `.0`），不是 `torch2.1`——后者是 404 死链（这是 Windows 那次 mmcv 也装不上的根因）。cp310 对应 `mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl`。
```bash
~/stgcn/bin/pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
~/stgcn/bin/pip install -U openmim
~/stgcn/bin/mim install mmengine==0.10.4
~/stgcn/bin/mim install "mmdet==3.2.0"
~/stgcn/bin/mim install "mmpose==1.3.2"
```
> 🔴 **【2026-07-16 实测·chumpy 构建失败】** `mim install mmpose==1.3.2` 会因传递依赖 `chumpy` 在 **PEP 517 构建隔离环境**里 `import pip` 失败（`ModuleNotFoundError: No module named 'pip'`）。新版 pip 的 build isolation 只带 setuptools/wheel、不带 pip，而 chumpy 0.70 的 `setup.py` 直接 `import pip` → 崩。
> ✅ **解法 A（推荐·先单独装 chumpy）**：关掉构建隔离复用 venv 里的 pip，再装 mmpose 就不会重新构建它：
> ```bash
> ~/stgcn/bin/pip install -U pip setuptools wheel numpy
> ~/stgcn/bin/pip install chumpy --no-build-isolation
> ~/stgcn/bin/mim install "mmpose==1.3.2"
> ```
> ✅ **解法 B（一行·对所有包关隔离）**：`PIP_NO_BUILD_ISOLATION=1 ~/stgcn/bin/mim install "mmpose==1.3.2"`（复用 venv 的 pip/numpy）。

**⑥ 步骤 4/5：下载权重 + 冒烟测试**
```bash
cd /mnt/d/myproject/elder_decete     # 进 Windows 项目（WSL2 可访问；或 git clone 到 ~/ 更快）
~/stgcn/bin/mim download mmpose --config rtmpose-m_8xb512-700e_body8-halpe26-256x192 --dest model/rtmpose
~/stgcn/bin/mim download mmdet --config rtmdet_m_300e_coco --dest model/rtmdet
~/stgcn/bin/python smoke_test_body26.py
```
> 期望打印 `keypoints shape: (26, 2)` + `OK: body26 推理通过`。

---

## 2. 安装 PyTorch（按本机 CUDA 选一条）

> ⚠️ **版本号必须带 `+cu121` 本地标签**！若只写 `torch==2.5.1`，pip 从主源抓到 **CPU 版**（`2.5.1+cpu`、`cuda.is_available()=False`）。
>
> **源选择（实测 curl 探明）**：
> - TUNA / USTC 的 `pytorch-wheels/cu121` **没同步 2.5.1**（查无）。
> - 阿里云 / 华为云 的 `pytorch-wheels/cu121` **有**（文件直链 200），但其 `/torch/` **索引页是 404**（扁平目录、非标准 simple），故**只能完整文件 URL 直装，不能作 `--index-url`**。
> - 官方 `download.pytorch.org/whl/cu121` 也有，但国内直连慢（2.4 GB 约 1 小时级）。
>
> **推荐：阿里云直链满速**（cmd 单行；URL 必须带双引号，否则 `%2B` 里的 `%` 会被 cmd 误处理）：

```bash
# 1) 若已装 CPU 版，先卸干净
pip uninstall -y torch torchvision torchaudio

# 2) 阿里云直链安装（国内满速；TUNA pypi 仅用于拉 numpy 等依赖）
pip install "https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl" "https://mirrors.aliyun.com/pytorch-wheels/cu121/torchvision-0.20.1%2Bcu121-cp311-cp311-win_amd64.whl" "https://mirrors.aliyun.com/pytorch-wheels/cu121/torchaudio-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl" --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 华为云同款（域名替换即可）：
# pip install "https://mirrors.huaweicloud.com/pytorch-wheels/cu121/torch-2.5.1%2Bcu121-cp311-cp311-win_amd64.whl" ...同上加 --extra-index-url TUNA

# 备选：官方源（稳定但慢；开代理可加 --proxy http://127.0.0.1:7890 加速）
# pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

> ⚠️ **若 env 已有可用 torch（如已装 `2.5.1+cu121`），不要重装**，直接跳到步骤 3。**mmcv 的 find-url 必须匹配你实际的 torch 主版本**（见步骤 3），与本章示例版本号无关。
> 验证：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` 应打印版本且 `cuda.is_available()=True`。

---

## 3. 安装 OpenMMLab 栈（mmengine → mmcv → mmdet → mmpose）

### 方式 A：固定版本（推荐，可复现）

```bash
pip install -U openmim

# mmcv 必须匹配 torch/cuda，用 find-url 指定（torch 主版本务必一致！）
#   例如本机 torch 2.5.1+cu121 -> cu121/torch2.5
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.5/index.html
#   CUDA 11.8 -> cu118/torch2.5；CPU -> cpu/torch2.5
#   先确认你的 torch：python -c "import torch;print(torch.__version__)"

mim install mmengine==0.10.4
mim install "mmdet==3.2.0"          # 含 RTMDet（跟踪检测器）
mim install "mmpose==1.3.2"         # 含 RTMPose + PoseTracker
```

### 方式 B：一键（让 mim 自动解依赖，最省心但版本浮动）

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

> 从源码装（开发常用，可改 mmpose 源码）：
> ```bash
> git clone https://github.com/open-mmlab/mmpose.git -b 1.x
> cd mmpose && pip install -r requirements.txt && pip install -v -e . && cd ..
> ```

---

## 4. 安装 MMDeploy（部署用，阶段 9 才需要，可先跳过）

```bash
mim install mmdeploy==1.3.0
# 后端按需：
pip install onnxruntime onnxruntime-gpu   # ONNXRuntime
# TensorRT 见 MMDeploy 官方安装（需匹配本机 TensorRT/CUDA）
```

---

## 5. 下载模型权重（落到 `model/`，该目录被 .gitignore 忽略，不入库）

```bash
# RTMPose body-26（官方 alias: body26）
mim download mmpose --config rtmpose-m_8xb512-700e_body8-halpe26-256x192 --dest model/rtmpose
#   ↑ 下载得到 .py 配置 + .pth 权重；alias 也可写作 body26

# RTMDet（PoseTracker 的检测器，ByteTrack 关联必需）
mim download mmdet --config rtmdet_m_300e_coco --dest model/rtmdet
#   ↑ 轻量可选 rtmdet_tiny_300e_coco
```

下载后目录应类似：
```
model/rtmpose/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py
model/rtmpose/rtmpose-m_8xb512-700e_body8-halpe26-256x192-*.pth
model/rtmdet/rtmdet_m_300e_coco.py
model/rtmdet/rtmdet_m_300e_coco-*.pth
```

---

## 6. 冒烟测试（单图跑通 body-26 推理，输出 26 关键点）

```bash
python3 - <<'PY'
from mmpose.apis import MMPoseInferencer
inferencer = MMPoseInferencer(pose2d='body26')   # alias 自动加载 RTMDet+RTMPose
result = inferencer('tests/data/coco/000000000785.jpg')  # 换你自己的图
for item in result:
    kpts = item['predictions'][0]['keypoints']     # ndarray (26, 2) 或 (26,3)
    scores = item['predictions'][0]['keypoint_scores']
    print('keypoints shape:', kpts.shape, '| first 3:', kpts[:3])
    print('scores shape:', scores.shape)
PY
```

预期：输出 `(26, 2)` 坐标 + 26 个置信度，无报错即环境 OK。

> 注：`pose2d='body26'` 会自动调用 RTMDet 检测器，正好顺带验证了阶段 1 跟踪链路的检测前置依赖已就绪。

---

## 7. 固化依赖（可复现）

```bash
pip freeze > requirements-stgcn.txt
# 或只导出 OpenMMLab 相关：
pip list | grep -iE "torch|mmcv|mmdet|mmpose|mmengine|mmdeploy|onnxruntime" >> requirements-stgcn.txt
```
把 `requirements-stgcn.txt` 提交进仓库（它不在 `.gitignore` 忽略范围），`model/` 权重不提交。

---

## 8. 常见坑

- **numpy 2.x 冲突**：部分 OpenMMLab 版本与 numpy≥2 不兼容。报错就 `pip install "numpy<2"`。
- **`ModuleNotFoundError: mmcv._ext`**：mmcv 与 torch/cuda 不匹配。删掉重装并带正确 find-url（见步骤 3）。
- **`mmcv-full` 是旧版**：mmcv 1.x 才叫 full；本构建用 mmcv 2.x，先 `pip uninstall mmcv-full` 再装 2.x。
- **mim 安装依赖冲突（pandas/click）**：按报错提示降级冲突包，或用源码装 mim（`pip install -e .`）。
- **CUDA 不匹配**：torch 的 CUDA 版本须 ≥ 驱动支持版本；`torch.cuda.is_available()` 为 False 时先解决 torch。
- **共享 env 的依赖冲突警告可忽略**：在已装很多包的环境里，`pip install openmim/mim` 末尾可能报 `d2l / datasets / selenium / transformers` 等「requires X, but you have Y」——这是**其他无关包**与 openmim 的冲突，**只是提示、安装已成功**。千万不要为迎合它们去 `pip install numpy==1.21.5` 之类的降级，否则会破坏 torch 2.5 / mmcv。当前 numpy 1.26.x 对 torch 2.5 是兼容的，保持不动。

---

## 9. 目录与 gitignore

- 权重落在 `model/rtmpose/`、`model/rtmdet/`，`model/` 整体被 `.gitignore` 忽略，**不入库**（符合仓库约定）。
- `requirements-stgcn.txt` 应入库；`frames/`、`env_labels/`、`.env`、`.claude/`、`.workbuddy/` 均不提交。
- 环境装好后回到任务清单逐项勾选阶段 0 的 0.1~0.6。
