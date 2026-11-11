@echo off
REM ST-GCN 训练一键脚本（Windows 命令提示符用）
REM 用法：在 D:\myproject\elder_decete 下执行  scripts\run_train_stgcn.bat
REM 切到项目根目录（无论从哪里双击/调用都能正确定位）
cd /d %~dp0\..
REM S2 实验：窗口60 + 权重 [1,2,2]（由 [1,5,3] 下调，平衡过渡动作敏感度）
REM 数据集 skeleton_w60 需先由 scripts\build_skeleton.py 生成
REM 不删除旧的 run_01，新结果自动写到下一个 run_XX（run_02...）；run 目录自动递增
python -u scripts\train_stgcn.py --data_dir datasets\experiment\skeleton_w60 --run_dir model\stgcn\stgcn_v1 --epochs 60 --batch_size 64 --loss class_weighted
