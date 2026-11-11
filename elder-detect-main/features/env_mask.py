"""
features/env_mask.py — 环境掩码构建器（轨道B 几何层）

逐帧环境危险点空间掩码 M(t)。三种模式：

  vlm_coarse (v0, 保留兼容): VLM hazard 标签 → 地面ROI填1 + 人脚挖孔
    ⚠️ 运行时依赖 VLM 标签，env_labels 未覆盖即失效，已不推荐

  bg_diff (v1): MOG2 背景差分(检测新增物) + HSV 反光(水渍) + 地面ROI
    + 人脚挖孔 + 合并 VLM 静态障碍物 bbox。
    ✅ 运行时纯 OpenCV，不调 VLM API（VLM 仅部署时一次性标定静态基线）。

  absdiff (v2, 推荐·零依赖): 一张"干净空房基准图"做帧差
    ------------------------------------------------
    逻辑：部署时截取一张地面干净、无人的基准背景图；
         实时帧与基准图做 cv2.absdiff → 二值化 → 超过阈值即"地上多了一块东西"。
         不用 MOG2、不用 VLM 打标。纯 OpenCV，确定性高、几乎不误报。
    构建：EnvMaskBuilder.from_baseline_image(干净图路径或数组) → mode="absdiff"。
    配套：差分前高斯模糊去噪 → 人脚/整个人挖孔(移动中的老人会差出人形) →
          连通域面积过滤(去小噪声) → 地面ROI(只保留下方) → 突变保护(光照突变整帧清空)。

VLM 离线标定见 scripts/vlm_baseline_calibrate.py → env_baselines/<scene>.json
  含 ground_top_y(地面顶部y) + obstacles(固定障碍物bbox)。
EnvMaskBuilder.from_baseline(json) 加载基线 → bg_diff 模式运行时纯算法。
干净基准图获取见 scripts/capture_baseline.py（pose 找无人帧 → 中值合成，不用 VLM）。

设计见 docs/环境交互风险预测_设计.md §5 通道1。
用到的关键点：LEFT_HEEL(29)/RIGHT_HEEL(30)，fallback LEFT_ANKLE(27)/RIGHT_ANKLE(28)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pose.keypoint import BodyPart
from scripts.build_config import (
    ENV_MASK_MODE,
    ENV_MASK_H,
    ENV_MASK_W,
    ENV_MASK_GROUND_RATIO,
    ENV_MASK_FOOT_RADIUS,
    ENV_MASK_POKE_PERSON,
    ENV_MASK_PERSON_MARGIN,
    ENV_MASK_USE_SPECULAR,
    ENV_MASK_SPEC_MIN_AREA,
    ENV_MASK_MOG2_VAR,
    ENV_MASK_MIN_AREA_RATIO,
    ENV_MASK_MAX_AREA_RATIO,
    ENV_MASK_CLOSE_KERNEL,
    ENV_MASK_ABS_THRESH,
    ENV_MASK_ABS_BLUR,
    ENV_MASK_ABS_USE_COLOR,
    ENV_MASK_GROUND_ROI,
)

# HSV 反光（水渍）阈值：V 高亮 + S 低饱和
_SPEC_V_MIN = 200   # 明度高于此 → 疑似反光
_SPEC_S_MAX = 60    # 饱和度低于此 → 非彩色（水渍近白/灰反光）


class EnvMaskBuilder:
    """环境掩码构建器

    Args:
        mode: ``"bg_diff"``（v1 推荐，纯算法）| ``"vlm_coarse"``（v0 兼容）
        mask_h, mask_w: 输出掩码降采样尺寸
        ground_ratio: vlm_coarse 地面 ROI 占下方比例（bg_diff 用 ground_top_y）
        foot_radius: 人脚挖孔半径（归一化坐标）
        ground_top_y: VLM 基线地面顶部 y（归一化），None 时 bg_diff 用 1−ground_ratio
        obstacles: VLM 基线固定障碍物列表 [{bbox:[x,y,w,h], type}]
    """

    def __init__(
        self,
        mode: str = ENV_MASK_MODE,
        mask_h: int = ENV_MASK_H,
        mask_w: int = ENV_MASK_W,
        ground_ratio: float = ENV_MASK_GROUND_RATIO,
        foot_radius: float = ENV_MASK_FOOT_RADIUS,
        ground_top_y: float | None = None,
        obstacles: list | None = None,
        poke_person: bool = ENV_MASK_POKE_PERSON,
        person_margin: float = ENV_MASK_PERSON_MARGIN,
        use_specular: bool = ENV_MASK_USE_SPECULAR,
        spec_min_area: float = ENV_MASK_SPEC_MIN_AREA,
        mog2_var: int = ENV_MASK_MOG2_VAR,
        min_area_ratio: float = ENV_MASK_MIN_AREA_RATIO,
        max_area_ratio: float = ENV_MASK_MAX_AREA_RATIO,
        close_kernel: int = ENV_MASK_CLOSE_KERNEL,
        # —— absdiff (v2) 参数 ——
        bg_image=None,
        abs_threshold: int = ENV_MASK_ABS_THRESH,
        abs_blur: int = ENV_MASK_ABS_BLUR,
        abs_use_color: bool = ENV_MASK_ABS_USE_COLOR,
        ground_roi: bool = ENV_MASK_GROUND_ROI,
    ) -> None:
        self.mode = mode
        self.mask_h = mask_h
        self.mask_w = mask_w
        self.ground_ratio = ground_ratio
        self.foot_radius = foot_radius
        self.ground_top_y = ground_top_y
        self.obstacles = obstacles or []
        self.poke_person = poke_person
        self.person_margin = person_margin
        self.use_specular = use_specular
        self.spec_min_area = spec_min_area
        self.mog2_var = mog2_var
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.close_kernel = close_kernel
        self.abs_threshold = abs_threshold
        self.abs_blur = abs_blur
        self.abs_use_color = abs_use_color
        self.ground_roi = ground_roi
        self._foot_parts = (
            BodyPart.LEFT_HEEL, BodyPart.RIGHT_HEEL,
            BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE,
        )
        # 参与 bbox 的关键点（用于挖掉整个人，避免把移动中的老人当前景）
        self._person_parts = (
            BodyPart.NOSE, BodyPart.LEFT_EYE, BodyPart.RIGHT_EYE,
            BodyPart.LEFT_EAR, BodyPart.RIGHT_EAR,
            BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER,
            BodyPart.LEFT_ELBOW, BodyPart.RIGHT_ELBOW,
            BodyPart.LEFT_WRIST, BodyPart.RIGHT_WRIST,
            BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP,
            BodyPart.LEFT_KNEE, BodyPart.RIGHT_KNEE,
            BodyPart.LEFT_ANKLE, BodyPart.RIGHT_ANKLE,
            BodyPart.LEFT_HEEL, BodyPart.RIGHT_HEEL,
        )
        self._bg = None          # MOG2 背景模型（bg_diff 用）
        self._bg_img = None      # 干净基准图（absdiff 用），BGR uint8 @ (mask_h, mask_w)
        if mode == "bg_diff":
            import cv2
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=mog2_var, detectShadows=False)
        elif mode == "absdiff":
            if bg_image is not None:
                self._load_baseline_image(bg_image)

    @classmethod
    def from_baseline(cls, json_path, **kw) -> "EnvMaskBuilder":
        """从 VLM 基线 JSON 构建（bg_diff 模式 + VLM 静态基线）。

        VLM 部署时一次性标定的 ground_top_y + obstacles 在此加载，
        之后运行时纯算法，不再调 VLM。
        """
        d = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return cls(
            mode="bg_diff",
            ground_top_y=d.get("ground_top_y"),
            obstacles=d.get("obstacles", []),
            **kw,
        )

    @classmethod
    def from_baseline_image(cls, img, ground_top_y: float | None = None, **kw) -> "EnvMaskBuilder":
        """从一张「干净空房基准图」构建 absdiff 模式（v2，零依赖、不用 VLM）。

        Args:
            img: 基准图路径(str/Path) 或 已加载的 BGR ndarray。
                 「干净」= 地面无杂物、画面中无人；部署时截取一次即可。
            ground_top_y: 地面顶部 y（归一化）。默认 None → 用 1 - ground_ratio（下方 1/3）。
                         仅影响地面 ROI（absdiff 已在全帧上差分，ROI 用于聚焦"地上"）。

        逻辑：实时帧与该基准图做 cv2.absdiff → 二值化 → 超阈值即新增障碍物。
        """
        return cls(mode="absdiff", bg_image=img, ground_top_y=ground_top_y, **kw)

    def set_baseline_image(self, img) -> None:
        """运行时补设/替换干净基准图（用于自动标定后再注入）。"""
        if self.mode != "absdiff":
            raise RuntimeError("set_baseline_image 仅 absdiff 模式可用")
        self._load_baseline_image(img)

    def _load_baseline_image(self, img) -> None:
        """加载基准图 → resize 到 (mask_h, mask_w) → 存为 BGR uint8。"""
        import cv2
        if isinstance(img, (str, Path)):
            bgr = cv2.imread(str(img), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"读不到基准背景图: {img}")
        elif isinstance(img, np.ndarray):
            bgr = img
        else:
            raise TypeError(f"bg_image 类型不支持: {type(img)}")
        if bgr.shape[:2] != (self.mask_h, self.mask_w):
            bgr = cv2.resize(bgr, (self.mask_w, self.mask_h), interpolation=cv2.INTER_AREA)
        self._bg_img = bgr.astype(np.uint8)

    def build_per_frame(self, result, has_hazard: bool = False, frame_bgr=None) -> np.ndarray:
        """逐帧构建环境掩码 M(t)。

        Args:
            result: PoseResult（人脚挖孔定位）
            has_hazard: vlm_coarse 模式用（是否有环境隐患）
            frame_bgr: bg_diff 模式用（BGR 帧数组 H×W×3）

        Returns:
            (mask_h, mask_w) uint8 二值掩码，1 = 危险点
        """
        if self.mode == "bg_diff":
            if frame_bgr is None:
                raise ValueError("bg_diff 模式需要 frame_bgr（BGR 帧数组）")
            return self._build_bg_diff(frame_bgr, result)
        if self.mode == "absdiff":
            if frame_bgr is None:
                raise ValueError("absdiff 模式需要 frame_bgr（BGR 帧数组）")
            return self._build_absdiff(frame_bgr, result)
        return self._build_vlm_coarse(result, has_hazard)

    # ==================================================================
    # bg_diff 模式（v1 纯算法）
    # ==================================================================

    def _build_bg_diff(self, frame_bgr: np.ndarray, result) -> np.ndarray:
        import cv2
        H, W = frame_bgr.shape[:2]

        # 1. MOG2 背景差分（在线学习背景，检测新增物）。varThreshold 收紧以抑制小噪声。
        fg = self._bg.apply(frame_bgr)  # uint8 0/255

        mask_full = (fg > 0)

        # 2. HSV 反光检测（水渍：高亮 + 低饱和）。默认关闭——误报最重：
        #    白墙/亮地/投影幕布/窗户都满足"高亮+低饱和"，几乎必然大面积误报。
        if self.use_specular:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            specular = ((hsv[:, :, 2] > _SPEC_V_MIN) & (hsv[:, :, 1] < _SPEC_S_MAX)).astype(np.uint8)
            # 仅保留面积足够大的反光连通域，丢弃小亮点
            spec_area = self.spec_min_area * W * H
            num, lab, stats = cv2.connectedComponentsWithStats(specular, 8)[:3]
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] < spec_area:
                    specular[lab == i] = 0
            mask_full |= (specular > 0)

        # 3. 合并 VLM 静态障碍物 bbox（固定障碍物区也是危险点）
        for ob in self.obstacles:
            x, y, w, h = ob.get("bbox", [0, 0, 0, 0])
            px, py, pw, ph = int(x * W), int(y * H), int(w * W), int(h * H)
            mask_full[py:py + ph, px:px + pw] = True

        # 4. 形态学：开运算去噪 + 闭运算填小洞（close_kernel=0 则跳过闭运算）。
        #    注意：此步在挖人之前，否则闭运算的膨胀会把"挖掉的人洞"重新填回。
        mask_u8 = (mask_full.astype(np.uint8)) * 255
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k_open)
        if self.close_kernel and self.close_kernel > 0:
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (self.close_kernel, self.close_kernel))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k_close)
        mask_full = (mask_u8 > 127)

        # 5. 连通域面积过滤：丢弃面积 < min_area_ratio 的噪声连通域（灯光闪烁/尘埃）
        #    注意：connectedComponentsWithStats 的标签 0 是背景（面积=全图），必须排除，
        #    否则 keep[0]=True 会把整张背景图标成前景（整间房被标红的真正元凶）。
        if self.min_area_ratio > 0 and mask_full.sum() > 0:
            num, lab, stats = cv2.connectedComponentsWithStats(mask_u8, 8)[:3]
            min_area = self.min_area_ratio * W * H
            keep = np.zeros(num, dtype=bool)  # 默认全 False（含背景标签 0）
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    keep[i] = True
            mask_full = keep[lab].astype(bool)

        # 6. 人挖孔：先挖脚，再挖整个人 bbox（关键：避免 MOG2 把移动中的老人当危险）。
        #    必须放在形态学之后——否则闭运算会把人洞重新填回。挖人后不再做膨胀操作。
        if result is not None:
            self._poke_foot_holes_full(mask_full, result, W, H)
            if self.poke_person:
                self._poke_person_box_full(mask_full, result, W, H)

        # 7. 突变保护：前景总占比过高（光照突变/切场景）→ MOG2 整帧不可信，清空防整帧误报
        if self.max_area_ratio > 0 and mask_full.mean() > self.max_area_ratio:
            mask_full[:] = False

        # 8. 地面区域限定（VLM 基线 ground_top_y 或默认下方 ground_ratio）
        gty = self.ground_top_y if self.ground_top_y is not None else (1.0 - self.ground_ratio)
        ground_top_px = int(H * gty)
        roi = np.zeros((H, W), dtype=bool)
        roi[ground_top_px:, :] = True
        mask_full &= roi

        # 9. 降采样到 mask_h × mask_w（边缘端友好）
        mask_u8 = (mask_full.astype(np.uint8)) * 255
        mask = cv2.resize(mask_u8, (self.mask_w, self.mask_h), interpolation=cv2.INTER_AREA)
        return (mask > 127).astype(np.uint8)

    # ==================================================================
    # absdiff 模式（v2 零依赖：干净基准图帧差，不用 MOG2 / 不用 VLM）
    # ==================================================================

    def _build_absdiff(self, frame_bgr: np.ndarray, result) -> np.ndarray:
        """帧差法：实时帧 − 干净基准图 → 二值化 → 新增障碍物。

        流程：尺寸对齐 → 高斯模糊去噪 → absdiff → (彩色取最大通道/灰度) →
              阈值二值化 → 开运算去噪 → 连通域面积过滤 → 人挖孔 →
              突变保护 → 地面 ROI → 降采样输出。
        """
        import cv2
        if self._bg_img is None:
            raise RuntimeError("absdiff 模式需要先提供干净基准图："
                               "用 from_baseline_image() / set_baseline_image() 注入")
        H, W = frame_bgr.shape[:2]
        # 1. 尺寸对齐到掩码分辨率（process_frame 通常已对齐；保险再做一次）
        cur = frame_bgr
        if cur.shape[:2] != (self.mask_h, self.mask_w):
            cur = cv2.resize(cur, (self.mask_w, self.mask_h), interpolation=cv2.INTER_AREA)
        bg = self._bg_img  # 已在 __init__/_load_baseline_image 中 resize 好

        # 2. 差分前高斯模糊：压制传感器噪声 / 视频压缩块效应造成的满屏小噪声
        if self.abs_blur and self.abs_blur > 0:
            k = int(self.abs_blur) | 1  # 核必须是奇数
            cur = cv2.GaussianBlur(cur, (k, k), 0)
            bg = cv2.GaussianBlur(bg, (k, k), 0)

        # 3. 绝对差值（absdiff）—— 不关心"是什么"，只关心"和干净时不一样"
        diff = cv2.absdiff(cur, bg)
        if self.abs_use_color:
            # 取三通道差值的最大值：能抓出灰度相近但彩色不同的异物（如红毛巾 vs 灰地）
            g = diff.max(axis=2)
        else:
            g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(g, self.abs_threshold, 255, cv2.THRESH_BINARY)
        mask_full = (fg > 0)

        # 4. 形态学开运算：去掉孤立噪点（灯光闪烁/尘埃/压缩伪影）
        mask_u8 = (mask_full.astype(np.uint8)) * 255
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k_open)
        mask_full = (mask_u8 > 127)

        # 5. 连通域面积过滤：丢弃 < min_area_ratio 的噪声连通域。
        #    关键：标签0是背景(面积=全图)必须排除，否则 keep[0]=True 把整帧标红。
        if self.min_area_ratio > 0 and mask_full.sum() > 0:
            num, lab, stats = cv2.connectedComponentsWithStats(mask_u8, 8)[:3]
            min_area = self.min_area_ratio * W * H
            keep = np.zeros(num, dtype=bool)
            for i in range(1, num):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    keep[i] = True
            mask_full = keep[lab].astype(bool)

        # 6. 人挖孔：移动中的老人会和"空房基准"差出整个人形 → 必须把人挖掉，
        #    否则老人自己被当成"新增障碍物"。先挖脚再挖整个人 bbox。
        if result is not None:
            self._poke_foot_holes_full(mask_full, result, W, H)
            if self.poke_person:
                self._poke_person_box_full(mask_full, result, W, H)

        # 7. 突变保护：前景总占比过高（光照突变/窗帘飘动/切场景）→ 整帧不可信，清空
        if self.max_area_ratio > 0 and mask_full.mean() > self.max_area_ratio:
            mask_full[:] = False

        # 8. 地面区域限定（默认下方 1/3），聚焦"地上多了一块东西"
        if self.ground_roi:
            gty = self.ground_top_y if self.ground_top_y is not None else (1.0 - self.ground_ratio)
            ground_top_px = int(H * gty)
            roi = np.zeros((H, W), dtype=bool)
            roi[ground_top_px:, :] = True
            mask_full &= roi

        # 9. 降采样到 mask_h × mask_w（边缘端友好）
        mask_u8 = (mask_full.astype(np.uint8)) * 255
        mask = cv2.resize(mask_u8, (self.mask_w, self.mask_h), interpolation=cv2.INTER_AREA)
        return (mask > 127).astype(np.uint8)

    # ==================================================================
    # vlm_coarse 模式（v0 兼容）
    # ==================================================================

    def _build_vlm_coarse(self, result, has_hazard: bool) -> np.ndarray:
        mask = np.zeros((self.mask_h, self.mask_w), dtype=np.uint8)
        if not has_hazard:
            return mask
        roi_top = int(self.mask_h * (1.0 - self.ground_ratio))
        mask[roi_top:, :] = 1
        if result is not None:
            self._poke_foot_holes(mask, result)
        return mask

    # ==================================================================
    # 人脚挖孔
    # ==================================================================

    def _poke_foot_holes(self, mask: np.ndarray, result) -> None:
        """vlm_coarse：降采样坐标挖孔。"""
        r = max(1, int(self.foot_radius * self.mask_w))
        yy, xx = np.ogrid[:self.mask_h, :self.mask_w]
        for part in self._foot_parts:
            kp = result.get(part)
            if kp is None or kp.visibility < 0.5:
                continue
            cx = int(kp.x * self.mask_w)
            cy = int(kp.y * self.mask_h)
            mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 0

    def _poke_foot_holes_full(self, mask: np.ndarray, result, W: int, H: int) -> None:
        """bg_diff：原图坐标挖脚孔。"""
        r = max(3, int(self.foot_radius * W))
        yy, xx = np.ogrid[:H, :W]
        for part in self._foot_parts:
            kp = result.get(part)
            if kp is None or kp.visibility < 0.5:
                continue
            cx = int(kp.x * W)
            cy = int(kp.y * H)
            mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = False

    def _poke_person_box_full(self, mask: np.ndarray, result, W: int, H: int) -> None:
        """bg_diff：用可见关键点算整个人 bbox 并挖掉（避免 MOG2 把移动中的老人当危险物）。

        外扩 person_margin 防止手臂/衣服边缘漏标；clamp 到画面内。
        """
        xs, ys = [], []
        for part in self._person_parts:
            kp = result.get(part)
            if kp is None or kp.visibility < 0.5:
                continue
            xs.append(kp.x)
            ys.append(kp.y)
        if not xs:
            return
        m = self.person_margin
        x0 = max(0.0, min(xs) - m)
        y0 = max(0.0, min(ys) - m)
        x1 = min(1.0, max(xs) + m)
        y1 = min(1.0, max(ys) + m)
        px0, py0 = int(x0 * W), int(y0 * H)
        px1, py1 = int(x1 * W), int(y1 * H)
        mask[py0:py1, px0:px1] = False

    # ==================================================================
    # 生命周期
    # ==================================================================

    def reset(self) -> None:
        """重置背景模型（切换视频/场景时调用）。"""
        if self.mode == "bg_diff" and self._bg is not None:
            import cv2
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=25, detectShadows=False)
