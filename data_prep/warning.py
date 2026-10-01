# -*- coding: utf-8 -*-
"""
感知降级预警（Perception Degradation Index, PDI）
==================================================
思想来源：不确定性/OOD 检测——模型"知道自己不行了"就该报警，而不是硬输出。
PDI = 亮度漂移分 × 0.55 + 置信度衰减分 × 0.45，0~100，越高越危险。
可直接用于真实管线（每帧算一次），演示网页里也用它驱动红色预警横幅。

用法：
  from warning import compute_pdi, advise
    pdi = compute_pdi(brightness=gray_mean_0_255, conf_mean=mean_det_conf)
    level, msg = advise(pdi)
"""
import math


def compute_pdi(brightness, conf_mean=None, normal_band=(90, 190)):
    """
    brightness: 当前帧灰度均值 (0~255)
    conf_mean:  当前帧检测置信度均值 (0~1)，None 时只看亮度
    normal_band: 正常亮度区间，越出越扣分
    """
    lo, hi = normal_band
    if brightness < lo:
        b_score = min(1.0, (lo - brightness) / lo)
    elif brightness > hi:
        b_score = min(1.0, (brightness - hi) / (255 - hi))
    else:
        b_score = 0.0
    c_score = 0.0
    if conf_mean is not None:
        # 置信度 0.5 以上视为健康，线性衰减到 0
        c_score = max(0.0, min(1.0, (0.5 - conf_mean) / 0.5))
    pdi = 100 * (0.55 * b_score + (0.45 if conf_mean is not None else 0.0) * c_score)
    if conf_mean is None:
        pdi = 100 * b_score          # 无检测信息时退化为纯亮度指标
    return round(min(pdi, 100.0), 1)


def advise(pdi):
    """返回 (等级, 提示语)。等级阈值可按实车标定调整。"""
    if pdi < 30:
        return "正常", "感知正常"
    if pdi < 55:
        return "关注", "光照异常，感知置信度下降，请保持车距"
    if pdi < 75:
        return "降级", "感知降级，建议减速并提高注意力"
    return "严重", "感知严重受限，建议立即减速/接管"


if __name__ == "__main__":
    for b, c in [(120, 0.6), (80, 0.45), (45, 0.30), (20, 0.15), (150, None)]:
        p = compute_pdi(b, c)
        lv, msg = advise(p)
        print("亮度%3d 置信度%s -> PDI %5.1f  [%s] %s" % (b, c, p, lv, msg))
