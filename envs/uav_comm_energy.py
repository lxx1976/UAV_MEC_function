# -*- coding: utf-8 -*-
#保留！
"""
功能：整合论文中的通信模型（LoS 概率、路径损耗、上行速率）与推进能耗模型（剖面/寄生/诱导/垂直），
     并提供高度更新与垂直速度计算的实用函数。
约定：函数/变量使用英文，注释使用中文。单位请保持一致（SI）。
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Union, Optional
import numpy as np

ArrayLike = Union[float, int, np.ndarray]


# =========================
# 通信模型：LoS 概率 / 路损 / 速率
# =========================


@dataclass
class CommParams:
    """通信模型参数（带默认值），便于统一配置"""
    # 物理/系统参数
    fc_hz: float = 2.4e9            # 载频 (Hz)这个值论文中没给啊，但是好像这个是默认值
    bandwidth_hz: float = 1e6       # 带宽 (Hz)
    p_tx_w: float = 0.1             # 发射功率 (W) ≈ 20 dBm   P_k
    n0_w_per_hz: float = 4e-21      # 噪声谱密度 (W/Hz) ~ -174 dBm/Hz

    # LoS 概率与附加损耗（城市场景示例）
    a: float = 9.61                 # LoS 概率拟合参数 a
    b: float = 0.16                 # LoS 概率拟合参数 b
    eta_los_db: float = 1.0         # LoS 附加损耗 (dB)
    eta_nlos_db: float = 20.0       # NLoS 附加损耗 (dB)

    # 其他
    c_mps: float = 299_792_458.0    # 光速 (m/s)
    snr_min_linear: float = 0.0     # SNR 门限（线性），小于门限按 0 处理
    los_mode: str = "expected"      # "expected"：期望模型；"sample"：伯努利采样
    rng_seed: Optional[int] = 123   # 采样模式的随机种子（可设为 None）


class UAVCommModel:
    """
    通信模型封装（不影响上面的函数）：
    - 提供默认参数与统一接口；支持期望/采样两种 LoS 处理
    - 所有输入支持标量或 ndarray（返回 ndarray）
    """
    def __init__(self, params: CommParams):
        self.p = params
        self.rng = np.random.default_rng(params.rng_seed)

    @staticmethod
    def _np(x: ArrayLike) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    # 仰角（度）
    def elevation_angle_deg(self, h: ArrayLike, d_horizontal: ArrayLike) -> np.ndarray:
        return np.degrees(np.arctan2(self._np(h), np.maximum(self._np(d_horizontal), 1e-9)))

    # LoS 概率
    def p_los(self, theta_deg: ArrayLike) -> np.ndarray:
        a, b = self.p.a, self.p.b
        th = self._np(theta_deg)
        return 1.0 / (1.0 + a * np.exp(-b * (th - a)))

    # FSPL (dB)
    def fspl_db(self, d_3d_m: ArrayLike) -> np.ndarray:
        f = self.p.fc_hz
        c = self.p.c_mps
        d = np.maximum(self._np(d_3d_m), 1e-3)
        x = 4.0 * np.pi * f * d / c
        return 20.0 * np.log10(x)

    # 附加损耗（期望或采样）
    def _extra_loss_db(self, p_los_val: ArrayLike) -> np.ndarray:
        p = self._np(p_los_val)
        etaL, etaN = self.p.eta_los_db, self.p.eta_nlos_db
        if self.p.los_mode == "expected":
            return p * etaL + (1.0 - p) * etaN
        elif self.p.los_mode == "sample":
            u = self.rng.random(size=p.shape)
            los = (u < np.clip(p, 0.0, 1.0))
            return np.where(los, etaL, etaN)
        else:
            raise ValueError("los_mode must be 'expected' or 'sample'.")

    # 路径损耗 (dB)
    def pathloss_db(self, h: ArrayLike, d_horizontal: ArrayLike) -> np.ndarray:
        h_ = self._np(h)
        d_ = self._np(d_horizontal)
        d3d = np.hypot(d_, h_)
        fspl = self.fspl_db(d3d)
        theta = self.elevation_angle_deg(h_, d_)
        p = self.p_los(theta)
        extra = self._extra_loss_db(p)
        return fspl + extra

    # 期望/采样上行速率 (bit/s)
    def rate_bps(self, h: ArrayLike, d_horizontal: ArrayLike) -> np.ndarray:
        L = self.pathloss_db(h, d_horizontal)
        B, Ptx, N0 = self.p.bandwidth_hz, self.p.p_tx_w, self.p.n0_w_per_hz
        Pr = Ptx * (10.0 ** (-L / 10.0))         # 接收功率 (W)
        snr = Pr / (B * N0)                      # 线性 SNR
        if self.p.snr_min_linear > 0:
            snr = np.where(snr >= self.p.snr_min_linear, snr, 0.0)
        return B * np.log2(1.0 + snr)


# =========================
# 便捷封装函数（新增，可选）
# 说明：保持你原始函数不变；以下以 *_cfg 为后缀，走默认参数
# =========================

def expected_pathloss_dB_cfg(h: float, d_horizontal: float, cfg: Optional[CommParams] = None) -> float:
    """用 CommParams 的默认值计算期望路径损耗（不改动原函数）"""
    if cfg is None:
        cfg = CommParams(los_mode="expected")
    model = UAVCommModel(cfg)
    return float(model.pathloss_db(h, d_horizontal))

def expected_uplink_rate_bps_cfg(h: float, d_horizontal: float, cfg: Optional[CommParams] = None) -> float:
    """用 CommParams 的默认值计算期望上行速率（不改动原函数）"""
    if cfg is None:
        cfg = CommParams(los_mode="expected")
    model = UAVCommModel(cfg)
    return float(model.rate_bps(h, d_horizontal))

def instantaneous_uplink_rate_bps_cfg(h: float, d_horizontal: float, cfg: Optional[CommParams] = None) -> float:
    """用 CommParams 的默认值计算采样模式下的瞬时上行速率"""
    if cfg is None:
        cfg = CommParams(los_mode="sample")
    model = UAVCommModel(cfg)
    return float(model.rate_bps(h, d_horizontal))


def elevation_angle_deg(h: float, d_horizontal: float) -> float:
    """仰角 θ(度) = arctan(h/d)。h: 高度(m), d_horizontal: 水平距离(m)"""
    return float(np.degrees(np.arctan2(h, max(d_horizontal, 1e-9))))
#转换为角度值了

def p_los(theta_deg: float, a: float, b: float) -> float:
    """
    LoS 概率模型（常见空-地通道的逻辑函数近似）  theta_deg是arctan(h/d)
    p_LOS(θ) = 1 / (1 + a * exp(-b * (θ - a)))
    说明：a,b 由场景拟合（城市/郊区/密集城区等），θ 为“度数”不是弧度
    """
    return float(1.0 / (1.0 + a * np.exp(-b * (theta_deg - a))))
#pkn，或者说是plos（Los连接概率），这里的theta_deg是arctan(h/d),要实时传入

def sample_los(p: float, rng: Optional[np.random.Generator] = None) -> bool:
    """按给定概率采样 LoS/NLoS（True=LoS, False=NLoS）"""
    if rng is None:
        rng = np.random.default_rng()
    return bool(rng.random() < p)
#采样LoS/NLoS

def fspl_dB(d_3d_m: float, f_c_Hz: float) -> float:
    """
    自由空间路径损耗（dB）
    FSPL = 20*log10(4π f_c d / c)
    """
    c = 299_792_458.0
    x = 4.0 * np.pi * f_c_Hz * max(d_3d_m, 1e-3) / c
    return float(20.0 * np.log10(x))
#C的前一项

def expected_pathloss_dB(h: float,
                         d_horizontal: float,
                         f_c_Hz: float,
                         a: float, b: float,
                         eta_LoS_dB: float,
                         eta_NLoS_dB: float) -> float:
    """
    期望路径损耗（dB）：将 LoS/NLoS 的附加损耗按概率做加权
    L_exp = FSPL(d3D) + [ p*η_LoS + (1-p)*η_NLoS ]
          = FSPL + η_NLoS + p*(η_LoS - η_NLoS)
    """
    d3d = float(np.hypot(d_horizontal, h))  # sqrt(d^2 + h^2)
    fspl = fspl_dB(d3d, f_c_Hz)
    theta = elevation_angle_deg(h, d_horizontal)
    p = p_los(theta, a, b)
    extra = p * eta_LoS_dB + (1.0 - p) * eta_NLoS_dB
    return float(fspl + extra)
#l_kn

def expected_uplink_rate_bps(h: float,
                             d_horizontal: float,
                             f_c_Hz: float,
                             a: float, b: float,
                             eta_LoS_dB: float,
                             eta_NLoS_dB: float,
                             B_Hz: float,
                             P_tx_W: float,
                             N0_W_per_Hz: float) -> float:
    """
    期望上行速率（Shannon 形式）：r = B * log2(1 + SNR_exp)
    其中 SNR_exp 按“期望路径损耗”的平均接收功率计算 (简化)：
      Pr_exp_W = P_tx_W * 10^(-L_exp/10)
      SNR_exp  = Pr_exp_W / (B*N0)
    """
    L_exp_dB = expected_pathloss_dB(h, d_horizontal, f_c_Hz, a, b, eta_LoS_dB, eta_NLoS_dB)
    Pr_exp_W = P_tx_W * 10.0 ** (-L_exp_dB / 10.0)
    snr = Pr_exp_W / (B_Hz * N0_W_per_Hz)
    return float(B_Hz * np.log2(1.0 + snr))
#rkn

# =========================
# 高度更新 / 速度计算
# =========================

def update_height(h_old: float,
                  action_vertical: int,
                  dt: float,
                  h_min: float,
                  h_max: float,
                  delta_h: float) -> Tuple[float, float]:
    """
    更新无人机高度并返回垂直速度 v_v（m/s）
    参数:
        h_old: 上一时刻高度
        action_vertical: 垂直动作 (0=保持, 1=上升, 2=下降)
        dt: step 时长 (s)
        h_min, h_max: 高度上下界
        delta_h: 单步高度变化量
    返回:
        h_new: 约束后的新高度
        v_v: 垂直速度 (m/s) = (h_new - h_old) / dt
    """
    # 基于离散动作更新高度（上升/下降/保持），按照  env_uav_edge_computing.py 中的编码：  - vertical_action = 0：下降  - vertical_action = 1：上升  - vertical_action = 2：悬停 来修改了一下
    if action_vertical == 1:
        h_new_raw = h_old + delta_h
    elif action_vertical == 0:
        h_new_raw = h_old - delta_h
    else:
        h_new_raw = h_old

    # 应用高度约束
    h_new = float(np.clip(h_new_raw, h_min, h_max))

    # 计算垂直速度（避免 dt 为 0）
    v_v = (h_new - h_old) / max(dt, 1e-9)
    return h_new, v_v


def speeds_from_positions(p_old: np.ndarray,
                          p_new: np.ndarray,
                          dt: float) -> Tuple[float, float]:
    """
    基于三维位置计算水平/垂直速度
    参数:
        p_old, p_new: 形如 [x, y, h] 的位置向量
        dt: 时间步长 (s)
    返回:
        v_h: 水平速度 (m/s)
        v_v: 垂直速度 (m/s)
    """
    dp = (np.asarray(p_new, dtype=np.float64) - np.asarray(p_old, dtype=np.float64))
    v_h = float(np.linalg.norm(dp[:2]) / max(dt, 1e-9))
    v_v = float(dp[2] / max(dt, 1e-9))
    return v_h, v_v


# =========================
# 推进能耗模型（剖面/寄生/诱导/垂直）
# =========================

@dataclass
class RotorcraftParams:
    """
    旋翼机推进能耗模型参数
    - P0、P1 采用论文表格的表达式，已带入具体数值：rho=1.225, s=0.05, G=0.503
    - P2 按论文表格设为常数 11.46（默认不考虑下降回收）
    - 其他参数按常见量级初始化，可按平台修改
    """
    # —— 常量与环境参数（对寄生/诱导项有影响）——
    U_tip: float = 120.0     # m/s，桨尖速度
    v0: float = 4.3          # m/s，悬停诱导速度
    d0: float = 0.6          # 无量纲，机身阻力比
    rho: float = 1.225       # kg/m^3，空气密度（用于寄生项）
    s: float = 0.05          # 无量纲，旋翼实度（用于寄生项）
    G: float = 0.503         # m^2，总桨盘面积（用于寄生项）
    eps: float = 1e-9        # 数值稳定项

    # —— 论文表格三常数（带入数值后的“公式”写死）——
    # P0 = (12 * 30^3 * 0.4^3 / 8) * rho * s * G
    P0: float = (12.0 * (30.0**3) * (0.4**3) / 8.0) * 1.225 * 0.05 * 0.503  # ≈ 79.8563 W

    # P1 = (1.1 * 20^(3/2)) / sqrt(2 * rho * G)
    P1: float = (1.1 * (20.0**1.5)) / np.sqrt(2.0 * 1.225 * 0.503)          # ≈ 88.6279 W

    # P2 给定为常数 11.46（W/(m/s)）
    P2: float = 11.46

    # —— 质量/重力仅在你要改动 P2≈m*g 时会用到（此处默认不用）——
    mass_kg: float = 1.5
    g: float = 9.80665


def uav_power_components(v_h: ArrayLike,
                         v_v: ArrayLike,
                         params: RotorcraftParams,
                         descent_saving: bool = False) -> Dict[str, np.ndarray]:
    """
    计算各推进功率分量与总功率（单位：W）
    参数:
        v_h: 水平速度 (m/s)，标量或数组
        v_v: 垂直速度 (m/s)，向上为正，向下为负
        params: 旋翼参数
        descent_saving: 是否允许下降回收（默认 False）
    返回:
        dict: {profile, parasite, induced, vertical, total}
    """
    v_h = np.asarray(v_h, dtype=np.float64)
    v_v = np.asarray(v_v, dtype=np.float64)

    # 1) 剖面功率：P0 * (1 + 3*v_h^2 / U_tip^2)
    profile = params.P0 * (1.0 + 3.0 * (v_h**2) / (params.U_tip**2 + params.eps))

    # 2) 寄生功率：0.5 * d0 * rho * s * G * v_h^3
    parasite = 0.5 * params.d0 * params.rho * params.s * params.G * (v_h**3)

    # 3) 诱导功率：P1 * ( sqrt(1 + (v_h^4)/(4*v0^4)) - (v_h^2)/(2*v0^2) )^(1/2)
    #    令 y = v_h^2 / (2*v0^2)，则 = P1 * sqrt( sqrt(1 + y^2) - y )
    y = (v_h**2) / (2.0 * (params.v0**2 + params.eps))
    inner = np.sqrt(1.0 + y*y) - y
    inner = np.maximum(inner, 0.0)  # 数值保护
    induced = params.P1 * np.sqrt(inner)

    # 4) 垂直功率：P2 * v_v_plus（默认不回收下降能量）
    v_v_plus = v_v if descent_saving else np.maximum(v_v, 0.0)
    vertical = params.P2 * v_v_plus

    total = profile + parasite + induced + vertical
    return {"profile": profile, "parasite": parasite, "induced": induced, "vertical": vertical, "total": total}



def uav_step_energy(v_h: ArrayLike,
                    v_v: ArrayLike,
                    t_step: ArrayLike,
                    params: RotorcraftParams,
                    descent_saving: bool = False) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    计算单个 step 的推进能量（单位：J）
    参数:
        v_h: 水平速度 (m/s)
        v_v: 垂直速度 (m/s)
        t_step: step 时长 (s)
        params: 旋翼参数
        descent_saving: 是否允许下降回收（默认 False）
    返回:
        energy_J: 本 step 的能量 (J)
        power_dict: 各功率分量（W）
    """
    t_step = np.asarray(t_step, dtype=np.float64)
    power = uav_power_components(v_h, v_v, params, descent_saving=descent_saving)
    energy = power["total"] * t_step
    return energy, power




# =========================
# 通信–计算耦合（卸载）补充：alpha、传输量、处理量
# =========================

def allocate_bandwidth(B_total_Hz: float, weights: ArrayLike) -> np.ndarray:
    """
    简单 OMA/TDMA 带宽分配：按给定权重瓜分总带宽
    参数:
        B_total_Hz: 总带宽 (Hz)
        weights: 权重向量 (K, )，非负；全零时平均分配
    返回:
        B_k: (K,) 每个用户得到的带宽 (Hz)
    """
    w = np.asarray(weights, dtype=np.float64)
    K = w.size
    if np.all(w <= 0):
        return np.full(K, B_total_Hz / max(K, 1), dtype=np.float64)
    wsum = np.sum(w)
    return B_total_Hz * (w / wsum)


def comm_offload_alpha(r_bps: ArrayLike, f_u_cps: float, F_cycles: ArrayLike, D_bits: ArrayLike) -> np.ndarray:
    """
   计算一个时隙内通信和计算这两者时间分配的最优占比（α）
   论文中的闭式时间分配:
      alpha = (f_u * D) / (r * F + f_u * D)
    参数:
        r_bps: 上行速率 (bit/s)
        f_u_cps: UAV CPU 频率 (cycles/s)
        F_cycles: 任务总 CPU 周期数 (cycles/bit * bit = cycles) 或每任务总 cycles
        D_bits: 任务数据量 (bit)
    返回:
        alpha: (与输入可广播) ∈ [0,1]
    """
    r = np.asarray(r_bps, dtype=np.float64)
    F = np.asarray(F_cycles, dtype=np.float64)
    D = np.asarray(D_bits, dtype=np.float64)
    num = f_u_cps * D
    den = r * F + num
    alpha = num / np.maximum(den, 1e-12)
    return np.clip(alpha, 0.0, 1.0)


def comm_tx_bits(a_kn: ArrayLike, alpha: ArrayLike, t_slot_s: ArrayLike, r_bps: ArrayLike) -> np.ndarray:
    """
    上传bit数
    时隙内的上行传输量:
      c_{k,n} = a_{k,n} * alpha_n * t_n^u * r_{k,n}
    参数:
        a_kn: 卸载决策 (0/1)
        alpha: 传输时间占比
        t_slot_s: 时隙时长 (s)
        r_bps: 上行速率 (bit/s)
    返回:
        c_bits: 传输比特数
    """
    a = np.asarray(a_kn, dtype=np.float64)
    al = np.asarray(alpha, dtype=np.float64)
    t = np.asarray(t_slot_s, dtype=np.float64)
    r = np.asarray(r_bps, dtype=np.float64)
    return a * al * t * r


def comm_processed_bits(a_kn: ArrayLike,
                        f_u_cps: float,
                        D_bits: ArrayLike,
                        t_slot_s: ArrayLike,
                        r_bps: ArrayLike,
                        F_cycles: ArrayLike,
                        f_g_cps: float) -> np.ndarray:
    """
    这里是每个时间间隙内上传的数据量或者是地面终端自己计算掉的数据量，这个我觉得可以优化，因为最终我的需求是计算掉的量，所以不应该使用上传量而是使用uav计算掉的量
    时隙内被处理的数据量（远程+本地）:
      d_{k,n} = a * [ f_u * D * t * r / (r*F + f_u*D) ] + (1 - a) * t * f_g
    参数:
        a_kn: 卸载决策 (0/1)
        f_u_cps: UAV CPU 频率 (cycles/s)
        D_bits: 任务数据量 (bit)
        t_slot_s: 时隙时长 (s)
        r_bps: 上行速率 (bit/s)
        F_cycles: 任务 CPU 周期数 (cycles/bit * bit 或总 cycles)
        f_g_cps: 终端本地 CPU 频率 (cycles/s)，这里假设 1 bit 对应 1 cycle，若有 cycles/bit，请自行换算
    返回:
        d_bits: 时隙内完成的“等效数据量”(bit)
    说明:
        若你的实现中 f_g_cps 与 F_cycles 以 cycles/bit 绑定，亦可改写本地项为：t * f_g_cps / (F_cycles_per_bit)
    """
    a = np.asarray(a_kn, dtype=np.float64)
    D = np.asarray(D_bits, dtype=np.float64)
    t = np.asarray(t_slot_s, dtype=np.float64)
    r = np.asarray(r_bps, dtype=np.float64)
    F = np.asarray(F_cycles, dtype=np.float64)

    # 远程（卸载）部分
    denom = r * F + f_u_cps * D
    remote = f_u_cps * D * t * r / np.maximum(denom, 1e-12)

    # 本地部分（简化为：bit/s = cycles/s / (cycles/bit)=f_g/ (F/D)，此处按论文给出形式用 t*f_g）
    local = t * f_g_cps

    return a * remote + (1.0 - a) * local


# =========================
# 可选：瞬时（随机）LoS/NLoS 版本
# =========================

def instantaneous_pathloss_dB(h: float,
                              d_horizontal: float,
                              f_c_Hz: float,
                              a: float, b: float,
                              eta_LoS_dB: float,
                              eta_NLoS_dB: float,
                              rng: Optional[np.random.Generator] = None) -> float:
    """
    先采样 LoS/NLoS，再计算瞬时路径损耗（dB）
    """
    d3d = float(np.hypot(d_horizontal, h))
    base = fspl_dB(d3d, f_c_Hz)
    theta = elevation_angle_deg(h, d_horizontal)
    p = p_los(theta, a, b)
    los = sample_los(p, rng)
    extra = eta_LoS_dB if los else eta_NLoS_dB
    return float(base + extra)


def instantaneous_uplink_rate_bps(h: float,
                                  d_horizontal: float,
                                  f_c_Hz: float,
                                  a: float, b: float,
                                  eta_LoS_dB: float,
                                  eta_NLoS_dB: float,
                                  B_Hz: float,
                                  P_tx_W: float,
                                  N0_W_per_Hz: float,
                                  rng: Optional[np.random.Generator] = None) -> float:
    """
    瞬时上行速率（香农形式），基于瞬时路径损耗
    """
    L_dB = instantaneous_pathloss_dB(h, d_horizontal, f_c_Hz, a, b, eta_LoS_dB, eta_NLoS_dB, rng)
    Pr_W = P_tx_W * 10.0 ** (-L_dB / 10.0)
    snr = Pr_W / (B_Hz * N0_W_per_Hz)
    return float(B_Hz * np.log2(1.0 + snr))
