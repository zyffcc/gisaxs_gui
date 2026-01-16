# utils/fitting.py
import numpy as np

__all__ = [
    "make_mixed_model",        # <- 工厂：根据 spec 生成可拟合函数 f(q,*p)
    "params_template",         # <- 给定 spec 返回参数名顺序
    "sphere_form_factor_pd",   # 单形状工具
    "cylinder_form_factor_pd", # 单形状工具（含各向平均）
    "structure_factor_1d",
    "gaussian_resolution_smear",
    "apply_scaling_factor",    # 应用缩放因子
    "mixed_model_components",  # 分解各组分：粒子贡献、总BG、分辨率核
]

# =========================
#  Common utilities
# =========================
def _gaussian_grid(mu, sigma, nsig=4.0, n=25, clip_min=0.0):
    """生成高斯采样点与权重（截断并归一化）"""
    if sigma <= 0:
        return np.array([max(mu, clip_min)], dtype=float), np.array([1.0], dtype=float)
    lo = max(clip_min, mu - nsig * sigma)
    hi = mu + nsig * sigma
    if hi <= lo:
        return np.array([max(mu, clip_min)], dtype=float), np.array([1.0], dtype=float)
    x = np.linspace(lo, hi, int(n))
    w = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    w /= (w.sum() + 1e-300)
    return x, w

def apply_scaling_factor(I, k):
    """应用缩放因子 k 到强度数据"""
    if k is None or k <= 0:
        return I
    return np.asarray(I, dtype=float) * k

def gaussian_resolution_smear(q, I, sigma_res):
    """中心在 0 的高斯分辨率展宽；q 等间距更快，不等距也能跑但更慢。"""
    if sigma_res is None or sigma_res <= 0:
        return I
    q = np.asarray(q, dtype=float)
    I = np.asarray(I, dtype=float)
    dq = np.mean(np.diff(q))
    if not np.allclose(np.diff(q), dq, rtol=1e-3, atol=1e-9):
        # 不等距：逐点核
        out = np.empty_like(I)
        for i in range(I.size):
            kern = np.exp(-0.5 * ((q - q[i]) / sigma_res) ** 2)
            kern /= (kern.sum() + 1e-300)
            out[i] = np.sum(I * kern)
        return out
    half = int(np.ceil(5.0 * sigma_res / max(dq, 1e-30)))
    offs = np.arange(-half, half + 1)
    kern = np.exp(-0.5 * (offs * dq / sigma_res) ** 2)
    kern /= (kern.sum() + 1e-300)
    pad = half
    Ipad = np.pad(I, pad, mode="reflect")
    conv = np.convolve(Ipad, kern, mode="same")
    return conv[pad:-pad]


def _resolution_peak(q, width, exponent):
    """GISAXS-Fit 风格的分辨率峰：1 / (1 + (|q|/width)^exponent)。"""
    q_arr = np.abs(np.asarray(q, dtype=float))
    width = float(width)
    exponent = float(exponent)
    if width <= 0 or exponent <= 0:
        return np.ones_like(q_arr, dtype=float)
    denom = np.maximum(width, 1e-30)
    ratio = q_arr / denom
    return 1.0 / (1.0 + np.power(ratio, exponent))


def _resolution_component(q, width, exponent, intensity):
    """返回 Int * resolution_peak，若强度<=0则为零。"""
    intensity = float(intensity)
    if not np.isfinite(intensity) or intensity == 0:
        return np.zeros_like(np.asarray(q, dtype=float))
    return intensity * _resolution_peak(q, width, exponent)

# =========================
#  Form factors
# =========================
def _F_sphere_amp(q, R):
    """归一化球振幅 F -> 3 (sin x - x cos x)/x^3, x=qR；F(0)=1。"""
    x = np.asarray(q, dtype=float) * R
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    xs = x[small]
    out[small] = 1.0 - xs**2/10.0 + xs**4/280.0
    xc = x[~small]
    out[~small] = 3.0 * (np.sin(xc) - xc * np.cos(xc)) / (xc**3)
    return out

def sphere_form_factor_pd(q, R, sigma_R, n_samples=25, nsig=4.0):
    """P(q)=<F^2> over 高斯 R 分布（截断到 R>=0）。"""
    q = np.asarray(q, dtype=float)
    Rs, W = _gaussian_grid(R, sigma_R, nsig=nsig, n=n_samples, clip_min=0.0)
    F2 = np.zeros_like(q, dtype=float)
    for Ri, wi in zip(Rs, W):
        Fi = _F_sphere_amp(q, Ri)
        F2 += wi * (Fi * Fi)
    return F2

# ---- Cylinder (随机取向各向平均) ----
def _F_cylinder_amp(q, R, h, alpha):
    """
    归一化圆柱振幅（均质圆柱）：
      F(q;R,h,alpha) = [2 J1(qR sinα)/(qR sinα)] * sinc(q h cosα / 2)
    其中 sinc(x)=sin(x)/x；F(0)=1
    alpha 为圆柱轴与 q 的夹角。
    """
    q = np.asarray(q, dtype=float)
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    qr = q * R * sa
    qz = q * h * ca / 2.0

    # 2*J1(x)/x
    # 用近似：J1(x) ~ x/2 - x^3/16 + ...
    out = np.empty_like(q, dtype=float)
    small_r = np.abs(qr) < 1e-6
    xs = qr[small_r]
    j1_over_x = 0.5 - xs**2/16.0 + xs**4/384.0
    fr_small = 2.0 * j1_over_x
    fr = np.empty_like(q, dtype=float)
    fr[small_r] = fr_small
    xr = qr[~small_r]
    # 用 numpy 的近似：J1(x)≈sin(x)/x^2 - cos(x)/x
    j1 = np.sin(xr) / (xr**2) - np.cos(xr) / xr
    fr[~small_r] = 2.0 * j1 / xr

    # sinc(qz)
    small_z = np.abs(qz) < 1e-6
    sinc = np.empty_like(q, dtype=float)
    xz = qz[small_z]
    sinc[small_z] = 1.0 - xz**2/6.0 + xz**4/120.0
    xz2 = qz[~small_z]
    sinc[~small_z] = np.sin(xz2) / xz2

    return fr * sinc

def cylinder_form_factor_pd(
    q, R, sigma_R, h, sigma_h,
    n_R=13, n_h=13, nsig=4.0,
    n_orient=24
):
    """
    P(q)= < F^2(q;R,h,alpha) >_{R,h,alpha}
    - R ~ N(R, sigma_R), h ~ N(h, sigma_h), 截断为正并归一化
    - alpha 取 0..pi/2，随机取向权重 ~ sin(alpha)；用 Gauss-Legendre 或均匀采样近似
    """
    q = np.asarray(q, dtype=float)

    # 尺寸分布
    Rs, WR = _gaussian_grid(R, sigma_R, nsig=nsig, n=n_R, clip_min=0.0)
    Hs, WH = _gaussian_grid(h, sigma_h, nsig=nsig, n=n_h, clip_min=0.0)

    # 取向权重：alpha in [0, pi/2]，权重 ~ sin(alpha)
    # 用简单均匀 alpha 采样并乘 sin(alpha) 做权
    alphas = np.linspace(0.0, np.pi/2, int(n_orient))
    WA = np.sin(alphas)
    WA /= (WA.sum() + 1e-300)

    P = np.zeros_like(q, dtype=float)
    for Ri, wi in zip(Rs, WR):
        for hi, wh in zip(Hs, WH):
            w_size = wi * wh
            # 各向平均
            F2_avg = np.zeros_like(q, dtype=float)
            for a, wa in zip(alphas, WA):
                Fi = _F_cylinder_amp(q, Ri, hi, a)
                F2_avg += wa * (Fi * Fi)
            P += w_size * F2_avg
    return P

# =========================
#  Structure factor (1D)
# =========================
def structure_factor_1d(q, D, sigma_D):
    """
    S(q) = (1 - phi^2) / (1 + phi^2 - 2*phi*cos(q D)),
    phi = exp(-pi * q^2 * sigma_D^2)
    """
    q = np.asarray(q, dtype=float)
    phi = np.exp(-np.pi * (q**2) * (sigma_D**2))
    num = 1.0 - phi**2
    den = 1.0 + phi**2 - 2.0 * phi * np.cos(q * D)
    den = np.where(np.abs(den) < 1e-15, 1e-15, den)
    return num / den

# =========================
#  Model factory
# =========================
def params_template(spec):
    """
    给定 spec（如 ["sphere"], ["sphere","cylinder"], ...）返回参数名列表。
    每个组分的顺序：
      Sphere:   Int, R, sigma_R, D, sigma_D, BG
      Cylinder: Int, R, sigma_R, h, sigma_h, D, sigma_D, BG
    结尾统一追加：sigma_Res(=Br), nu_Res(=Nu), int_Res, k
    """
    names = []
    for i, shape in enumerate(spec, 1):
        if shape.lower() == "sphere":
            names += [f"Int{i}", f"R{i}", f"sigma_R{i}", f"D{i}", f"sigma_D{i}", f"BG{i}"]
        elif shape.lower() == "cylinder":
            names += [f"Int{i}", f"R{i}", f"sigma_R{i}", f"h{i}", f"sigma_h{i}", f"D{i}", f"sigma_D{i}", f"BG{i}"]
        else:
            raise ValueError(f"Unsupported shape: {shape}")
    names += ["sigma_Res", "nu_Res", "int_Res", "k"]
    return names

def make_mixed_model(
    spec,
    *,
    pd_opts=None,        # 尺寸分布采样设置 dict：{ 'nsig':4.0, 'n_samples':... }
    cyl_opts=None        # 圆柱各向平均与分布设置 dict：{ 'n_R':31,'n_h':31,'n_orient':50,'nsig':4.0 }
):
    """
    根据 spec(list[str]) 生成 f(q,*params)：
    I(q) = sum_i Int_i * P_i(q) * S_i(q; D_i, sigma_Di) + BG + I_res * R(q; Br, Nu)

    spec 例子：
      ["sphere"]
      ["sphere", "cylinder"]
      ["sphere", "sphere", "cylinder"]

    返回的函数签名固定：f(q, *params)，参数顺序见 params_template(spec)。
    """
    spec = [s.lower() for s in spec]
    if not (1 <= len(spec) <= 3):
        raise ValueError("spec 长度必须是 1~3。")

    # 采样配置
    pd_defaults = dict(nsig=4.0, n_samples=25)
    if pd_opts:
        pd_defaults.update(pd_opts)

    cyl_defaults = dict(n_R=13, n_h=13, nsig=4.0, n_orient=24)
    if cyl_opts:
        cyl_defaults.update(cyl_opts)

    # 解析函数：从 params 中依 spec 依次取出每个组分的参数
    template = params_template(spec)

    def f(q, *params):
        if len(params) != len(template):
            raise ValueError(f"参数数量不匹配。需要 {len(template)} 个：{template}，实际 {len(params)} 个。")
        q_arr = np.asarray(q, dtype=float)
        idx = 0
        I_mix = np.zeros_like(q_arr, dtype=float)
        BG_total = np.zeros_like(q_arr, dtype=float)

        for i, shape in enumerate(spec, 1):
            if shape == "sphere":
                Int = params[idx]; R = params[idx+1]; sR = params[idx+2]; D = params[idx+3]; sD = params[idx+4]; BG = params[idx+5]
                idx += 6
                P = sphere_form_factor_pd(q_arr, R, sR,
                                          n_samples=int(pd_defaults["n_samples"]),
                                          nsig=pd_defaults["nsig"])
                
                # 如果 D 或 sigma_D 为 0，不应用结构因子
                if D == 0 or sD == 0:
                    S = np.ones_like(q_arr, dtype=float)  # S = 1，无结构因子
                else:
                    S = structure_factor_1d(q_arr, D, sD)
                
                # 每个particle的贡献：Int加权的形状因子和背景
                I_mix += Int * P * S
                BG_total += Int * BG  # BG也按Int加权，Int=0时BG不贡献

            elif shape == "cylinder":
                Int = params[idx]; R = params[idx+1]; sR = params[idx+2]; h = params[idx+3]; sh = params[idx+4]; D = params[idx+5]; sD = params[idx+6]; BG = params[idx+7]
                idx += 8
                P = cylinder_form_factor_pd(
                    q_arr, R, sR, h, sh,
                    n_R=int(cyl_defaults["n_R"]), n_h=int(cyl_defaults["n_h"]),
                    nsig=cyl_defaults["nsig"], n_orient=int(cyl_defaults["n_orient"])
                )
                
                # 如果 D 或 sigma_D 为 0，不应用结构因子
                if D == 0 or sD == 0:
                    S = np.ones_like(q_arr, dtype=float)  # S = 1，无结构因子
                else:
                    S = structure_factor_1d(q_arr, D, sD)
                
                # 每个particle的贡献：Int加权的形状因子和背景
                I_mix += Int * P * S
                BG_total += Int * BG  # BG也按Int加权，Int=0时BG不贡献

            else:
                raise ValueError(f"Unsupported shape: {shape}")

        sigma_Res = params[idx]; nu_Res = params[idx+1]; int_Res = params[idx+2]; k = params[idx+3]
        resolution = _resolution_component(q_arr, sigma_Res, nu_Res, int_Res)
        I_total = I_mix + BG_total + resolution
        return apply_scaling_factor(I_total, k)

    # 给外部看得到的参数名（便于 GUI 提示）
    f.param_names = template  # type: ignore[attr-defined]
    return f


def mixed_model_components(
        spec,
        q,
        params,
        *,
        pd_opts=None,
        cyl_opts=None,
        scale_resolution_to_total=True,
    ):
        """
        给定 spec、q、完整参数列表 params（顺序同 params_template(spec)），返回：
          {
            'particles': [ {'shape': 'sphere'|'cylinder', 'index': i, 'I': np.ndarray}, ... ],
            'BG_total': np.ndarray,
            'resolution': np.ndarray
          }

        约定：
        - 每个粒子的曲线为 Int_i * P_i(q) * S_i(q; D_i, sigma_Di)，若 D 或 sigma_D 为 0 则 S=1。
        - BG_total = sum_i Int_i * BG_i（常数项，按 Int 加权）。
                - 分辨率曲线使用 GISAXS-Fit 形式：I_res * [1 / (1 + (|q|/Br)^Nu)]。
                - 最终所有曲线都乘以全局缩放因子 k。
                - scale_resolution_to_total 参数保留以保持签名兼容，目前不再调整分辨率幅度。
        """
        spec = [s.lower() for s in spec]
        q_arr = np.asarray(q, dtype=float)
        # 采样配置
        pd_defaults = dict(nsig=4.0, n_samples=25)
        if pd_opts:
            pd_defaults.update(pd_opts)
        cyl_defaults = dict(n_R=13, n_h=13, nsig=4.0, n_orient=24)
        if cyl_opts:
            cyl_defaults.update(cyl_opts)

        # 解析参数
        template = params_template(spec)
        if len(params) != len(template):
            raise ValueError(f"参数数量不匹配：需要 {len(template)}，给定 {len(params)}")

        parts = []
        idx = 0
        BG_total = np.zeros_like(q_arr, dtype=float)

        # 粒子贡献（未展宽）
        raw_particles = []
        for i, shape in enumerate(spec, 1):
            if shape == "sphere":
                Int = params[idx]; R = params[idx+1]; sR = params[idx+2]; D = params[idx+3]; sD = params[idx+4]; BG = params[idx+5]
                idx += 6
                P = sphere_form_factor_pd(q_arr, R, sR, n_samples=int(pd_defaults["n_samples"]), nsig=pd_defaults["nsig"])
                S = np.ones_like(q_arr, dtype=float) if (D == 0 or sD == 0) else structure_factor_1d(q_arr, D, sD)
                I_part = float(Int) * P * S
                BG_total = BG_total + float(Int) * float(BG)
                raw_particles.append(("sphere", i, I_part))
            elif shape == "cylinder":
                Int = params[idx]; R = params[idx+1]; sR = params[idx+2]; h = params[idx+3]; sh = params[idx+4]; D = params[idx+5]; sD = params[idx+6]; BG = params[idx+7]
                idx += 8
                P = cylinder_form_factor_pd(
                    q_arr, R, sR, h, sh,
                    n_R=int(cyl_defaults["n_R"]), n_h=int(cyl_defaults["n_h"]), nsig=cyl_defaults["nsig"], n_orient=int(cyl_defaults["n_orient"]) 
                )
                S = np.ones_like(q_arr, dtype=float) if (D == 0 or sD == 0) else structure_factor_1d(q_arr, D, sD)
                I_part = float(Int) * P * S
                BG_total = BG_total + float(Int) * float(BG)
                raw_particles.append(("cylinder", i, I_part))
            else:
                raise ValueError(f"Unsupported shape: {shape}")

        # 取出分辨率与缩放因子
        sigma_Res = float(params[idx]); nu_Res = float(params[idx+1]); int_Res = float(params[idx+2]); k = float(params[idx+3])

        # 组件曲线直接乘以 k（新分辨率模型无需再对粒子做高斯展宽）
        for shape, i, I_part in raw_particles:
            parts.append({"shape": shape, "index": i, "I": apply_scaling_factor(I_part, k)})

        BG_curve = apply_scaling_factor(BG_total, k)

        # 分辨率曲线
        res_curve_raw = _resolution_component(q_arr, sigma_Res, nu_Res, int_Res)
        res_curve = apply_scaling_factor(res_curve_raw, k)

        return {
            'particles': parts,      # list of {shape,index,I}
            'BG_total': BG_curve,    # ndarray
            'resolution': res_curve, # ndarray
        }
