# -*- coding: utf-8 -*-
"""
sdplane-sim.py

功能：
本脚本使用蒙特卡洛光线追踪方法，模拟并计算一个由点光源、一个理想漫反射平面
和一个理想镜面反射平面组成的场景中，一个圆形接收器所接收到的光通量（Flux）。
该模拟可以分析在不同的平面夹角（α, alpha）和接收器旋转角度（θ, theta）下，
光通量的变化情况。

核心技术：
- 镜面反射: 采用镜像光源法（Image Source Method）进行高效计算。
- 漫反射: 采用多重重要性采样（Multiple Importance Sampling, MIS）结合光源采样与
           接收器采样两种策略，以降低蒙特卡洛积分的方差。
- 几何与遮挡: 实现了精确的射线-平面相交测试，并能在有限大小的平面上进行边界判断。
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------
# 1. 参数定义
# --------------------------
# 物理参数
P = 1000.0      # 光源总功率 (W), 假设其为各向同性点光源
rho_d = 0.8     # 漫反射率 (dimensionless, 0 to 1)
rho_s = 0.95    # 镜面反射率 (dimensionless, 0 to 1)

# 几何参数
z1 = 2.0        # 接收器（眼睛）在初始位置 (θ=0) 处的高度 (m)
z2 = 2.5        # 光源在初始位置 (θ=0) 处的高度 (m)
R_eye = 0.05    # 接收器（眼睛）的半径 (m)
A_eye = np.pi * R_eye**2 # 接收器的面积 (m^2)

# 模型约束
Lx, Ly = 50.0, 50.0 # 平面有限尺寸 (m)。
                    # 漫反射面定义域为 x∈[0,Lx], y∈[-Ly,Ly]。
                    # 镜面是一个倾斜的矩形，其在x-y平面的投影宽度为Lx。

# 数值精度相关的微小量
EPS = 1e-9

# --------------------------
# 2. 几何与辅助函数
# --------------------------
def get_geometry(alpha_deg, theta_deg):
    """
    根据给定的平面夹角(alpha)和观测者旋转角度(theta)，计算场景的几何状态。
    场景本身是固定的，通过旋转光源、接收器和接收器法线来模拟观测者视角的改变。
    
    Args:
        alpha_deg (float): 镜面与漫反射面形成的二面角 (degree). 90度表示垂直，180度表示共面。
        theta_deg (float): 观测者（光源S和接收器E）绕X轴的旋转角度 (degree).

    Returns:
        tuple: 包含场景几何信息的元组 (法线, 位置向量等)。
    """
    alpha = np.radians(alpha_deg)
    theta = np.radians(theta_deg)
    
    # 漫反射面 (平坦, z=0)，法线朝向+z
    n_diffuse = np.array([0.0, 0.0, 1.0])
    # 镜面 (绕y轴倾斜)，法线在x-z平面内
    n_specular = np.array([np.sin(alpha), 0.0, -np.cos(alpha)])
    
    # FIX for Defect 1: 为镜面创建一个局部坐标系的基向量 (u_spec, v_spec)。
    # 这对于后续在倾斜的镜面上进行有限区域判断至关重要。
    v_spec = np.array([0.0, 1.0, 0.0])      # v_spec 沿着全局 y 轴
    u_spec = np.cross(v_spec, n_specular)  # u_spec 在镜面内且与 v_spec 正交
    
    # 计算绕x轴的旋转矩阵，用于旋转观测者（光源和接收器）
    c, s = np.cos(-theta), np.sin(-theta)
    Rx_neg = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    
    # S, E, n_eye 的初始位置在z轴上，随 theta 旋转
    S = Rx_neg @ np.array([0.0, 0.0, z2])      # 光源 S 的位置
    E = Rx_neg @ np.array([0.0, 0.0, z1])      # 接收器 E 的中心位置
    n_eye = Rx_neg @ np.array([0.0, 0.0, 1.0]) # 接收器平面的法线方向
    
    return n_diffuse, n_specular, S, E, n_eye, u_spec, v_spec

def make_basis_from_z(z_axis):
    """
    给定一个 Z 轴向量，构建一个标准正交基 (t1, t2, z)。
    这常用于从一个法线方向构建一个局部坐标系，以便进行半球采样。
    """
    z = z_axis / (np.linalg.norm(z_axis) + EPS) 
    # 选择一个与z不共线的参考向量来生成切线
    ref = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(ref, z)
    t1 /= (np.linalg.norm(t1) + EPS)
    t2 = np.cross(z, t1)
    return t1, t2, z

def is_on_finite_plane(M, is_specular, alpha_rad=None, u_spec=None, v_spec=None):
    """
    检查一个三维点 M 是否在指定的有限平面区域内。
    """
    if is_specular:
        # 对于镜面，在其局部坐标系 (u_spec, v_spec) 下判断。
        # 镜面在 x-y 平面的投影宽度为 Lx，其真实宽度需要根据倾斜角进行校正。
        spec_width = Lx / (abs(np.cos(alpha_rad)) + EPS)
        # 将点 M 投影到局部坐标轴上
        u_coord = np.dot(M, u_spec)
        v_coord = np.dot(M, v_spec)
        # 检查坐标是否在镜面矩形区域内
        return -spec_width < u_coord < 0 and -Ly < v_coord < Ly
    else:
        # 对于漫反射面，直接根据其在全局坐标系下的定义域判断。
        return 0.0 < M[0] < Lx and -Ly < M[1] < Ly

def is_path_occluded(P, M, blocker_normal, is_blocker_specular, alpha_rad=None, u_spec_blocker=None, v_spec_blocker=None):
    """
    测试从点 P 到点 M 的路径是否被一个有限的“遮挡体”平面阻挡。
    """
    seg_dir = M - P # 从 P 指向 M 的向量
    denom = np.dot(seg_dir, blocker_normal)
    
    # 检查射线是否与遮挡体的无限平面平行
    if abs(denom) > EPS:
        # 遮挡体平面方程为 n . X = 0 (因为两个平面都过原点)
        # 计算射线 P + s*seg_dir 与平面的交点参数 s
        s_param = -np.dot(P, blocker_normal) / denom
        # 如果交点在 P 和 M 之间 (0 < s < 1)，则可能存在遮挡
        if EPS < s_param < 1.0 - EPS:
            # 计算交点 I
            I = P + s_param * seg_dir
            # 检查交点 I 是否在遮挡体的“有限区域”内
            if is_on_finite_plane(I, is_blocker_specular, alpha_rad, u_spec_blocker, v_spec_blocker):
                return True # 路径被遮挡
    return False # 路径未被遮挡

def stratified_samples(n):
    """
    生成 n 个分层样本点 (u1, u2) 在 [0,1)x[0,1) 区间内。
    分层采样是一种方差缩减技术，比纯随机采样效果更好。
    """
    n_sqrt = int(np.sqrt(n))
    n = n_sqrt * n_sqrt # 确保样本数是平方数以便于分层
    # 为每个层随机排序
    u1_idx = np.random.permutation(np.arange(n))
    u2_idx = np.random.permutation(np.arange(n))
    # 在每个子格内生成一个随机样本
    u1 = (u1_idx + np.random.rand(n)) / n
    u2 = (u2_idx + np.random.rand(n)) / n
    return u1, u2, n

# --------------------------
# 3. 光通量计算函数
# --------------------------
def solve_specular(alpha_deg, theta_deg, N=10000):
    """
    使用镜像光源法计算从镜面反射到接收器的光通量。
    """
    if abs(theta_deg) >= 90: return 0.0 # 接收器转到平面下方，收不到光
    alpha_rad = np.radians(alpha_deg)
    n_diffuse, n_specular, S, E, n_eye, u_spec, v_spec = get_geometry(alpha_deg, theta_deg)
    
    # 1. 计算光源 S 关于镜面的镜像点 S_prime
    S_prime = S - 2.0 * np.dot(S, n_specular) * n_specular
    
    # 2. 在接收器圆盘上进行蒙特卡洛积分
    u1, u2, N = stratified_samples(N)
    r = R_eye * np.sqrt(u1) # 使用 sqrt(u1) 进行均匀圆盘采样
    phi = 2.0 * np.pi * u2
    t1, t2, _ = make_basis_from_z(n_eye) # 创建接收器平面的局部基
    # 生成接收器圆盘上的 N 个采样点
    E_points = E + np.outer(r * np.cos(phi), t1) + np.outer(r * np.sin(phi), t2)
    
    integrand_sum = 0.0
    for ep in E_points:
        # 3. 连接镜像点 S_prime 和接收器上的点 ep，找到与镜面的交点 M
        line_dir = ep - S_prime
        denom = np.dot(line_dir, n_specular)
        if abs(denom) < EPS: continue # 射线与镜面平行
        
        t_intersect = -np.dot(S_prime, n_specular) / denom
        if t_intersect <= EPS: continue # 交点在射线反方向
        M = S_prime + t_intersect * line_dir
        
        # 4. 可见性测试
        # 检查交点 M 是否在有限的镜面区域内
        if not is_on_finite_plane(M, True, alpha_rad, u_spec, v_spec):
            continue
            
        # 检查真实光路 S->M 是否被漫反射面遮挡
        if is_path_occluded(S, M, n_diffuse, False):
            continue
        
        # 5. 计算光照贡献
        d = ep - S_prime
        d_norm_sq = d.dot(d)
        if d_norm_sq < EPS: continue
        cos_psi_prime = np.dot(d, n_eye) / np.sqrt(d_norm_sq) # 接收器法线与光线的夹角余弦
        if cos_psi_prime <= EPS: continue
        
        # 累加辐射度积分项 (cos(psi') / |ep - S_prime|^2)
        integrand_sum += cos_psi_prime / d_norm_sq
        
    # 乘以常数项得到总光通量
    # P/(4*pi) 是点光源在单位立体角上的光强
    return (rho_s * P / (4.0 * np.pi)) * A_eye * (integrand_sum / N)

def solve_diffuse_mis(alpha_deg, theta_deg, N=20000):
    """
    使用多重重要性采样（MIS）计算从漫反射面到接收器的光通量。
    """
    if abs(theta_deg) >= 90: return 0.0
    alpha_rad = np.radians(alpha_deg)
    n_diffuse, n_specular, S, E, n_eye, u_spec, v_spec = get_geometry(alpha_deg, theta_deg)
    
    # 该函数估算的是辐照度(Irradiance)，最后需要乘以接收器面积 A_eye 得到光通量(Flux)
    total_irradiance_estimator = 0.0

    # --- 策略1: 从光源采样 (Light Sampling) ---
    # 从光源 S 发出光线，击中漫反射面上的点 M，再计算 M 到接收器 E 的贡献
    N_light = N // 2
    light_aim_dir = -n_diffuse # 光源应朝向漫反射面进行采样
    t1_l, t2_l, b1_l = make_basis_from_z(light_aim_dir)
    u1_l, u2_l, N_light = stratified_samples(N_light)

    for i in range(N_light):
        # 按余弦分布在半球上采样一个方向 d
        cos_theta_local = np.sqrt(u1_l[i])
        sin_theta_local = np.sqrt(max(0.0, 1.0 - u1_l[i]))
        phi_local = 2.0 * np.pi * u2_l[i]
        d = t1_l * (sin_theta_local*np.cos(phi_local)) + t2_l * (sin_theta_local*np.sin(phi_local)) + b1_l * cos_theta_local
        
        # 计算光线与漫反射面的交点 M
        denom_d = np.dot(d, n_diffuse)
        if abs(denom_d) < EPS: continue
        t_plane = -np.dot(S, n_diffuse) / denom_d
        if t_plane <= 0: continue
        M = S + t_plane*d

        # 可见性测试
        if not is_on_finite_plane(M, False): continue
        if is_path_occluded(S, M, n_specular, True, alpha_rad, u_spec, v_spec): continue
        if is_path_occluded(M, E, n_specular, True, alpha_rad, u_spec, v_spec): continue

        # 计算几何项
        L, V = S-M, E-M
        L_norm_sq, V_norm_sq = max(L.dot(L), EPS), max(V.dot(V), EPS)
        L_norm, V_norm = np.sqrt(L_norm_sq), np.sqrt(V_norm_sq)
        cos_alpha_S = np.dot(L, n_diffuse)/L_norm # 光源方向与法线夹角
        cos_alpha_E = np.dot(V, n_diffuse)/V_norm # 接收器方向与法线夹角
        cos_psi = np.dot(V, n_eye)/V_norm       # 接收器法线与来自M的光线夹角
        if cos_alpha_S <= 0 or cos_alpha_E <= 0 or cos_psi <= 0: continue
        
        # 渲染方程的被积函数（针对辐照度）
        irradiance_integrand = (rho_d*P/np.pi) * (cos_alpha_S/(4*np.pi*L_norm_sq)) * (cos_alpha_E*cos_psi/V_norm_sq)
        
        # 计算此样本在两种策略下的概率密度函数 (PDF)
        pdf_light = (cos_alpha_S**2) / (np.pi * L_norm_sq + EPS)
        pdf_camera = (cos_psi * cos_alpha_E) / (np.pi * V_norm_sq + EPS)
        # 根据平衡启发式计算 MIS 权重
        weight_mis = pdf_light / (pdf_light + pdf_camera + EPS)
        
        if pdf_light > EPS:
            total_irradiance_estimator += irradiance_integrand / pdf_light * weight_mis

    # --- 策略2: 从接收器采样 (Camera/Receiver Sampling) ---
    # 从接收器 E 发出光线，击中漫反射面上的点 M，再计算从 S 到 M 的贡献
    N_camera = N - N_light
    cam_aim_dir = n_eye
    t1_c, t2_c, b1_c = make_basis_from_z(cam_aim_dir)
    u1_c, u2_c, N_camera = stratified_samples(N_camera)
    for i in range(N_camera):
        # 按余弦分布在半球上采样一个方向 d
        cos_theta_local = np.sqrt(u1_c[i])
        phi_local = 2.0 * np.pi * u2_c[i]
        sin_theta_local = np.sqrt(max(0.0, 1.0 - u1_c[i]))
        d = t1_c * (sin_theta_local*np.cos(phi_local)) + t2_c * (sin_theta_local*np.sin(phi_local)) + b1_c * cos_theta_local

        # 计算光线与漫反射面的交点 M
        denom_d = np.dot(d, n_diffuse)
        if abs(denom_d) < EPS: continue
        t_plane = -np.dot(E, n_diffuse) / denom_d
        if t_plane <= 0: continue
        M = E + t_plane*d
        
        # 可见性测试
        if not is_on_finite_plane(M, False): continue
        if is_path_occluded(S, M, n_specular, True, alpha_rad, u_spec, v_spec): continue
        # if is_path_occluded(M, E, n_specular, True, alpha_rad, u_spec, v_spec): continue # 此路径已被采样，无需重复测试

        # 计算几何项
        L, V = S-M, E-M
        L_norm_sq, V_norm_sq = max(L.dot(L),EPS), max(V.dot(V),EPS)
        L_norm, V_norm = np.sqrt(L_norm_sq), np.sqrt(V_norm_sq)
        cos_alpha_S = np.dot(L, n_diffuse) / L_norm
        cos_alpha_E = np.dot(-V, n_diffuse) / V_norm # V 是 E-M, M->E 的向量是 -V
        cos_psi = np.dot(-V, n_eye) / V_norm         # 同上
        if cos_alpha_S <= 0 or cos_alpha_E <= 0 or cos_psi <= 0: continue

        # 渲染方程的被积函数（与策略1相同）
        irradiance_integrand = (rho_d*P/np.pi) * (cos_alpha_S/(4*np.pi*L_norm_sq)) * (cos_alpha_E*cos_psi/V_norm_sq)
        
        # 计算 PDF（与策略1相同）
        pdf_light = (cos_alpha_S**2) / (np.pi * L_norm_sq + EPS)
        pdf_camera = (cos_psi * cos_alpha_E) / (np.pi * V_norm_sq + EPS)
        # 根据平衡启发式计算 MIS 权重
        weight_mis = pdf_camera / (pdf_light + pdf_camera + EPS)
        
        if pdf_camera > EPS:
            total_irradiance_estimator += irradiance_integrand / pdf_camera * weight_mis
    
    # 将估算的平均辐照度乘以接收器面积，得到总光通量
    return (total_irradiance_estimator / N) * A_eye if N > 0 else 0.0

# --------------------------
# 4. 可视化驱动
# --------------------------
def visualize_results(thetas, alphas):
    """
    运行模拟并绘制结果曲线图。
    """
    # case 1: 固定 alpha, 变化 theta
    fixed_alpha = 135.0
    print(f"Calculating Final Corrected Flux vs. Theta (alpha={fixed_alpha})...")
    flux_vs_theta_s = [solve_specular(fixed_alpha, t, N=20000) for t in tqdm(thetas)]
    flux_vs_theta_d = [solve_diffuse_mis(fixed_alpha, t, N=40000) for t in tqdm(thetas)] # 漫反射通常需要更多采样
    flux_vs_theta_total = np.array(flux_vs_theta_s) + np.array(flux_vs_theta_d)

    # case 2: 固定 theta, 变化 alpha
    fixed_theta = 0.0
    print(f"Calculating Final Corrected Flux vs. Alpha (theta={fixed_theta})...")
    flux_vs_alpha_s = [solve_specular(a, fixed_theta, N=20000) for a in tqdm(alphas)]
    flux_vs_alpha_d = [solve_diffuse_mis(a, fixed_theta, N=40000) for a in tqdm(alphas)]
    flux_vs_alpha_total = np.array(flux_vs_alpha_s) + np.array(flux_vs_alpha_d)

    # 绘图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

    # 图1: Flux vs. Theta
    ax1.plot(thetas, flux_vs_theta_total, '-o', markersize=4, label='Total', color='k')
    ax1.plot(thetas, flux_vs_theta_d, '--', label='Diffuse', color='b')
    ax1.plot(thetas, flux_vs_theta_s, '--', label='Specular', color='r')
    ax1.set_xlabel('Rotation Angle θ (deg)')
    ax1.set_ylabel('Flux (W)')
    ax1.set_title(f'Flux vs θ (α={fixed_alpha}°)')
    ax1.legend()
    ax1.grid(True)

    # 图2: Flux vs. Alpha
    ax2.plot(alphas, flux_vs_alpha_total, '-o', markersize=4, label='Total', color='k')
    ax2.plot(alphas, flux_vs_alpha_d, '--', label='Diffuse', color='b')
    ax2.plot(alphas, flux_vs_alpha_s, '--', label='Specular', color='r')
    ax2.set_xlabel('Dihedral Angle α (deg)')
    ax2.set_ylabel('Flux (W)')
    ax2.set_title(f'Flux vs α (θ={fixed_theta}°)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 定义需要扫描的角度范围和步长
    theta_range = np.linspace(-90, 90, 61)
    alpha_range = np.linspace(95, 175, 35)
    visualize_results(theta_range, alpha_range)