# Computer Assignment - 2024s
# PETE 332: Petroleum Production Engineering II
# Department of Petroleum and Natural Gas Engineering
# Middle East Technical University
# Prepared by:
# Mert Yücel
# Arda Çimen
# Ege Filizli
# 05/06/2025

"""
MIT License

Copyright (c) 2025 Mert Yücel, Arda Çimen, Ege Filizli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

ComputerAssignment = cv2.imread("pete332_2024s_readme.jpeg", cv2.IMREAD_COLOR)
cv2.imshow("Computer Assignment - 2024s", ComputerAssignment)

cv2.waitKey()
cv2.destroyAllWindows()

student_id = str(input("Enter your student ID: "))
list_id = []
for i in range(len(student_id)):
    list_id.append(int(student_id[i]))


""" Common Input Data """
p_res = 2600  # psia
perf_depth = 5000  # ft
tubing_length = perf_depth  # ft
target_oil_prod_rate = 500 + list_id[-2] * 80  # stb/d
res_liq_pi = 2.0  # stb/d/psia
prod_glr = 100  # scf/stb
wc = .2
api_G = 25  # API
water_SG = 1.05
gas_SG = .65
air_SG = 1
water_FVF = 1.02  # rb/stb

""" Gas Injection Input Data """
gas_inj_valve_depth = 4200  # ft
gas_inj_flow_rate = 10 + list_id[-1] * 100  # Mscf/d
csg_ID = 4.55  # in
rel_roughness = .0041
gas_temp_at_surf = 60  # F
grad_T_ann = 14/1000  # F/ft
p_drop = 100  # psia

""" Production Tubing Input Data """
prod_tubing_length = tubing_length  # ft
tubing_ID = 2.347  # in
tubing_OD = 2.875  # in
T_bh = 150  # F
grad_T_prod_tubing = 8/1000  # F/ft

p_bhf = p_res - target_oil_prod_rate / res_liq_pi  # psia
oil_SG = 141.5 / (131.5 + api_G)
number_of_segments = 25
change_in_h = perf_depth / number_of_segments  # ft
q_liq = gas_inj_flow_rate * 1000 / prod_glr
q_water = q_liq - target_oil_prod_rate
GOR = gas_inj_flow_rate * 1000 / target_oil_prod_rate
density_of_air = .0765  # lb/cuft

p_pc = 677 + 15 * gas_SG - 37.5 * (gas_SG ** 2)  # psi
T_pc = 168 + 325 * gas_SG - 12.5 * (gas_SG ** 2)  # R

def plot_section(p_list_tubing, p_list_ann, depth_list_tubing, depth_list_ann, list_p_inj, inj_depth, list_rho_av, list_rs):
    plt.figure(figsize=(10, 6))
    plt.plot(p_list_tubing, depth_list_tubing, label="Tubing Pressure Profile", linewidth=1, marker=".", mec="r", mfc="r")
    plt.plot(p_list_ann, depth_list_ann, label="Annulus Pressure Profile", linewidth=1, marker=".", mec="r", mfc="r")
    plt.plot(list_p_inj, [inj_depth, inj_depth], color= 'k', linestyle='dashed')
    plt.ylim(bottom = 0)
    for xy in zip(p_list_tubing, depth_list_tubing):
        plt.annotate('(%.2f psi, %.2f ft)' % xy, xy=xy, fontsize=5, ha="right", va="center")
    for xy in zip(p_list_ann, depth_list_ann):
        plt.annotate('(%.2f psi, %.2f ft)' % xy, xy=xy, fontsize=5)
    plt.gca().invert_yaxis()
    plt.xlabel("Pressure (psia)")
    plt.ylabel("Depth (ft)")
    plt.title("Pressure Profiles from Bottomhole to Wellhead")
    plt.legend()
    plt.grid(True)
    plt.figtext(0.98, 0.02, '© PETE 332 - METU 2024s - CA | Mert Yücel, Arda Çimen, Ege Filizli', fontsize=8, color='gray', ha='right', alpha=0.5)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(list_rho_av, depth_list_tubing)
    plt.ylim(bottom = 0)
    plt.gca().invert_yaxis()
    plt.xlabel("Average Fluid Density (lbm/cuft)")
    plt.ylabel("Depth (ft)")
    plt.title("Average Fluid Density vs Depth")
    plt.legend()
    plt.grid(True)
    plt.figtext(0.98, 0.02, '© PETE 332 - METU 2024s - CA | Mert Yücel, Arda Çimen, Ege Filizli', fontsize=8, color='gray', ha='right', alpha=0.5)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(list_rs, depth_list_tubing)
    plt.ylim(bottom = 0)
    plt.gca().invert_yaxis()
    plt.xlabel("Dissolved Gas in Oil (scf/stb)")
    plt.ylabel("Depth (ft)")
    plt.title("Dissolved Gas in Oil vs Depth")
    plt.legend()
    plt.grid(True)
    plt.figtext(0.98, 0.02, '© PETE 332 - METU 2024s - CA | Mert Yücel, Arda Çimen, Ege Filizli', fontsize=8, color='gray', ha='right', alpha=0.5)
    plt.show()

def dRHOv(M, q_liq, tubing_id):
    return max((1.4737 * 10 ** -5 * M * (q_liq * 0.9)) / (tubing_id / 12), 1e-6)

def f_2F(d_rho_v):
    return max(14 * 10**(1.444 - 2.5 * np.log10(max(d_rho_v, 1e-6))), 1e-6)

def k_Factor(f_2f, q_liq, M, tubing_id):
    return (f_2f * (q_liq * 0.9)**2 * M ** 2) / (7.4137 * 10 ** 10 * (tubing_id / 12) ** 5)

def RHO_av(oil_specific_gravity, wor, water_specific_gravity, gor, gas_specific_gravity, z, p, t, water_FVF):
    M = 350.17 * (oil_specific_gravity + wor * water_specific_gravity) + gor * gas_specific_gravity * 0.075
    rs = min(gas_specific_gravity * ((p / 18) * (10**(0.0125 * (141.5 / oil_specific_gravity - 131.5)) / (10**(0.00091 * t))))**1.2048, 166.67)
    bo = 0.9759 + 0.00012 * (rs * (gas_specific_gravity / oil_specific_gravity)**0.5 + 1.25 * t)**1.2
    Vm = 5.615 * (bo + wor * water_FVF) + ((gor - rs) * ((14.7 / p) * ((t + 460) / 520) * z))
    return M / Vm, M, rs, bo, Vm

def Z_Factor(p_r, t_r):
    A = 1.39 * (t_r - 0.92)**0.5 - 0.36 * t_r - 0.101
    B = (0.62 - 0.23 * t_r) * p_r + (0.066 / (t_r - 0.86) - 0.037) * p_r**2 + 0.32 * p_r**6 / (10**(9 * (t_r - 1)))
    C = 0.132 - 0.32 * np.log10(t_r)
    D = 10**(0.3106 - 0.49 * t_r + 0.1824 * t_r**2)

    try:
        z = A + (1 - A) / np.exp(B) + C * (p_r**D)
    except (OverflowError, ValueError):
        z = A + (1 - A) / np.exp(B)

    return z
def p_drop_comp(tubing_length, segments, p_bhf, q_liq, wc, oil_SG, water_SG, gas_SG, grad_T, bottomhole_temperature):
    segment_length = tubing_length / segments
    pressures = [p_bhf]
    depths = [perf_depth]
    p_current = p_bhf
    rho_list = []
    rs_list = []
    pr = p_bhf / p_pc
    tr = (bottomhole_temperature + 460) / T_pc
    z = Z_Factor(pr, tr)
    wor = wc * q_liq / (q_liq * (1 - wc))
    rho_mixture, M, rs, bo, Vm = RHO_av(oil_SG, wor, water_SG, GOR, gas_SG, z, p_bhf, bottomhole_temperature, water_FVF)
    rho_list.append(rho_mixture)
    rs_list.append(rs)
    for i in range(segments):
        depth = depths[0] - (i + 1) * segment_length
        t_segment = bottomhole_temperature - (i + 1) * grad_T * segment_length
        pr = p_current / p_pc
        tr = (t_segment + 460) / T_pc
        z = Z_Factor(pr, tr)
        wor = wc * q_liq / (q_liq * (1 - wc))
        rho_mixture, M, rs, bo, Vm = RHO_av(oil_SG, wor, water_SG, GOR, gas_SG, z, p_current, t_segment, water_FVF)
        rho_list.append(rho_mixture)
        rs_list.append(rs)
        dpv = dRHOv(M, q_liq, tubing_ID)
        f2f = f_2F(dpv)
        k = k_Factor(f2f, q_liq, M, tubing_ID)
        delta_p = ((rho_mixture + (k / rho_mixture)) * segment_length / 144)
        p_current -= delta_p
        pressures.append(p_current)
        depths.append(depth)
    return pressures, depths, rho_list, rs_list



segments = number_of_segments
pressures, depths, rho_list, rs_list = p_drop_comp(tubing_length, segments, p_bhf, q_liq, wc, oil_SG, water_SG, gas_SG, grad_T_prod_tubing, T_bh)
p_inj = pressures[4] + 100
pressures_2 = [p_inj]
depths_2 = [gas_inj_valve_depth]
seg_len = gas_inj_valve_depth / segments
temp0 = gas_temp_at_surf + gas_inj_valve_depth * grad_T_ann
list_temp =[temp0]
MW_air = 28.966  # lb / lb.mol
MW_a = gas_SG * MW_air  # lb / lb.mol
R = 10.732  # (ft^3 * psi) / (Rankine * lb.mol)
for i in range(segments):
    p_estimated = pressures_2[i] - .0001  # psi
    depth = depths_2[0] - (i + 1) * seg_len
    depths_2.append(depth)
    t_seg = gas_temp_at_surf + depth * grad_T_ann
    p_av = (pressures_2[i] + p_estimated) / 2
    t_av = (list_temp[0] + t_seg) / 2
    pr = p_av / p_pc
    tr = (t_seg + 460) / T_pc
    z_av = Z_Factor(pr, tr)
    density_gas = MW_a * p_av / ((t_av + 460) * R * z_av)   # lb / ft^3
    constant_K = (9.4 + 0.02 * MW_a) * ((t_av + 460) ** 1.5) / (209 + 19 * MW_a + (t_av + 460)) * (10 ** (-4))
    constant_X = 3.5 + (986 / (t_av + 460)) + .01 * MW_a
    constant_Y = 2.4 - .2 * constant_X
    viscosity = constant_K * np.exp(constant_X * ((density_gas / 62.4) ** constant_Y))   # cP
    N_Re = 20.1 * gas_inj_flow_rate * gas_SG / ((csg_ID - tubing_OD) * viscosity)
    # A condition with respect to the type of fluid flow, laminar or turbulent
    # If the type of fluid flow is laminar:
    if np.all(N_Re < 2000):
        f_F = 16 / N_Re
    else:
        f_F = (-4 * np.log10(rel_roughness / 3.7065 - 5.0452 / N_Re * np.log10((rel_roughness ** 1.1098) / 2.8257 + (7.149 / N_Re) ** .8981))) ** (-2)
    f_M = f_F * 4
    s = .0375 * gas_SG * gas_inj_valve_depth * 1 / (z_av * (t_av + 460) * segments)

    # To compute exp(s) value
    e_s = np.exp(s)

    # To compute p_2 value
    p_2 = np.sqrt(e_s * np.power(pressures_2[i], 2) + (6.67 * (10 ** (-4)) * (e_s - 1) * f_M * np.power((gas_inj_flow_rate * z_av * (t_av + 460)), 2) / (np.power((csg_ID - tubing_OD), 5) * np.cos(np.radians(0)))))

    while np.any(np.abs(100 * (p_2 - p_estimated) / p_estimated) > .0001):
        p_estimated = p_2
        p_av = (p_2 + p_estimated) / 2
        pr = p_av / p_pc
        tr = (t_seg + 460) / T_pc
        z_av = Z_Factor(pr, tr)
        density_gas = MW_a * p_av / ((t_av + 460) * R * z_av)   # lb / ft^3
        constant_K = (9.4 + 0.02 * MW_a) * ((t_av + 460) ** 1.5) / (209 + 19 * MW_a + (t_av + 460)) * (10 ** (-4))
        constant_X = 3.5 + (986 / (t_av + 460)) + .01 * MW_a
        constant_Y = 2.4 - .2 * constant_X
        viscosity = constant_K * np.exp(constant_X * ((density_gas / 62.4) ** constant_Y))   # cP
        N_Re = 20.1 * gas_inj_flow_rate * gas_SG / ((csg_ID - tubing_OD) * viscosity)
        # A condition with respect to the type of fluid flow, laminar or turbulent
        # If the type of fluid flow is laminar:
        if np.all(N_Re < 2000):
            f_F = 16 / N_Re
        else:
            f_F = (-4 * np.log10(rel_roughness / 3.7065 - 5.0452 / N_Re * np.log10((rel_roughness ** 1.1098) / 2.8257 + (7.149 / N_Re) ** .8981))) ** (-2)
        f_M = f_F * 4
        s = .0375 * gas_SG * gas_inj_valve_depth * 1 / (z_av * (t_av + 460) * segments)

        # To compute exp(s) value
        e_s = np.exp(s)

        # To compute p_2 value
        p_2 = np.sqrt(e_s * np.power(pressures_2[i], 2) + (6.67 * (10 ** (-4)) * (e_s - 1) * f_M * np.power((gas_inj_flow_rate * z_av * (t_av + 460)), 2) / (np.power((csg_ID - tubing_OD), 5) * np.cos(np.radians(0)))))
    diff = p_2 - pressures_2[i]
    p_inv = pressures_2[i] - diff
    pressures_2.append(p_inv)
p_inj_point = [pressures[4], p_inj]
plotting = plot_section(pressures, pressures_2, depths, depths_2, p_inj_point, gas_inj_valve_depth, rho_list, rs_list)




