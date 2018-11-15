import numpy as np
import random

def vecs_angle(vec1, vec2):
    return np.arccos(np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2))

def angle2rad(angle):
    return angle * np.pi / 180

def rad2angle(rad):
    return rad * 180 / np.pi

def forward_kinematics(phi_angle_ary):
    theta_bias_ary = np.array([0, angle2rad(90), angle2rad(-90), 0, 0, 0])
    phi_ary = np.array([angle2rad(e) for e in phi_angle_ary])
    alpha_angle_ary = np.array([0, 90, 0, -90, 90, 90])
    alpha = np.array([angle2rad(e) for e in alpha_angle_ary])
    a = np.array([0, 140, 265, 0, 0, 0])
    # for debug, the third element is 0, should be 20
    d = np.array([320, 0, 20, 343, 0, -156])
    theta = phi_ary + theta_bias_ary

    T_list = []
    for i in range(6):
        T = np.array([[np.cos(theta[i]), -np.sin(theta[i]), 0, a[i]],
                      [np.sin(theta[i]) * np.cos(alpha[i]), np.cos(theta[i]) * np.cos(alpha[i]), -np.sin(alpha[i]), -np.sin(alpha[i]) * d[i]],
                      [np.sin(theta[i]) * np.sin(alpha[i]), np.cos(theta[i]) * np.sin(alpha[i]), np.cos(alpha[i]), np.cos(alpha[i]) * d[i]],
                      [0, 0, 0, 1]])
        T_list.append(T)

    T_list.reverse()
    pt6 = np.array([0, 0, 0, 1])
    for T in T_list:
        pt6 = np.matmul(T, pt6)

    pt5 = np.array([0, 0, 0, 1])
    for T in T_list[1:]:
        pt5 = np.matmul(T, pt5)

    pt4 = np.array([0, 0, 0, 1])
    for T in T_list[2:]:
        pt4 = np.matmul(T, pt4)

    pt3 = np.array([0, 0, 0, 1])
    for T in T_list[3:]:
        pt3 = np.matmul(T, pt3)

    pt2 = np.array([0, 0, 0, 1])
    for T in T_list[4:]:
        pt2 = np.matmul(T, pt2)

    pt1 = np.array([0, 0, 0, 1])
    for T in T_list[5:]:
        pt1 = np.matmul(T, pt1)

    '''
    print("forward points:")
    print(pt1)
    print(pt2)
    print(pt3)
    print(pt4)
    print(pt5)
    print(pt6)
    '''

    dir_vec = pt6 - pt5
    dir_vec = dir_vec / np.linalg.norm(dir_vec)

    return np.hstack([pt6[:3], dir_vec[:3]])

def inverse_kinematics(coord):
    d0 = 320
    d1 = 140
    d2 = 265
    d3 = 20
    d4 = 343
    d5 = 156

    tcp = coord[:3]
    dir_vec = coord[3:]

    # point A is the intersection point of joint 4, 5, and 6
    dir_vec = dir_vec * d5
    A = tcp - dir_vec

    # print("inverse dir_pt: ")
    # print(A)

    # calculate point B's coordinate
    Bz = A[2]
    Bx_1 = (A[0] * (A[0]**2 + A[1]**2 - d3**2) - d3 * A[1] * np.sqrt(A[0]**2 + A[1]**2 - d3**2)) / (A[0]**2 + A[1]**2)
    Bx_2 = (A[0] * (A[0]**2 + A[1]**2 - d3**2) + d3 * A[1] * np.sqrt(A[0]**2 + A[1]**2 - d3**2)) / (A[0]**2 + A[1]**2)
    By_1 = (A[1] * (A[0]**2 + A[1]**2 - d3**2) + d3 * A[0] * np.sqrt(A[0]**2 + A[1]**2 - d3**2)) / (A[0]**2 + A[1]**2)
    By_2 = (A[1] * (A[0]**2 + A[1]**2 - d3**2) - d3 * A[0] * np.sqrt(A[0]**2 + A[1]**2 - d3**2)) / (A[0]**2 + A[1]**2)

    # calculate phi_1
    Bphi_1 = np.arctan2(By_1, Bx_1)
    Bphi_2 = np.arctan2(By_2, Bx_2)

    phi_1_minus_2 = np.mod(Bphi_1 - Bphi_2, 2*np.pi)
    if phi_1_minus_2 < np.pi:
        B_list = [np.array([Bx_1, By_1, Bz]),
                  np.array([Bx_1, By_1, Bz]),
                  np.array([Bx_2, By_2, Bz]),
                  np.array([Bx_2, By_2, Bz])]
    else:
        B_list = [np.array([Bx_2, By_2, Bz]),
                  np.array([Bx_2, By_2, Bz]),
                  np.array([Bx_1, By_1, Bz]),
                  np.array([Bx_1, By_1, Bz])]

    def norm_neg_pi_2_pos_pi(rad):
        return (rad + np.pi) % (2 * np.pi) - np.pi

    phi_1_list = []
    for idx in range(len(B_list)):
        if idx in [0, 1]:
            phi_1_list.append(norm_neg_pi_2_pos_pi(np.arctan2(B_list[idx][1], B_list[idx][0])))
        else:
            phi_1_list.append(norm_neg_pi_2_pos_pi(np.arctan2(B_list[idx][1], B_list[idx][0]) + np.pi))

    # calculate phi_2 and phi_3, should have two pair of solutions
    D_list = []
    for idx in range(len(B_list)):
        if idx in [0, 1]:
            Dx = np.sqrt(B_list[idx][0]**2 + B_list[idx][1]**2) - d1
        else:
            Dx = np.sqrt(B_list[idx][0]**2 + B_list[idx][1]**2) + d1
        Dy = A[2] - d0
        D_list.append(np.array([Dx, Dy]))

    if np.sqrt(Dx**2 + Dy**2) > d2 + d4:
        D_list = D_list[:2]
        phi_1_list = phi_1_list[:2]

    def calculate_phi_3(D):
        cos_phi_3 = (D[0]**2 + D[1]**2 - d2**2 - d4**2) / (2 * d2 * d4)
        phi_3 = np.arccos(cos_phi_3)
        return [phi_3, -phi_3]

    phi_3_list = []
    for idx in range(len(D_list)):
        if idx % 2 == 0:
            cur_phi_3_list = calculate_phi_3(D_list[idx])
            # if idx >= 2:
            #     cur_phi_3_list = [-e for e in cur_phi_3_list]
            phi_3_list += cur_phi_3_list

    def sin_phi_2(phi_3):
        return -((d4 * np.cos(phi_3) + d2) * Dx + d4 * np.sin(phi_3) * Dy) / \
                ((d4 * np.cos(phi_3) + d2)**2 + d4**2 * np.sin(phi_3)**2)

    def cos_phi_2(phi_3, s_phi_2):
        return (Dy + d4 * np.sin(phi_3) * s_phi_2) / (d4 * np.cos(phi_3) + d2)

    def calculate_phi_2(phi_3):
        s_phi_2 = sin_phi_2(phi_3)
        c_phi_2 = cos_phi_2(phi_3, s_phi_2)
        phi_2 = np.arctan2(s_phi_2, c_phi_2)
        return phi_2

    phi_2_list = []
    for idx in range(len(phi_3_list)):
        phi_2_list.append(calculate_phi_2(phi_3_list[idx]))

    for idx in range(len(phi_2_list)):
        if idx >= 2:
            # when the direction of target and the first joint is opposite
            phi_2_list[idx] = -phi_2_list[idx]
            phi_3_list[idx] = -phi_3_list[idx]

    # calculate point E's coordinate
    def calculate_E(A, B, phi_2, phi_3):
        Ex = B[0] * (np.sqrt(B[0]**2 + B[1]**2) + d4 * np.sin(phi_2 + phi_3)) / np.sqrt(B[0]**2 + B[1]**2)
        Ey = B[1] * (np.sqrt(B[0]**2 + B[1]**2) + d4 * np.sin(phi_2 + phi_3)) / np.sqrt(B[0]**2 + B[1]**2)
        Ez = A[2] - d4 * np.cos(phi_2 + phi_3)
        return np.array([Ex, Ey, Ez])

    E_list = []
    for idx in range(len(phi_2_list)):
        if idx < 2:
            E_list.append(calculate_E(A, B_list[idx], phi_2_list[idx], phi_3_list[idx]))
        else:
            E_list.append(calculate_E(A, B_list[idx], -phi_2_list[idx], -phi_3_list[idx]))

    # calculate phi_5
    def calculate_phi_5(E, B, A, T):
        EB = B - E
        AT = T - A
        cos_phi_5 = np.dot(EB, AT) / d4 / d5

        BA = A - B
        vec = np.cross(EB, AT)

        ang = vecs_angle(BA, vec)
        phi_5 = np.arccos(cos_phi_5)

        # if BA and vec are in the same direction, phi_5 is positive, else is negative
        if abs(ang) > 0.001:
            phi_5 = -phi_5

        return phi_5

    phi_5_list = []
    for idx in range(len(E_list)):
        phi_5_list.append(calculate_phi_5(E_list[idx], B_list[idx], A, tcp))

    # calculate point F's coordinate
    def calculate_F(A, B, E, phi_5):
        Fx = A[0] + np.cos(phi_5) * d5 * (B[0] - E[0]) / d4
        Fy = A[1] + np.cos(phi_5) * d5 * (B[1] - E[1]) / d4
        Fz = A[2] + np.cos(phi_5) * d5 * (B[2] - E[2]) / d4
        return np.array([Fx, Fy, Fz])

    F_list = []
    for idx in range(len(E_list)):
        F_list.append(calculate_F(A, B_list[idx], E_list[idx], phi_5_list[idx]))

    # calculate beta
    beta_list = []
    for idx in range(len(E_list)):
        beta_list.append(np.arcsin((B_list[idx][2] - E_list[idx][2]) / d4))

    # calculate point T_prime's coordinate
    def calculate_T_prime(A, F, phi_5, beta):
        T_prime_x = A[0] + (F[0] - A[0]) * d5 * np.cos(phi_5 + beta) / (d5 * np.cos(phi_5) * np.cos(beta))
        T_prime_y = A[1] + (F[1] - A[1]) * d5 * np.cos(phi_5 + beta) / (d5 * np.cos(phi_5) * np.cos(beta))
        T_prime_z = A[2] + d5 * np.sin(phi_5 + beta)
        return np.array([T_prime_x, T_prime_y, T_prime_z])

    T_prime_list = []
    for idx in range(len(F_list)):
        T_prime_list.append(calculate_T_prime(A, F_list[idx], phi_5_list[idx], beta_list[idx]))

    # calculate phi_4
    def calculate_phi_4(A, F, T, T_prime):
        FT_prime = T_prime - F
        FT = T - F
        cos_phi_4 = np.dot(FT_prime, FT) / (np.linalg.norm(FT_prime) * np.linalg.norm(FT))

        AF = F - A
        vec = np.cross(FT, FT_prime)

        ang = vecs_angle(AF, vec)
        phi_4 = np.arccos(cos_phi_4)

        # if AF and vec are in the same direction, phi_4 is positive, else is negative
        if abs(ang) < 0.001:
            phi_4 = -phi_4

        return phi_4

    phi_4_list = []
    for idx in range(len(F_list)):
        phi_4 = calculate_phi_4(A, F_list[idx], tcp, T_prime_list[idx])
        if idx >= 2:
            phi_4 = norm_neg_pi_2_pos_pi(phi_4 + np.pi)
        phi_4_list.append(phi_4)

    def list_rad2angle(rad_list):
        angle_list = [rad2angle(e) for e in rad_list]
        return angle_list

    phi_1_angle_list = list_rad2angle(phi_1_list)
    phi_2_angle_list = list_rad2angle(phi_2_list)
    phi_3_angle_list = list_rad2angle(phi_3_list)
    phi_4_angle_list = list_rad2angle(phi_4_list)
    phi_5_angle_list = list_rad2angle(phi_5_list)


    sol_list = []
    for idx in range(len(phi_1_angle_list)):
        sol_list.append(np.array([phi_1_angle_list[idx],
                                  phi_2_angle_list[idx],
                                  phi_3_angle_list[idx],
                                  phi_4_angle_list[idx],
                                  phi_5_angle_list[idx]]))

    print("solutions:")
    for sol in sol_list:
        print(np.around(sol, 3))
    return sol_list

if __name__ == "__main__":
    for idx in range(1):
        print("Round %d" % idx)
        phi_angle_ary = [random.uniform(-160, 160),
                         random.uniform(-70, 70),
                         random.uniform(-120, 120),
                         random.uniform(-120, 120),
                         random.uniform(-90, 90),
                         random.uniform(-180, 180)]
        phi_angle_ary = [94.354, 42.293, 27.137, 22.269, -45.978, 0]
        print(np.around(phi_angle_ary[:5], 3))
        ret = forward_kinematics(phi_angle_ary)
        # print("tcp and direction:")
        # print(np.around(ret, 3))
        sol_list = inverse_kinematics(np.array(ret))
