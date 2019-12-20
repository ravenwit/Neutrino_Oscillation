#!/usr/bin/env python

"""
    Importing necessary libraries
"""

from __future__ import division
import numpy as np
import time
from random import random as rnd
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import mpmath
import simulate


directory = "Data/3"

# Preprocessor Definitions
NO = 0
YES = 1
exi = 1
NXM = 10240
NKXM = 5120
NPMAX = 4000000  # maximum number of super-particles

#   Definitions of constants
PI = np.pi
HBAR = 1.05457266e-34  # Reduced Planck constant
LIGHT_V = 299792458  # Light velocity
M_e = 9.1093897e-31  # Electron mass
Q = 1.60217733e-19  # Electron charge in absolute value
eV = 1.6e-19  # 1 eV in J
M_eV = M_e * LIGHT_V * LIGHT_V  # Electon mass in J
M1_eV = 0.049  # 1st Eigenstate mass in eV
M1_J = M1_eV * eV
M1 = M1_J / (LIGHT_V ** 2)
M2_eV = 0.050  # 2nd Eigenstate mass in eV
M2_J = M2_eV * eV
M2 = M2_J / (LIGHT_V ** 2)
M3_eV = 0.070  # 3rd Eigenstate mass in eV
M3_J = M3_eV * eV
M3 = M3_J / (LIGHT_V ** 2)
M3i_eV = 0.0087  # 3rd Eigenstate mass in eV for inverted hierarchy
M3i_J = M3i_eV * eV
M3i = M3i_J / (LIGHT_V ** 2)

E_n_eV = 1e6  # Neutrino energy in eV
E_n_J = E_n_eV * eV  # Neutrino energy in J

#   In Hartree atomic unit
H_HBAR = 1
H_LIGHT_V = 137  # Light veelocity in Hertree unit
H_LENGTH = H_LIGHT_V / LIGHT_V
H_M_e = 1
H_M1 = M1 / M_e
H_M2 = M2 / M_e
H_M3 = M3 / M_e
H_M3i = M3i / M_e

H_eV = 27.211385  # Hartree energy in eV

M_M1 = M_e
M_M2 = (M2 / M1) * M_M1
M_M3 = (M3 / M1) * M_M1

#   Particular Calculation

E_n = E_n_eV / H_eV
M = M_M3
P_n = np.sqrt(E_n ** 2 - M ** 2 * H_LIGHT_V ** 4) / H_LIGHT_V

#   All integers
FINAL = 0
K = np.zeros(NPMAX + exi)
W = np.zeros(NPMAX + exi)
DIST = np.zeros((NXM + exi, 2 * NKXM + exi))
ISEED = 38467
UPDATED = np.zeros(NPMAX + exi)

# All doubles here...
FW1 = np.zeros((NXM + exi, 2 * NKXM + exi))
DENSX = np.zeros(NXM + exi)
DENSK = np.zeros(2 * NKXM + exi)
PHI = np.zeros(NXM + exi)
TIME = 0.0
BKTQ = 0.0
QH = 0.0

P = np.zeros(NPMAX + exi)

VW = np.zeros((NXM + exi, 2 * NKXM + exi))

GAMMA = np.zeros(NXM + exi)

PTIME = np.zeros(NPMAX + exi)

# Initial conditions

INUM = 2000  # maximum number of particles in a phase-space cell for the initial distribution 20 (#200)
# LX = 200.e-9  # total length of spatial domain
LX = 500  # total length of spatial domain (#200)
# LX = 2.44e10 * H_LENGTH  # total length of spatial domain (#200)
# DT = 0.01e-15  # time step
DT = 10  # time step (1)    (#10)
# LC = 50.e-9  # coherence length
# LC = (LX / 10) * 0.3  # coherence length
LC = 250  # coherence length (#100)
# LC = PI / (n_v * DT)  # coherence length (#100)
NX = 500  # number of cells in x-direction (500) (#200)
# ITMAX = 400000  # total number of time steps 200
# ITMAX = int(LX/DT) *5 # total number of time steps 200 (#4000)
ITMAX = 8000  # total number of time steps 200 (#4000)
ANNIHILATION_FREQUENCY = 100  # (#100)

# SIGMA_WAVE_PACKET = 3.15e-9  # wave packet dispersion
# SIGMA_WAVE_PACKET = (5.1/3.4)*0.1  # wave packet dispersion    (#10)
SIGMA_WAVE_PACKET = 10  # wave packet dispersion    (#10)
# SIGMA_WAVE_PACKET = 3.15e7 * H_LENGTH  # wave packet dispersion    (#10)
# X0_WAVE_PACKET = LX / 2 - 31.5e-9  # wave packet initial position
X0_WAVE_PACKET = SIGMA_WAVE_PACKET + 10  # wave packet initial position (#SIGMA_WAVE_PACKET + 10 )

BARRIER_POTENTIAL = 0  # value of the potential barrier
BARRIER_POSITION = 0.5 * LX  # barrier center position
BARRIER_WIDTH = 6.e-9  # barrier width

# spatial cell length
DX = LX / NX

# automatic calculation of NKX
NKX = (int)(0.5 * LC / DX)

# pseudo - wave vector length
DKX = 10 * PI / LC

K0_WAVE_PACKET = 100 * DKX  # 1st eigenmass wave packet initial wave vector (150) (#500)
K1_WAVE_PACKET = 50.0 * DKX  # 2nd eigenmass wave packet initial wave vector (120) (#150)
K2_WAVE_PACKET = 1000 * DKX  # 3rd eigenmass wave packet initial wave vector (100)  (#100)

# PMNS Matrix
theta12 = (np.pi / 180) * 35.26
theta23 = (np.pi / 180) * 45
theta13 = (np.pi / 180) * 13.2
delta = np.pi / 4
c12 = np.cos(theta12)
c23 = np.cos(theta23)
c13 = np.cos(theta13)
s12 = np.sin(theta12)
s23 = np.sin(theta23)
s13 = np.sin(theta13)

PMNS = [[c12 * c13, s12 * c13, s13 * np.exp(delta)],
        [-s12 * c23 - c12 * s23 * s13 * np.exp(delta), c12 * c23 - s12 * s23 * s13 * np.exp(delta), s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * np.exp(delta), -c12 * s23 - s12 * c23 * s13 * np.exp(delta), c23 * c13]
        ]
theta = (np.pi / 180) * 45
PMNS2 = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]
         ]


def rnda():
    global ISEED
    ISEED = (1027.0 * ISEED) % 1048576.0
    return ISEED / 1048576.0


def distribution():
    global INUM
    print("Calculation of distribution fucntion\n")
    print("Number of particles {} \n".format(INUM))

    for i in range(0, NX + 1):
        for k in range(0, 2 * NKX - 1):
            DIST[i][k] = 0

    # cloud in cell algorithm

    for n in range(0, INUM):
        i = int(P[n] / DX) + 1
        k = K[n]
        if (0 < i) and (i <= NX) and (-NKX < k) and (k < NKX):
            DIST[int(i)][int(k + NKX - 1)] += W[n]  ###########njkefhjkwehfjkewb

    # stores the normalized quasi-distribution function
    norm = 0
    for i in range(1, NX + 1):
        for j in range(-NKX + 1, NKX):
            FW1[i][j + NKX - 1] = (DIST[i][j + NKX - 1])
    for i in range(1, NX + 1):
        for j in range(-NKX + 1, NKX):
            norm += FW1[i][j + NKX - 1]
    norm *= DX * DKX
    for i in range(1, NX + 1):
        for j in range(-NKX + 1, NKX):
            FW1[i][j + NKX - 1] /= norm

    print("end of distribution function calculation")


def density():
    for i in range(1, NX + 1):
        sum = 0
        for j in range(-NKX + 1, NKX):
            sum += FW1[i][j + NKX - 1]
        DENSX[i] = sum * DKX
    DENSX[1] = DENSX[NX] = 0

    # in k space
    for j in range(-NKX + 1, NKX):
        sum = 0
        for i in range(1, NX + 1):
            sum += FW1[i][j + NKX - 1]
        DENSK[j + NKX - 1] = sum * DX


def kernel():
    for i in range(1, NX + 1):
        for j in range(0, NKX):
            VW[i][j] = 0

            for l in range(1, int(0.5 * LC / DX) + 1 + 1):
                if l <= (i + 1) and (i + 1) <= NX and l <= (i - 1) and (i - 1) <= NX:
                    VW[i][j] += np.sin(2. * j * DKX * (1 - 0.5) * DX) * PHI[i + 1] - PHI[i - 1]

            VW[i][j] *= -2. * (-Q) * DX / (HBAR * LC)


def calculate_gamma():
    for i in range(1, NX + 1):
        GAMMA[i] = 0

        # the implementation below holds taking into account the fact that
        # the Wigner potential is anti-symetric w.r.t the k-space

        # for j in range(1, NKX):
        #     GAMMA[i] += abs(VW[i][j])


def devconf():
    global INUM, PMNS, PMNS2

    d_max = 0

    # definition of the initial conditions

    for i in range(1, NX + 1):
        for j in range(-NKX + 1, NKX):
            # FW1[i][j + NKX - 1] = pow((PMNS2[0][0]), 2) * np.exp(
            #     -pow(((i - 0.5) * DX - X0_WAVE_PACKET) / SIGMA_WAVE_PACKET, 2.0)) * \
            #                       np.exp(-pow(((j * DKX) - K0_WAVE_PACKET) * SIGMA_WAVE_PACKET, 2.0)) + \
            #                       pow((PMNS2[0][1]), 2) * np.exp(
            #                           -pow(((i - 0.5) * DX - X0_WAVE_PACKET) / SIGMA_WAVE_PACKET, 2.0)) * \
            #                       np.exp(-pow(((j * DKX) - K1_WAVE_PACKET) * SIGMA_WAVE_PACKET, 2.0)) +\
            #                       pow((PMNS2[1][0]), 2) * np.exp(
            #                           -pow(((i - 0.5) * DX - X0_WAVE_PACKET) / SIGMA_WAVE_PACKET, 2.0)) * \
            #                       np.exp(-pow(((j * DKX) - K0_WAVE_PACKET) * SIGMA_WAVE_PACKET, 2.0)) + \
            #                       pow((PMNS2[1][1]), 2) * np.exp(
            #                           -pow(((i - 0.5) * DX - X0_WAVE_PACKET) / SIGMA_WAVE_PACKET, 2.0)) * \
            #                       np.exp(-pow(((j * DKX) - K1_WAVE_PACKET) * SIGMA_WAVE_PACKET, 2.0))

            # FW1[i][j + NKX - 1] = pow((PMNS[0][0]), 2) * np.exp(
            #     -pow(((i - 0.5) * DX - X0_WAVE_PACKET) / SIGMA_WAVE_PACKET, 2.0)) * \
            #                       np.exp(-pow(((j * DKX) - K0_WAVE_PACKET) * SIGMA_WAVE_PACKET, 2.0)) + \
            #                       pow((PMNS[0][1]), 2) * np.exp(
            #                           -pow(((i - 0.5) * DX - X0_WAVE_PACKET) / SIGMA_WAVE_PACKET, 2.0)) * \
            #                       np.exp(-pow(((j * DKX) - K1_WAVE_PACKET) * SIGMA_WAVE_PACKET, 2.0)) + \
            #                       pow((PMNS[0][2]), 2) * np.exp(
            #                           -pow(((i - 0.5) * DX - X0_WAVE_PACKET) / SIGMA_WAVE_PACKET, 2.0)) * \
            #                       np.exp(-pow(((j * DKX) - K2_WAVE_PACKET) * SIGMA_WAVE_PACKET, 2.0))


            FW1[i][j + NKX - 1] = np.exp(
                -pow(((i - 0.5) * DX - X0_WAVE_PACKET) / SIGMA_WAVE_PACKET, 2.0)) * \
                                  np.exp(-pow(((j * DKX) - K0_WAVE_PACKET) * SIGMA_WAVE_PACKET, 2.0))

    # normalization of the initial condition
    norm = 0
    for i in range(1, NX + 1):
        for j in range(-NKX + 1, NKX):
            norm += FW1[i][j + NKX - 1]

    norm *= DX * DKX

    for i in range(1, NX + 1):
        for j in range(-NKX + 1, NKX):
            FW1[i][j + NKX - 1] /= norm

    # calculate the EPP variable for the cloud in cell algorithm

    for i in range(1, NX + 1):
        for j in range(0, 2 * NKX - 1):
            if d_max < abs(FW1[i][j]):
                d_max = abs(FW1[i][j])

    epp = d_max / INUM

    #   calculate initial distribution function
    print("config() - calculating initial distribution\n")
    INUM = 0

    for i in range(1, NX + 1):
        for j in range(0, 2 * NKX - 1):
            local_number_of_particles = int(abs(FW1[i][j]) / epp + 0.5)

            #   creates the new local particels in the (i, k) th phase space cell
            #   the particles are uniformly distributes in space

            for n in range(1, local_number_of_particles + 1):
                m = INUM + n - 1
                if rnd() > 0.5:
                    P[m] = (i - 0.5 + 0.5 * rnd()) * DX
                else:
                    P[m] = (i - 0.5 - 0.5 * rnd()) * DX
                K[m] = j - NKX + 1
                if FW1[i][j] > 0:
                    W[m] = +1
                else:
                    W[m] = -1
            INUM += local_number_of_particles

    distribution()

    print("Initiail number of electron super particles {} \n".format(INUM))


def WMC():
    '''
        Evolution of the particles
        and creating of (+,-) couples
    :return:
    '''

    global INUM
    INUM = int(INUM)

    sum = 0
    number_of_created_particles = 0

    # initial settings
    number_of_outside_particles = 0
    all_particles_updated = NO

    for n in range(0, INUM):
        UPDATED[n] = NO
    for n in range(0, INUM):
        PTIME[n] = DT

    while all_particles_updated == NO:
        number_of_outside_particles = 0

        # evolution and couples creation
        for n in range(0, INUM):
            if UPDATED[n] == NO:
                hmt = HBAR / (M) * PTIME[n]

                # drift n-th particle
                x0 = P[n]
                k0 = K[n] * DKX
                i = int(x0 / DX + 1)  # int convert

                # evolve position and wave vector of the n-th particle
                if i > 0 and i <= NX and -NKX < K[n] and K[n] < NKX:
                    P[n] = x0 + hmt * k0

                    # calculate the probability that the wave-vector actually evoloves
                    # accordingly to the continuous dkx
                    # check if  a couple of (+,-) have to be created

                    if GAMMA[i] != 0:
                        time = 0
                        while time < PTIME[n]:
                            rdt = -np.log(rnd()) / GAMMA[i]
                            time += rdt
                            if time < PTIME[n]:
                                created = NO
                                r = rnd()
                                sum = 0.

                                #    random selection of the wave-vector
                                j = 0
                                while created == NO and j < NKX:
                                    p = abs(VW[i][j]) / GAMMA[i]
                                    if sum <= r and r < (sum + p):
                                        number_of_outside_particles += 2
                                        num = INUM + number_of_created_particles

                                        #   select a random time inettrval when the creating happens
                                        #   assign position

                                        P[num - 2] = P[num - 1] = x0 + HBAR / (MSTAR * M) * time * k0

                                        #   assign eave-vector
                                        if VW[i][j] >= 0.:
                                            K[num - 2] = K[n] + j
                                            K[num - 1] = K[n] - j
                                        else:
                                            K[num - 2] = K[n] - j
                                            K[num - 1] = K[n] + j

                                        # assign quantum weight
                                        if W[n] == 1:
                                            W[num - 2] = +1
                                            W[num - 1] = -1
                                        else:
                                            W[num - 2] = -1
                                            W[num - 1] = +1

                                            #   assign flag to evolove the particcles at the next loop
                                            UPDATED[num - 2] = UPDATED[num - 1] = NO

                                        # assign time
                                        PTIME[num - 2] = PTIME[num - 1] = PTIME[n] - time

                                        #   eventually ignore the just-created couples since at least
                                        #   one of them outside the device
                                        if K[num - 2] <= -NKX or K[num - 2] >= NKX or K[num - 1] <= -NKX or K[
                                                    num - 1] >= NKX:
                                            num = INUM - 2
                                            number_of_created_particles -= 2

                                        created = YES

                                    sum += p
                                    j += 1

                else:
                    number_of_outside_particles += 1

                UPDATED[n] = YES

        # end of for (n=0;...)

        INUM += number_of_created_particles

        print("INUM = {}  --  particles created = {}\n".format(INUM, number_of_created_particles))

        if INUM > NPMAX:
            print("Number of particles has exploded - please increase NPMAX and recompile\n")
            exit(0)

        # checks if all particles have been updated

        flag = YES

        for n in range(0, INUM):
            if UPDATED[n] == NO:
                flag = NO
        all_particles_updated = flag

    print("--number of particles outside  =  {}  -- \n".format(number_of_outside_particles))


def annihilation():
    global INUM

    print("\n# of particles before annihilation = {}\n".format(INUM))

    # calculates the new array of particles
    INUM = 0
    for i in range(1, NX + 1):
        for k in range(0, 2 * NKX - 1):
            local_number_of_particles = abs(DIST[i][k])

            #   creates the new local particles in the (i,k)-th phase-sapce cell
            #   the particles are uniformly distributed in space

            for n in range(1, int(local_number_of_particles + 1)):
                m = int(INUM + n)
                if rnd() > 0.5:
                    P[m] = (i - 0.5 + 0.5 * rnd()) * DX
                else:
                    P[m] = (i - 0.5 - 0.5 * rnd()) * DX
                K[m] = k - NKX + 1
                if DIST[i][k] > 0:
                    W[m] = +1
                else:
                    W[m] = -1

            INUM += local_number_of_particles

    print("# of particles after the annihilation = {}\n\n".format(INUM))


def save(ind):
    if ind == 0 or ind == 1:

        # saves potential
        fp = open("{}/potential.dat".format(directory), "w")
        for i in range(1, NX + 1):
            fp.write("{} {}\n".format((i - 0.5) * DX, PHI[i]))
        fp.close()

        # saves gamma function
        fp = open("{}/gamma.dat".format(directory), "w")
        for i in range(1, NX + 1):
            fp.write("{} {}\n".format((i - 0.5) * DX, GAMMA[i]))
        fp.close()

        # saves the coordinates axis values
        fp = open("{}/x.dat".format(directory), "w")
        for i in range(1, NX + 1):
            fp.write("{}\n".format((i - 0.5) * DX))
        fp.close()

        fp = open("{}/k.dat".format(directory), "w")
        for i in range(-NKX, NKX):
            fp.write("{}\n".format((i + 0.5) * DKX))
        fp.close()

    # saves normalized the Wigner quasi - distribution
    # == == == == == == == == == == == == == == == == == == == == == == ==
    fp = open("{}/Wigner_quasi_distribution_{}.dat".format(directory, ind), "w")
    for i in range(1, NX + 1):
        for j in range(-NKX, NKX):
            if j == -NKX:
                fp.write("{} ".format(FW1[i][-NKX + 1 + NKX - 1]))
            else:
                fp.write("{} ".format(FW1[i][j + NKX - 1]))
        fp.write("\n")
    fp.close()

    # saves the electron probability density in x - space
    fp = open("{}/wigner_probability_density_{}.dat".format(directory, ind), "w")
    for i in range(1, NX + 1):
        fp.write("{}\n".format(DENSX[i]))
    fp.close()

    # saves the electron probability density in k space
    fp = open("{}/Wigner_k_space_probability_density_{}.dat".format(directory, ind), "w")
    for i in range(-NKX, NKX):
        if i == -NKX:
            fp.write("{} {}\n".format((i + 0.5) * DKX, DENSK[0]))
        else:
            fp.write("{} {}\n".format((i + 0.5) * DKX, DENSK[i + NKX - 1]))
    fp.close()


def _main():
    global NO, YES, NXM, NKXM, NPMAX, \
        Q, HBAR, M, MSTAR, PI, \
        NKX, FINAL, K, W, DIST, ISEED, UPDATED, \
        FW1, DENSX, DENSK, PHI, DX, DKX, TIME, BKTQ, QH, \
        P, VW, GAMMA, PTIME, \
        SIGMA_WAVE_PACKET, X0_WAVE_PACKET, K0_WAVE_PACKET, \
        BARRIER_POTENTIAL, BARRIER_POSITION, BARRIER_WIDTH, \
        INUM, LX, LC, NX, DT, ITMAX, ANNIHILATION_FREQUENCY

    # Print the innitial number of particles
    print("\nMAXIMUM NUMBER OF PARTICLES ALLOWED = {}\n\n".format(NPMAX))

    # defines the potental barrier
    for i in range(1, NX + 1):
        PHI[i] = 0
    for i in range(1, NX + 1):
        pos = (i - 0.5) * DX
        if pos >= (BARRIER_POSITION - 0.5 * BARRIER_WIDTH) and pos <= (BARRIER_POSITION + 0.5 * BARRIER_WIDTH):
            PHI[i] += BARRIER_POTENTIAL

    # set gamma function to zero
    for i in range(1, NX + 1 + 1):
        GAMMA[i] = 0

    # get initial time
    nowtm = time.time()
    print('Simulation started: {}'.format(nowtm))

    ### Initilization

    devconf()
    density()
    save(0)

    print("\n")

    # updates the solution

    for i in range(1, ITMAX + 1):
        TIME += DT
        print("{} of {} -- Time={} \n\n".format(i, ITMAX, TIME))

        if i == 1:
            print("calculating Wigner potential\n")
            # kernel()
            print("calculating gamma function\n")
            calculate_gamma()

        print("evolving wigner\n")

        WMC()
        print("calculating distribution function\n")
        distribution()
        print("calculating density in x- and k-space\n")
        density()
        if i % ANNIHILATION_FREQUENCY == 0 and i is not ITMAX:
            print("Annihiliation of particles\n")
            annihilation()
        save(i)

    print("\n")

    print("output files saved\n\n")

    endt = time.time()

    print("Simulation ended:     {}".format(endt))


# def _visualize(index, type):
#     if type == 1:
#         fp = open("wigner_probability_density_{}.dat".format(index), 'r')
#         XX=[]
#         YY=[]
#         for line in fp:
#             a, b = line.split(' ')
#             a=float(a)
#             b=float(b.replace('\n',''))
#             XX.append(a)
#             YY.append(b)
#             plt.plot(XX, YY)
#
#
# def _simulate():
#     f = plt.figure(figsize=(10, 7), dpi=80)
#     ax=f.gca()
#     time0 = time.time()
#     _visualize(1, 1)
#     time1 = time.time()
#     interval = 1000 * (1 / 60) - float(time1 - time0)
#
#     ani = anim.FuncAnimation(f, _visualize, ITMAX,
#                              fargs=(1),
#                              interval=interval, blit=False)
#     plt.show()


if __name__ == '__main__':
    _main()
    con = int(input("Do you want to run simulation: "))
    if con == 1:
        simulate.simulate(1, int(ITMAX))
    else:
        exit(0)
