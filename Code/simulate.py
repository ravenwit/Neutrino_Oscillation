#!/usr/bin/env python

"""
    Importing necessary libraries
"""

from __future__ import division
import time
import numpy as np
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

interval = 100

fig = plt.figure()
ax = fig.add_subplot(aspect='equal', autoscale_on=False)

fig1 = plt.figure()

directory = "Data"

XX = []
_XX = []
NXX = 0
KK = []
NKK = 0

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


def plotEssentials(figure, index, data):
    global _XX
    plt.clf()
    axes = figure.gca()

    axes.set_xlim([min(_XX), max(_XX)])
    axes.set_ylim([0, 0.01])
    axes.set_xlabel('Length')
    axes.set_ylabel('Probability Difference')

    plt.title('Time {}'.format(index))

    axes.autoscale(enable=False, tight=False)

    axes.plot(_XX, data, label='Electron Neutrino -> Mu Neutrino')

    axes.legend()
    fig.show()


def getGrid():
    global XX, KK, NKK, NXX, _XX
    fp = open("{}/x.dat".format(directory), 'r')
    for line in fp:
        XX.append(float(line))
    fp.close()
    fp = open("{}/k.dat".format(directory), 'r')
    for line in fp:
        KK.append(float(line))
    fp.close()
    NXX = len(XX)
    NKK = len(KK)
    _XX = XX
    XX, KK = np.meshgrid(KK, XX)


def visualize(index):
    global _XX, fig
    index *= interval
    fp = open("{}/wigner_probability_density_{}.dat".format(directory, index), 'r')
    if fp is False:
        print('error')
        exit(0)
    WP = []
    for line in fp:
        WP.append(float(line.strip()))
    fp.close()

    plotEssentials(fig, index, WP)



def visualizeSuper(index):
    global _XX, fig
    index *= interval
    fp1 = open("{}/1/wigner_probability_density_{}.dat".format(directory, index), 'r')
    fp2 = open("{}/2/wigner_probability_density_{}.dat".format(directory, index), 'r')
    fp3 = open("{}/3/wigner_probability_density_{}.dat".format(directory, index), 'r')

    if fp1 is False and fp2 is False and fp3 is False:
        print('error')
        exit(0)

    WPD = []

    for line in fp1:
        WPD.append(pow(PMNS[2][0],2) * float(line.strip()))
    fp1.close()

    i = 0
    for line in fp2:
        WPD[i] += pow(PMNS[2][1],2) * float(line.strip())
        i += 1
    fp2.close()

    i = 0
    for line in fp3:
        WPD[i] += pow(PMNS[2][2],2) * float(line.strip())
        i += 1
    fp3.close()

    plotEssentials(fig, index, WPD)
    # plt.savefig("Data/figures/Electron/{}.png".format(index))


def visualizeEMuD(index):
    global _XX, fig
    index *= interval
    fp1 = open("{}/e/WPD_{}.dat".format(directory, index), 'r')
    fp2 = open("{}/mu/WPD_{}.dat".format(directory, index), 'r')

    if fp1 is False and fp2 is False:
        print('error')
        exit(0)

    WPDD = []

    for line in fp1:
        WPDD.append(float(line.strip()))
    fp1.close()

    i = 0
    for line in fp2:
        WPDD[i] -= float(line.strip())
        i += 1
    fp2.close()

    plotEssentials(fig, index, WPDD)
    # plt.savefig("Data/figures/DiffE/{}.png".format(index))


def visualizeElectron(index):
    global _XX, fig
    index *= interval
    fp = open("{}/e/WPD_{}.dat".format(directory, index), 'r')

    if fp is False:
        print('error')
        exit(0)

    WP = []

    for line in fp:
        WP.append(float(line.strip()))
    fp.close()

    plotEssentials(fig, index, WP)


def visualizeMu(index):
    global _XX, fig
    index *= interval
    fp = open("{}/mu/WPD_{}.dat".format(directory, index), 'r')

    if fp is False:
        print('error')
        exit(0)

    WP = []

    for line in fp:
        WP.append(float(line.strip()))
    fp.close()

    plotEssentials(fig, index, WP)


def visualizeTau(index):
    global _XX, fig
    index *= interval
    fp = open("{}/tau/WPD_{}.dat".format(directory, index), 'r')

    if fp is False:
        print('error')
        exit(0)

    WP = []

    for line in fp:
        WP.append(float(line.strip()))
    fp.close()

    plotEssentials(fig, index, WP)


def visualizeEvery(index):
    global _XX, fig
    index *= interval
    fp = open("{}/e/WPD_{}.dat".format(directory, index), 'r')

    if fp is False:
        print('error')
        exit(0)

    WPe = []

    for line in fp:
        WPe.append(float(line.strip()))
    fp.close()

    fp = open("{}/mu/WPD_{}.dat".format(directory, index), 'r')

    if fp is False:
        print('error')
        exit(0)

    WPmu = []

    for line in fp:
        WPmu.append(float(line.strip()))
    fp.close()

    fp = open("{}/tau/WPD_{}.dat".format(directory, index), 'r')

    if fp is False:
        print('error')
        exit(0)

    WPtau = []

    for line in fp:
        WPtau.append(float(line.strip()))
    fp.close()

    plt.clf()
    axes = fig.gca()

    axes.set_xlim([min(_XX), max(_XX)])
    axes.set_ylim([0, 0.1])
    axes.set_xlabel('Length')
    axes.set_ylabel('Amplitude')

    plt.title('Time {}'.format(index))

    axes.autoscale(enable=False, tight=False)

    axes.plot(_XX, WPe, label='Eletron Neutrino')
    axes.plot(_XX, WPmu, label='Muon Neutrino')
    axes.plot(_XX, WPtau, label='Tau Neutrino')

    axes.legend()
    fig.show()
    plt.savefig("Data/figures/Every/{}.png".format(index))



def visualizeQUASI(index):
    global XX, KK, NXX, NKK, fig1

    index *= interval
    fp = open("{}/1/Wigner_quasi_distribution_{}.dat".format(directory, index), 'r')
    if fp is False:
        print('error')
        exit(0)
    WQP = np.zeros((NXX, NKK))
    i = 0
    for line in fp:
        line = line.split(' ')[:-1]
        j = 0
        for value in line:
            WQP[i][j] = float(value)
            j += 1
        i += 1
    fp.close()

    plt.clf()
    axes = fig1.gca(projection='3d')

    # axes.get_xaxis().set_visible(False)
    # axes.get_yaxis().set_visible(False)
    # axes.get_zaxis().set_visible(False)
    axes.set_yticklabels([])
    axes.set_zticklabels([])
    axes.set_xticklabels([])

    plt.title("Time {}".format(index))

    surf = axes.plot_surface(XX, KK, WQP, cmap=cm.coolwarm,
                             linewidth=0, antialiased=False)

    # axes.set_xlabel("X", fontsize=10)
    # axes.set_ylabel("Y", fontsize=10)
    # axes.set_zlabel("$\phi (x, y)$", fontsize=12, rotation=-90)

    axes.zaxis.set_major_locator(LinearLocator(10))
    axes.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig1.colorbar(surf, shrink=0.5, aspect=5)

    fig1.show()
    plt.savefig('Data/figures/quasi/{}.png'.format(index))

def visualizeQUASIF(index):
    global XX, KK, NXX, NKK, fig1

    index *= interval
    fp1 = open("{}/1/Wigner_quasi_distribution_{}.dat".format(directory, index), 'r')
    fp2 = open("{}/2/Wigner_quasi_distribution_{}.dat".format(directory, index), 'r')
    fp3 = open("{}/3/Wigner_quasi_distribution_{}.dat".format(directory, index), 'r')
    if fp1 is False and fp2 is False and fp3 is False:
        print('error')
        exit(0)
    WQP = np.zeros((NXX, NKK))
    i = 0
    for line in fp1:
        line = line.split(' ')[:-1]
        j = 0
        for value in line:
            WQP[i][j] = pow(PMNS[0][0],2)*float(value)
            j += 1
        i += 1
    fp1.close()
    i=0
    for line in fp2:
        line = line.split(' ')[:-1]
        j = 0
        for value in line:
            WQP[i][j] += pow(PMNS[0][1],2)*float(value)
            j += 1
        i += 1
    fp2.close()
    i=0
    for line in fp3:
        line = line.split(' ')[:-1]
        j = 0
        for value in line:
            WQP[i][j] += pow(PMNS[0][2],2)*float(value)
            j += 1
        i += 1
    fp3.close()


    plt.clf()
    axes = fig1.gca(projection='3d')

    # axes.get_xaxis().set_visible(False)
    # axes.get_yaxis().set_visible(False)
    # axes.get_zaxis().set_visible(False)
    axes.set_yticklabels([])
    axes.set_zticklabels([])
    axes.set_xticklabels([])

    plt.title("Time {}".format(index))

    surf = axes.plot_surface(XX, KK, WQP, cmap=cm.coolwarm,
                             linewidth=0, antialiased=False)

    # axes.set_xlabel("X", fontsize=10)
    # axes.set_ylabel("Y", fontsize=10)
    # axes.set_zlabel("$\phi (x, y)$", fontsize=12, rotation=-90)

    axes.zaxis.set_major_locator(LinearLocator(10))
    axes.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig1.colorbar(surf, shrink=0.5, aspect=5)

    fig1.show()
    plt.savefig('Data/figures/quasi/{}.png'.format(index))


def simulate(type, T):
    T = int(T / interval)
    if type == 1:
        time0 = time.time()
        visualize(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualize, T,
                                 interval=duration, blit=False)

    elif type == 2:
        time0 = time.time()
        visualizeQUASI(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig1, visualizeQUASI, T,
                                 interval=duration, blit=False)
    elif type == 3:
        time0 = time.time()
        visualizeElectron(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeElectron, T,
                                 interval=duration, blit=False)
    elif type == 4:
        time0 = time.time()
        visualizeMu(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeMu, T,
                                 interval=duration, blit=False)
    elif type == 5:
        time0 = time.time()
        visualizeTau(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeTau, T,
                                 interval=duration, blit=False)
    elif type == 6:
        time0 = time.time()
        visualizeSuper(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeSuper, T,
                                 interval=duration, blit=False)
    elif type == 7:
        time0 = time.time()
        visualizeEMuD(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeEMuD, T,
                                 interval=duration, blit=False)
    elif type == 8:
        time0 = time.time()
        visualizeEvery(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeEvery, T,
                                 interval=duration, blit=False)
    elif type == 9:
        time0 = time.time()
        visualizeQUASIF(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeQUASIF, T,
                                 interval=duration, blit=False)

    plt.show()


def plotV(T):
    global _XX
    TT = [i for i in range(1, T + 1)]
    XT = []
    for i in range(1, T + 1):
        fp = open("{}/wigner_probability_density_{}.dat".format(directory, i), 'r')
        maxDEN = 0
        maxX_i = 0
        if fp is False:
            print('error')
            exit(0)
        i = 0
        for line in fp:
            a = float(line.strip())
            if a > maxDEN:
                maxDEN = a
                maxX_i = i
            i += 1
        XT.append(_XX[maxX_i])
    # plt.plot(TT, XX, label='Position')
    VV = []
    for i in range(len(TT) - 1):
        v = (XT[i] - XT[i + 1]) / (TT[i] - TT[i + 1])
        VV.append(v)
    plt.plot(TT[:-1], VV, label='Velocity')
    print(XT[-1], XT[1])
    plt.xlabel('Time')
    plt.legend()
    plt.ylabel('Velocity')
    plt.show()


def plotVSuper(T):
    global _XX
    TT = [i for i in range(1, T + 1)]
    XT = []
    for i in range(1, T + 1):
        fp1 = open("{}/1/wigner_probability_density_{}.dat".format(directory, i), 'r')
        fp2 = open("{}/2/wigner_probability_density_{}.dat".format(directory, i), 'r')
        fp3 = open("{}/3/wigner_probability_density_{}.dat".format(directory, i), 'r')
        maxDEN = 0
        maxX_i = 0
        if fp1 is False and fp2 is False and fp3 is False:
            print('error')
            exit(0)
        i = 0
        for line in fp1:
            a = float(line.strip())
            if a > maxDEN:
                maxDEN = a
                maxX_i = i
            i += 1
        XT.append(_XX[maxX_i])
    # plt.plot(TT, XX, label='Position')
    VV = []
    for i in range(len(TT) - 1):
        v = (XT[i] - XT[i + 1]) / (TT[i] - TT[i + 1])
        VV.append(v)
    plt.plot(TT[:-1], VV, label='Velocity')
    print(XT[-1], XT[1])
    plt.xlabel('Time')
    plt.legend()
    plt.ylabel('Velocity')
    plt.show()


def GetElectron(T):
    for index in range(T + 1):
        fp = open("{}/e/WPD_{}.dat".format(directory, index), 'w')
        fp1 = open("{}/1/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp2 = open("{}/2/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp3 = open("{}/3/wigner_probability_density_{}.dat".format(directory, index), 'r')

        if fp1 is False and fp2 is False and fp3 is False:
            print('error')
            exit(0)

        WPD = []

        for line in fp1:
            WPD.append(pow(PMNS[0][0], 2) * float(line.strip()))
        fp1.close()

        i = 0
        for line in fp2:
            WPD[i] += pow(PMNS[0][1], 2) * float(line.strip())
            i += 1
        fp2.close()

        i = 0
        for line in fp3:
            WPD[i] += pow(PMNS[0][2], 2) * float(line.strip())
            i += 1
        fp3.close()

        for i in range(len(WPD)):
            fp.write("{}\n".format(WPD[i]))
        fp.close()


def GetMuon(T):
    for index in range(T + 1):
        fp = open("{}/mu/WPD_{}.dat".format(directory, index), 'w')
        fp1 = open("{}/1/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp2 = open("{}/2/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp3 = open("{}/3/wigner_probability_density_{}.dat".format(directory, index), 'r')

        if fp1 is False and fp2 is False and fp3 is False:
            print('error')
            exit(0)

        WPD = []

        for line in fp1:
            WPD.append(pow(PMNS[1][0], 2) * float(line.strip()))
        fp1.close()

        i = 0
        for line in fp2:
            WPD[i] += pow(PMNS[1][1], 2) * float(line.strip())
            i += 1
        fp2.close()

        i = 0
        for line in fp3:
            WPD[i] += pow(PMNS[1][2], 2) * float(line.strip())
            i += 1
        fp3.close()

        for i in range(len(WPD)):
            fp.write("{}\n".format(WPD[i]))
        fp.close()


def GetTau(T):
    for index in range(T + 1):
        fp = open("{}/tau/WPD_{}.dat".format(directory, index), 'w')
        fp1 = open("{}/1/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp2 = open("{}/2/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp3 = open("{}/3/wigner_probability_density_{}.dat".format(directory, index), 'r')

        if fp1 is False and fp2 is False and fp3 is False:
            print('error')
            exit(0)

        WPD = []

        for line in fp1:
            WPD.append(pow(PMNS[2][0], 2) * float(line.strip()))
        fp1.close()

        i = 0
        for line in fp2:
            WPD[i] += pow(PMNS[2][1], 2) * float(line.strip())
            i += 1
        fp2.close()

        i = 0
        for line in fp3:
            WPD[i] += pow(PMNS[2][2], 2) * float(line.strip())
            i += 1
        fp3.close()

        for i in range(len(WPD)):
            fp.write("{}\n".format(WPD[i]))
        fp.close()


if __name__ == '__main__':
    getGrid()
    simulate(2, 8000)
    # simulate(1, 8000)
    # simulate(3, 8000)
    # plotV(2000)
    # GetElectron(8000)
    # GetMuon(8000)
    # GetTau(8000)
    # visualize(2 000)
    # plt.show()
