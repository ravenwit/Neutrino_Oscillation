#!/usr/bin/env python

"""
    Importing necessary libraries
"""

from __future__ import division
import time
import os.path
import numpy as np
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import splprep, splev
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d

interval = 100

fig = plt.figure()
ax = fig.add_subplot(aspect='equal', autoscale_on=False)

fig1 = plt.figure()

# directory = "Data"
directory = "../../Numerical/Neutrino Oscillation/Wigner Monte Carlo/Data"
plot_label = None

XX = []
_XX = []
NXX = 0
KK = []
NKK = 0

# PMNS Matrix
# theta12 = (np.pi / 180) * 35.26
# theta23 = (np.pi / 180) * 45
# theta13 = (np.pi / 180) * 13.2
# delta = np.pi / 4#
theta12 = (np.pi / 180) * 33.62
theta23 = (np.pi / 180) * 47.2
theta13 = (np.pi / 180) * 8.54
delta = (np.pi / 180)*234
c12 = np.cos(theta12)
c23 = np.cos(theta23)
c13 = np.cos(theta13)
s12 = np.sin(theta12)
s23 = np.sin(theta23)
s13 = np.sin(theta13)

# PMNS = [[c12 * c13, s12 * c13, s13 * np.exp(delta)],
#         [-s12 * c23 - c12 * s23 * s13 * np.exp(delta), c12 * c23 - s12 * s23 * s13 * np.exp(delta), s23 * c13],
#         [s12 * s23 - c12 * c23 * s13 * np.exp(delta), -c12 * s23 - s12 * c23 * s13 * np.exp(delta), c23 * c13]
#         ]

PMNS = [[c12 * c13, s12 * c13, s13 * np.complex(np.cos(-delta) + np.sin(-delta) * 1j)],
        [-s12 * c23 - c12 * s23 * s13 * np.complex(np.cos(delta) + np.sin(delta) * 1j),
         c12 * c23 - s12 * s23 * s13 * np.complex(np.cos(delta) + np.sin(delta) * 1j), s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * np.complex(np.cos(delta) + np.sin(delta) * 1j),
         -c12 * s23 - s12 * c23 * s13 * np.complex(np.cos(delta) + np.sin(delta) * 1j), c23 * c13]
        ]

theta = (np.pi / 180) * 45
PMNS2 = [[np.cos(theta), np.sin(theta)],
         [-np.sin(theta), np.cos(theta)]
         ]


def substractFunc(data, data_refer_point, refer_func, refer_func_point):
    out = data
    data_bound = [data_refer_point - refer_func_point,
                  data_refer_point + (len(refer_func) - refer_func_point) - 1]

    j = refer_func_point
    for i in range(data_refer_point, data_bound[1] + 1):
        if out[i] == 0:
            break
        out[i] = abs(out[i] - refer_func[j])
        j += 1
    for i in range(data_bound[1], len(data)):
        out[i] = 0

    j = 0
    for i in range(data_bound[0], data_refer_point):
        if out[i] == 0:
            break
        out[i] = abs(out[i] - refer_func[j])
        j += 1
    for i in range(0, data_bound[0]):
        out[i] = 0

    return out


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


def plotEssentials(figure, index, data):
    global _XX, plot_label
    plt.clf()
    axes = figure.gca()

    axes.set_xlim([min(_XX), max(_XX)])
    axes.set_ylim([0, max(data) + 0.02])
    axes.set_xlabel('Length')
    axes.set_ylabel('Probability Difference')

    plt.title('Time {}'.format(index))

    axes.autoscale(enable=False, tight=False)

    axes.plot(_XX, data, label=plot_label)

    axes.legend()
    # figure.show()


def GetElectron(T):
    for index in range(T):
        fp = open("{}/e/WPD_{}.dat".format(directory, index), 'w')
        fp1 = open("{}/1/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp2 = open("{}/2/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp3 = open("{}/3/wigner_probability_density_{}.dat".format(directory, index), 'r')

        if fp1 is False or fp2 is False or fp3 is False:
            print('error')
            exit(0)

        WPD = []

        for line in fp1:
            WPD.append(pow(abs(PMNS[0][0]), 2) * float(line.strip()))
        fp1.close()

        i = 0
        for line in fp2:
            WPD[i] += pow(abs(PMNS[0][1]), 2) * float(line.strip())
            i += 1
        fp2.close()

        i = 0
        for line in fp3:
            WPD[i] += pow(abs(PMNS[0][2]), 2) * float(line.strip())
            i += 1
        fp3.close()

        for i in range(len(WPD)):
            fp.write("{}\n".format(WPD[i]))
        fp.close()
    print('Success with getting electron')


def GetMuon(T):
    for index in range(T):
        fp = open("{}/mu/WPD_{}.dat".format(directory, index), 'w')
        fp1 = open("{}/1/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp2 = open("{}/2/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp3 = open("{}/3/wigner_probability_density_{}.dat".format(directory, index), 'r')

        if fp1 is False or fp2 is False or fp3 is False:
            print('error')
            exit(0)

        WPD = []

        for line in fp1:
            WPD.append(pow(abs(PMNS[1][0]), 2) * float(line.strip()))
        fp1.close()

        i = 0
        for line in fp2:
            WPD[i] += pow(abs(PMNS[1][1]), 2) * float(line.strip())
            i += 1
        fp2.close()

        i = 0
        for line in fp3:
            WPD[i] += pow(abs(PMNS[1][2]), 2) * float(line.strip())
            i += 1
        fp3.close()

        for i in range(len(WPD)):
            fp.write("{}\n".format(WPD[i]))
        fp.close()
    print('Success with getting muon')


def GetTau(T):
    for index in range(T):
        fp = open("{}/tau/WPD_{}.dat".format(directory, index), 'w')
        fp1 = open("{}/1/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp2 = open("{}/2/wigner_probability_density_{}.dat".format(directory, index), 'r')
        fp3 = open("{}/3/wigner_probability_density_{}.dat".format(directory, index), 'r')

        if fp1 is False or fp2 is False or fp3 is False:
            print('error')
            exit(0)

        WPD = []

        for line in fp1:
            WPD.append(pow(abs(PMNS[2][0]), 2) * float(line.strip()))
        fp1.close()

        i = 0
        for line in fp2:
            WPD[i] += pow(abs(PMNS[2][1]), 2) * float(line.strip())
            i += 1
        fp2.close()

        i = 0
        for line in fp3:
            WPD[i] += pow(abs(PMNS[2][2]), 2) * float(line.strip())
            i += 1
        fp3.close()

        for i in range(len(WPD)):
            fp.write("{}\n".format(WPD[i]))
        fp.close()

    print('Success with getting tau')


def GetDiff(T, type):
    dir_fp = None
    dir_fp1 = None
    dir_fp2 = None

    if type == 1:
        dir_fp = "{}/EMuDiff".format(directory)
        dir_fp1 = "{}/e".format(directory)
        dir_fp2 = "{}/mu".format(directory)
    elif type == 2:
        dir_fp = "{}/ETauDiff".format(directory)
        dir_fp1 = "{}/e".format(directory)
        dir_fp2 = "{}/tau".format(directory)
    elif type == 3:
        dir_fp = "{}/MuEDiff".format(directory)
        dir_fp1 = "{}/mu".format(directory)
        dir_fp2 = "{}/e".format(directory)
    elif type == 4:
        dir_fp = "{}/MuTauDiff".format(directory)
        dir_fp1 = "{}/mu".format(directory)
        dir_fp2 = "{}/tau".format(directory)
    elif type == 5:
        dir_fp = "{}/TauEDiff".format(directory)
        dir_fp1 = "{}/tau".format(directory)
        dir_fp2 = "{}/e".format(directory)
    elif type == 6:
        dir_fp = "{}/TauMuDiff".format(directory)
        dir_fp1 = "{}/tau".format(directory)
        dir_fp2 = "{}/mu".format(directory)
    elif type == 7:
        dir_fp = "{}/EEDiff".format(directory)
        dir_fp1 = "{}/e".format(directory)
        dir_fp2 = "{}/e".format(directory)
    elif type == 8:
        dir_fp = "{}/MuMuDiff".format(directory)
        dir_fp1 = "{}/mu".format(directory)
        dir_fp2 = "{}/mu".format(directory)
    elif type == 9:
        dir_fp = "{}/TauTauDiff".format(directory)
        dir_fp1 = "{}/tau".format(directory)
        dir_fp2 = "{}/tau".format(directory)
    else:
        print('error')
        exit(0)

    WPD = []
    fp1 = None
    try:
        fp1 = open("{}/WPD_0.dat".format(dir_fp1), 'r')
    except:
        print('error')
        exit(0)

    for line in fp1:
        if float(line.strip()) != 0:
            WPD.append(float(line.strip()))
    fp1.close()

    for index in range(T):

        fp = open("{}/WPD_{}.dat".format(dir_fp, index), 'w')

        fp2 = open("{}/WPD_{}.dat".format(dir_fp2, index), 'r')

        if fp2 is False or fp is False:
            print('error')
            exit(0)

        WPDD = []

        for line in fp2:
            WPDD.append(float(line.strip()))
        fp2.close()

        DIFF = substractFunc(WPDD, WPDD.index(max(WPDD)), WPD, WPD.index(max(WPD)))

        for i in range(len(DIFF)):
            fp.write("{}\n".format(DIFF[i]))
        fp.close()

    print("Success with {}".format(type))


def GetOscProb(T, type):
    global directory
    dir_diff = None
    if type == 1:
        dir_diff = "{}/EMuDiff".format(directory)
    elif type == 2:
        dir_diff = "{}/ETauDiff".format(directory)
    elif type == 3:
        dir_diff = "{}/MuEDiff".format(directory)
    elif type == 4:
        dir_diff = "{}/MuTauDiff".format(directory)
    elif type == 5:
        dir_diff = "{}/TauEDiff".format(directory)
    elif type == 6:
        dir_diff = "{}/TauMuDiff".format(directory)
    elif type == 7:
        dir_diff = "{}/EEDiff".format(directory)
    elif type == 8:
        dir_diff = "{}/MuMuDiff".format(directory)
    elif type == 9:
        dir_diff = "{}/TauTauDiff".format(directory)
    else:
        print('error')
        exit(0)

    fp = open("{}/osc.dat".format(dir_diff), 'w')

    for index in range(T):

        fp_diff = open("{}/WPD_{}.dat".format(dir_diff, index), 'r')

        if fp is False or fp_diff is False:
            print('error')
            exit(0)

        i = 0
        mean = 0
        WPD = []
        for line in fp_diff:
            # mean += float(line.strip())
            WPD.append(float(line.strip()))
        # mean /= (i + 1)
        # _mean = min(WPD, key=lambda x: abs(x - mean))
        # mean_index = WPD.index(_mean)
        mean = max(WPD)
        mean_index = WPD.index(mean)
        expect = _XX[mean_index]
        fp_diff.close()

        fp.write("{} {}\n".format(expect, mean))
        # print("{} {}\n".format(expect, mean))

    fp.close()

    print('Sucess with {}'.format(type))


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

    if fp1 is False or fp2 is False or fp3 is False:
        print('error')
        exit(0)

    WPD = []

    for line in fp1:
        WPD.append(pow(abs(PMNS[2][0]), 2) * float(line.strip()))
    fp1.close()

    i = 0
    for line in fp2:
        WPD[i] += pow(abs(PMNS[2][1]), 2) * float(line.strip())
        i += 1
    fp2.close()

    i = 0
    for line in fp3:
        WPD[i] += pow(abs(PMNS[2][2]), 2) * float(line.strip())
        i += 1
    fp3.close()

    plotEssentials(fig, index, WPD)
    # plt.savefig("{}/figures/Electron/{}.png".format(directory, index))


def visualizeDiff(index, type):
    global _XX, fig
    index *= interval
    fp = None
    if type == 1:
        fp = open("{}/EMuDiff/WPD_{}.dat".format(directory, index), 'r')
    elif type == 2:
        fp = open("{}/ETauDiff/WPD_{}.dat".format(directory, index), 'r')
    elif type == 3:
        fp = open("{}/MuEDiff/WPD_{}.dat".format(directory, index), 'r')
    elif type == 4:
        fp = open("{}/MuTauDiff/WPD_{}.dat".format(directory, index), 'r')
    elif type == 5:
        fp = open("{}/TauEDiff/WPD_{}.dat".format(directory, index), 'r')
    elif type == 6:
        fp = open("{}/TauMuDiff/WPD_{}.dat".format(directory, index), 'r')
    elif type == 7:
        fp = open("{}/EEDiff/WPD_{}.dat".format(directory, index), 'r')
    elif type == 8:
        fp = open("{}/MuMuDiff/WPD_{}.dat".format(directory, index), 'r')
    elif type == 9:
        fp = open("{}/TauTauDiff/WPD_{}.dat".format(directory, index), 'r')

    else:
        print('error')
        exit(0)

    if fp is False:
        print('error')
        exit(0)

    WPDD = []

    for line in fp:
        WPDD.append(float(line.strip()))
    fp.close()

    plotEssentials(fig, index, WPDD)

    # if type == 1:
    #     plt.savefig("{}/figures/EMuDiff/{}.png".format(directory, index))
    # elif type == 2:
    #     plt.savefig("{}/figures/ETauDiff/{}.png".format(directory, index))
    # elif type == 3:
    #     plt.savefig("{}/figures/MuEDiff/{}.png".format(directory, index))
    # elif type == 4:
    #     plt.savefig("{}/figures/MuTauDiff/{}.png".format(directory, index))
    # elif type == 5:
    #     plt.savefig("{}/figures/TauEDiff/{}.png".format(directory, index))
    # elif type == 6:
    #     plt.savefig("{}/figures/TauMuDiff/{}.png".format(directory, index))
    # elif type == 7:
    #     plt.savefig("{}/figures/EEDiff/{}.png".format(directory, index))
    # elif type == 8:
    #     plt.savefig("{}/figures/MuMuDiff/{}.png".format(directory, index))
    # elif type == 9:
    #     plt.savefig("{}/figures/TauTauDiff/{}.png".format(directory, index))


def visualizeOscProb(type, smooth_factor):
    global _XX, plot_label
    fp = None
    fp_r = None
    if type == 1:
        fp = open("{}/EMuDiff/osc.dat".format(directory), 'r')
        fp_r = open("{}/e/WPD_0.dat".format(directory), 'r')
        plot_label = r'$P(\nu_e(0)\/ \rightarrow \/ \nu_\mu(t))$'
    elif type == 2:
        fp = open("{}/ETauDiff/osc.dat".format(directory), 'r')
        fp_r = open("{}/e/WPD_0.dat".format(directory), 'r')
        plot_label = r'$P(\nu_e(0)\/ \rightarrow \/ \nu_\tau(t))$'
    elif type == 3:
        fp = open("{}/MuEDiff/osc.dat".format(directory), 'r')
        fp_r = open("{}/mu/WPD_0.dat".format(directory), 'r')
        plot_label = r'$P(\nu_\mu(0)\/ \rightarrow \/ \nu_e(t))$'
    elif type == 4:
        fp = open("{}/MuTauDiff/osc.dat".format(directory), 'r')
        fp_r = open("{}/mu/WPD_0.dat".format(directory), 'r')
        plot_label = r'$P(\nu_\mu(0)\/ \rightarrow \/ \nu_\tau(t))$'
    elif type == 5:
        fp = open("{}/TauEDiff/osc.dat".format(directory), 'r')
        fp_r = open("{}/tau/WPD_0.dat".format(directory), 'r')
        plot_label = r'$P(\nu_\tau(0)\/ \rightarrow \/ \nu_e(t))$'
    elif type == 6:
        fp = open("{}/TauMuDiff/osc.dat".format(directory), 'r')
        fp_r = open("{}/tau/WPD_0.dat".format(directory), 'r')
        plot_label = r'$P(\nu_\tau(0)\/ \rightarrow \/ \nu_\mu(t))$'
    elif type == 7:
        fp = open("{}/EEDiff/osc.dat".format(directory), 'r')
        fp_r = open("{}/e/WPD_0.dat".format(directory), 'r')
        plot_label = r'$P(\nu_e(0)\/ \rightarrow \/ \nu_e(t))$'
    elif type == 8:
        fp = open("{}/MuMuDiff/osc.dat".format(directory), 'r')
        fp_r = open("{}/mu/WPD_0.dat".format(directory), 'r')
        plot_label = r'$P(\nu_\mu(0)\/ \rightarrow \/ \nu_\mu(t))$'
    elif type == 9:
        fp = open("{}/TauTauDiff/osc.dat".format(directory), 'r')
        fp_r = open("{}/e/WPD_0.dat".format(directory), 'r')
        plot_label = r'$P(\nu_\tau(0)\/ \rightarrow \/ \nu_\tau(t))$'
    else:
        print('error')
        exit(0)

    if fp is False or fp_r is False:
        print('error')
        exit(0)

    Refer_WPD = []
    for line in fp_r:
        Refer_WPD.append(float(line.strip()))
    MAX_PROB = max(Refer_WPD)

    OSCWPD = []
    expect = []
    prob_range = [0, 1]
    if type == 7 or type == 8 or type == 9:
        prob_range = [1, 0]

    for line in fp:
        comp = line.strip().split(' ')
        expect.append(float(comp[0]))
        OSCWPD.append(float(np.interp(float(comp[1]), [0, MAX_PROB], prob_range)))

    max_expect = max(expect)
    x_input = np.linspace(0, max_expect, len(OSCWPD))

    # xnew = np.linspace(0, max_expect, 50)
    #
    # spl = make_interp_spline(x_input, OSCWPD, k=3)  # type: BSpline
    # power_smooth = spl(xnew)

    ysmoothed = gaussian_filter1d(OSCWPD, sigma=smooth_factor)

    plt.xlabel("Length (L)")
    # plt.ylabel(r'$P(\nu_e(0)\/ \rightarrow \/ \nu_\mu(t))$')
    # plt.plot(x_input, OSCWPD, label=plot_label)
    plt.plot(x_input, ysmoothed, label=plot_label)
    # plt.plot(x_input, ysmoothed)

    plt.legend()

    if type == 1:
        plt.savefig("{}/figures/Oscillation/EMu.png".format(directory))
    if type == 2:
        plt.savefig("{}/figures/Oscillation/ETau.png".format(directory))
    if type == 3:
        plt.savefig("{}/figures/Oscillation/MuE.png".format(directory))
    if type == 4:
        plt.savefig("{}/figures/Oscillation/MuTau.png".format(directory))
    if type == 5:
        plt.savefig("{}/figures/Oscillation/TauE.png".format(directory))
    if type == 6:
        plt.savefig("{}/figures/Oscillation/TauMu.png".format(directory))
    if type == 7:
        plt.savefig("{}/figures/Oscillation/EE.png".format(directory))
    if type == 8:
        plt.savefig("{}/figures/Oscillation/MuMu.png".format(directory))
    if type == 9:
        plt.savefig("{}/figures/Oscillation/TauTau.png".format(directory))

    # plt.show()


def visualizeParticle(index, type):
    global _XX, fig
    index *= interval
    fp = None
    if type.lower() == "electron":
        fp = open("{}/e/WPD_{}.dat".format(directory, index), 'r')
    elif type.lower() == "mu":
        fp = open("{}/mu/WPD_{}.dat".format(directory, index), 'r')
    elif type.lower() == "tau":
        fp = open("{}/tau/WPD_{}.dat".format(directory, index), 'r')

    if fp is False:
        print('error')
        exit(0)

    WP = []

    for line in fp:
        WP.append(float(line.strip()))
    fp.close()

    plotEssentials(fig, index, WP)

    # if type.lower() == "electron":
    #     plt.savefig("{}/figures/Electron/{}.png".format(directory, index))
    # elif type.lower() == "mu":
    #     plt.savefig("{}/figures/Muon/{}.png".format(directory, index))
    # elif type.lower() == "tau":
    #     plt.savefig("{}/figures/Tau/{}.png".format(directory, index))


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
    # fig.show()
    # plt.savefig("{}/figures/Every/{}.png".format(directory, index))


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
    # plt.savefig('{}/figures/quasi/{}.png'.format(directory, index))


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
            WQP[i][j] = pow(abs(PMNS[0][0]), 2) * float(value)
            j += 1
        i += 1
    fp1.close()
    i = 0
    for line in fp2:
        line = line.split(' ')[:-1]
        j = 0
        for value in line:
            WQP[i][j] += pow(abs(PMNS[0][1]), 2) * float(value)
            j += 1
        i += 1
    fp2.close()
    i = 0
    for line in fp3:
        line = line.split(' ')[:-1]
        j = 0
        for value in line:
            WQP[i][j] += pow(abs(PMNS[0][2]), 2) * float(value)
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
    # plt.savefig('{}/figures/quasi/{}.png'.format(directory, index))


def visualizeWaterfall(type, max_index, timestep):
    global _XX
    fp_dir = None
    fp = None
    plot_name = None
    if type == 1:
        fp_dir = "{}/e".format(directory)
        plot_name = "Electron_{}_{}.png".format(max_index, timestep)
    elif type == 2:
        fp_dir = "{}/mu".format(directory)
        plot_name = "Muon_{}_{}.png".format(max_index, timestep)
    elif type == 3:
        fp_dir = "{}/tau".format(directory)
        plot_name = "Tau_{}_{}.png".format(max_index, timestep)
    elif type == 4:
        fp_dir = "{}/EMuDiff".format(directory)
        plot_name = "Electron-Muon_{}_{}.png".format(max_index, timestep)
    elif type == 5:
        fp_dir = "{}/ETauDiff".format(directory)
        plot_name = "Electron-Tau_{}_{}.png".format(max_index, timestep)
    elif type == 6:
        fp_dir = "{}/MuEDiff".format(directory)
        plot_name = "Muon-Electron_{}_{}.png".format(max_index, timestep)
    elif type == 7:
        fp_dir = "{}/MuTauDiff".format(directory)
        plot_name = "Muon-Tau_{}_{}.png".format(max_index, timestep)
    elif type == 8:
        fp_dir = "{}/TauEDiff".format(directory)
        plot_name = "Tau-Electron_{}_{}.png".format(max_index, timestep)
    elif type == 9:
        fp_dir = "{}/TauMuDiff".format(directory)
        plot_name = "Tau-Muon_{}_{}.png".format(max_index, timestep)
    elif type == 10:
        fp_dir = "{}/EEDiff".format(directory)
        plot_name = "Electron-Electron_{}_{}.png".format(max_index, timestep)
    elif type == 11:
        fp_dir = "{}/MuMuDiff".format(directory)
        plot_name = "Muon-Muon_{}_{}.png".format(max_index, timestep)
    elif type == 12:
        fp_dir = "{}/TauTauDiff".format(directory)
        plot_name = "Tau-Tau_{}_{}.png".format(max_index, timestep)

    data = []
    sum = 0
    timespace_width = 0

    for i in range(timestep):
        try:
            fp = open("{}/WPD_{}.dat".format(fp_dir, i * int(max_index / timestep)), 'r')
        except:
            print('error')
            exit(0)
        data_comp = []
        for line in fp:
            data_comp.append(float(line.strip()))
        sum += max(data_comp)
        timespace_width = max(data_comp) * 2
        data.append(data_comp)

    x_lim = (timestep - 1) * timespace_width

    plotWaterfall(data, _XX, timespace_width, x_lim, plot_name)


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


def plotWaterfall(data, spatialspace, timespace_width, x_lim, fig_name):
    time_index = len(data)

    plt.xlim([min(spatialspace), 175])
    plt.yticks(list(np.linspace(0, x_lim, time_index)), [(i + 1) for i in range(time_index)])
    plt.xlabel("Length (Spatial index)")
    plt.ylabel("Time (Frame number)")
    plt.title('Flavor Wave Packet Progression')

    for i in range(time_index):
        if i != 0:
            for j in range(len(data[i])):
                data[i][j] += timespace_width * i
        plt.plot(spatialspace, data[i], color='green')

    plt.savefig("{}/figures/Waterfall/{}".format(directory, fig_name))
    plt.show()


def simulate(type, T):
    global plot_label
    T = int(T / interval)
    if type == 1:
        plot_label = "General"
        time0 = time.time()
        visualize(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualize, T,
                                 interval=duration, blit=False)

    elif type == 2:
        plot_label = "Quasi Distribution"
        time0 = time.time()
        visualizeQUASI(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig1, visualizeQUASI, T,
                                 interval=duration, blit=False)
    elif type == 3:
        plot_label = "Electron"
        time0 = time.time()
        visualizeParticle(1, 'electron')
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeParticle, T,
                                 interval=duration, fargs=['electron'], blit=False)
    elif type == 4:
        plot_label = "Muon"
        time0 = time.time()
        visualizeParticle(1, 'mu')
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeParticle, T,
                                 interval=duration, fargs=['mu'], blit=False)
    elif type == 5:
        plot_label = "Tau"
        time0 = time.time()
        visualizeParticle(1, 'tau')
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeParticle, T,
                                 interval=duration, fargs=['tau'], blit=False)
    elif type == 6:
        plot_label = "Specify in code"
        time0 = time.time()
        visualizeSuper(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeSuper, T,
                                 interval=duration, blit=False)
    elif type == 7:
        plot_label = r'$\nu_e(0)\/ \rightarrow \/ \nu_\mu(t)$'
        time0 = time.time()
        visualizeDiff(1, 1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeDiff, T,
                                 interval=duration, fargs=[1], blit=False)
    elif type == 8:
        plot_label = r'$\nu_e(0)\/ \rightarrow \/ \nu_\tau(t)$'
        time0 = time.time()
        visualizeDiff(1, 2)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeDiff, T,
                                 interval=duration, fargs=[2], blit=False)
    elif type == 9:
        plot_label = r'$\nu_\mu(0)\/ \rightarrow \/ \nu_e(t)$'
        time0 = time.time()
        visualizeDiff(1, 3)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeDiff, T,
                                 interval=duration, fargs=[3], blit=False)
    elif type == 10:
        plot_label = r'$\nu_\mu(0)\/ \rightarrow \/ \nu_\tau(t)$'
        time0 = time.time()
        visualizeDiff(1, 4)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeDiff, T,
                                 interval=duration, fargs=[4], blit=False)
    elif type == 11:
        plot_label = r'$\nu_\tau(0)\/ \rightarrow \/ \nu_e(t)$'
        time0 = time.time()
        visualizeDiff(1, 5)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeDiff, T,
                                 interval=duration, fargs=[5], blit=False)
    elif type == 12:
        plot_label = r'$\nu_\tau(0)\/ \rightarrow \/ \nu_\mu(t)$'
        time0 = time.time()
        visualizeDiff(1, 6)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeDiff, T,
                                 interval=duration, fargs=[6], blit=False)
    elif type == 13:
        plot_label = r'$\nu_e(0)\/ \rightarrow \/ \nu_e(t)$'
        time0 = time.time()
        visualizeDiff(1, 7)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeDiff, T,
                                 interval=duration, fargs=[7], blit=False)
    elif type == 14:
        plot_label = r'$\nu_\mu(0)\/ \rightarrow \/ \nu_\mu(t)$'
        time0 = time.time()
        visualizeDiff(1, 8)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeDiff, T,
                                 interval=duration, fargs=[8], blit=False)
    elif type == 15:
        plot_label = r'$\nu_\tau(0)\/ \rightarrow \/ \nu_\tau(t)$'
        time0 = time.time()
        visualizeDiff(1, 9)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeDiff, T,
                                 interval=duration, fargs=[9], blit=False)

    elif type == 16:
        time0 = time.time()
        visualizeEvery(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeEvery, T,
                                 interval=duration, blit=False)
    elif type == 17:
        time0 = time.time()
        visualizeQUASIF(1)
        time1 = time.time()
        duration = 4000 * (1 / 60) - float(time1 - time0)

        ani = anim.FuncAnimation(fig, visualizeQUASIF, T,
                                 interval=duration, blit=False)

    plt.show()


if __name__ == '__main__':
    getGrid()

    # visualizeWaterfall(2, 8000, 7)

    # simulate(1, 8000)  ## Wigner Probability Density
    # simulate(2, 8000)  ## Wigner Quasi Distribution
    # simulate(3, 8000)  ## Electron
    # simulate(4, 8000)  ## Muon
    # simulate(5, 8000)  ## Tau
    # simulate(6, 8000)  ## Superposed (Useless)
    # simulate(7, 8000)  ## Difference E->Mu
    # simulate(8, 8000)  ## Difference E->Tau
    # simulate(9, 8000)  ## Difference Mu->E
    # simulate(10, 8000)  ## Difference Mu->Tau
    # simulate(11, 8000)  ## Difference Tau->E
    # simulate(12, 8000)  ## Difference Tau->Mu
    # simulate(13, 8000)  ## Difference E->E
    # simulate(14, 8000)  ## Difference Mu->Mu
    # simulate(15, 8000)  ## Difference Tau->Tau
    # simulate(16, 8000)  ## Every
    # simulate(17, 8000)  ## Wigner Quasi 3-D

    # GetElectron(8000)  ## Get Electron
    # GetMuon(8000)  ## Get Muon
    # GetTau(8000)  ## Get Tau

    # GetDiff(8000, 1)  ## Difference E->Mu
    # GetDiff(8000, 2)  ## Difference E->Tau
    # GetDiff(8000, 3)  ## Difference Mu->E
    # GetDiff(8000, 4)  ## Difference Mu->Tau
    # GetDiff(8000, 5)  ## Difference Tau->E
    # GetDiff(8000, 6)  ## Difference Tau->Mu
    # GetDiff(8000, 7)  ## Difference E->E
    # GetDiff(8000, 8)  ## Difference Mu->Mu
    # GetDiff(8000, 9)  ## Difference Tau->Tau

    # GetOscProb(8000, 1)  ## Transition E->Mu
    # GetOscProb(8000, 2)  ## Transition E->Tau
    # GetOscProb(8000, 3)  ## Transition Mu->E
    # GetOscProb(8000, 4)  ## Transition Mu->Tau
    # GetOscProb(8000, 5)  ## Transition Tau->E
    # GetOscProb(8000, 6)  ## Transition Tau->Mu
    # GetOscProb(8000, 7)  ## Transition E->E
    # GetOscProb(8000, 8)  ## Transition Mu->Mu
    # GetOscProb(8000, 9)  ## Transition Tau->Tau

    visualizeOscProb(1, 3)  # Transition E->Mu
    visualizeOscProb(2, 3)  # Transition E->Tau
    visualizeOscProb(3, 3)  # Transition Mu->E
    visualizeOscProb(4, 3)  # Transition Mu->Tau
    visualizeOscProb(5, 3)  # Transition Tau->E
    visualizeOscProb(6, 3)  # Transition Tau->Mu
    visualizeOscProb(7, 3)  # Transition E->E
    visualizeOscProb(8, 3)  # Transition Mu->Mu
    visualizeOscProb(9, 3)  # Transition Tau->Tau

    plt.show()
