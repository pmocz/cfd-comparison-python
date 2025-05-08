import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

"""
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz 2023, @PMocz

Simulate the compressible Euler equations

Method based on:
https://philip-mocz.medium.com/create-your-own-smoothed-particle-hydrodynamics-simulation-with-python-76e1cec505f1

see link for details

"""


def W(x, y, h):
    """
    Gausssian Smoothing kernel (2D)
        x     is a vector/matrix of x positions
        y     is a vector/matrix of y positions
        h     is the smoothing length
        w     is the evaluated smoothing function
    """

    r = np.sqrt(x**2 + y**2)

    w = (1.0 / (h * np.sqrt(np.pi))) ** 2 * np.exp(-(r**2) / h**2)

    return w


def gradW(x, y, h):
    """
    Gradient of the Gausssian Smoothing kernel (2D)
    x     is a vector/matrix of x positions
    y     is a vector/matrix of y positions
    h     is the smoothing length
    wx, wy     is the evaluated gradient
    """

    r = np.sqrt(x**2 + y**2)

    n = -2 * np.exp(-(r**2) / h**2) / h**4 / np.pi
    wx = n * x
    wy = n * y

    return wx, wy


def getPairwiseSeparations(ri, rj):
    """
    Get pairwise desprations between 2 sets of coordinates
    ri    is an M x 2 matrix of positions
    rj    is an N x 2 matrix of positions
    dx, dy   are M x N matrices of separations
    """

    M = ri.shape[0]
    N = rj.shape[0]

    # positions ri = (x,y)
    rix = ri[:, 0].reshape((M, 1))
    riy = ri[:, 1].reshape((M, 1))

    # other set of points positions rj = (x,y)
    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))

    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T
    dy = riy - rjy.T

    # periodic domain
    dx[dx <= -0.5] = dx[dx <= -0.5] + 1
    dy[dy <= -0.5] = dy[dy <= -0.5] + 1
    dx[dx > 0.5] = dx[dx > 0.5] - 1
    dy[dy > 0.5] = dy[dy > 0.5] - 1

    return dx, dy


def getDensity(dx, dy, m, h):
    """
    Get Density at sampling locations from SPH particle distribution
    dx, dy   are N x N matrices of separations
    m     is the particle mass
    h     is the smoothing length
    rho   is N x 1 vector of densities
    """

    N = dx.shape[0]

    rho = m * np.sum(W(dx, dy, h), 1).reshape((N, 1))

    return rho


def getAcc(pos, vel, rho, m, h, dx, dy):
    """
    Calculate the acceleration on each SPH particle
    pos   is an N x 2 matrix of positions
    vel   is an N x 2 matrix of velocities
    rho   is N x 1 vector of densities
    m     is the particle mass
    h     is the smoothing length
    dx, dy   are N x N matrices of separations
    a     is N x 2 matrix of accelerations
    """

    N = pos.shape[0]

    # Get the pressures
    P = rho

    # Get pairwise distances and gradients
    dWx, dWy = gradW(dx, dy, h)

    # Calculate/add  artificial viscosity (Monaghan 1992)
    # alpha = 1.
    # beta = 2.
    # etaSq = 0.01 * h * h
    # vx = vel[:,0].reshape((N,1))
    # vy = vel[:,1].reshape((N,1))
    # v_dot_r = (vx - vx.T) * dx + (vy - vy.T) * dy
    # mu = h * v_dot_r / ( dx*dx+dy*dy + etaSq )
    # mu[v_dot_r > 0] = 0
    # fac = 0
    # Pi = (-alpha * mu + beta * mu * mu) / (0.5*(rho + rho.T))
    Pi = 0

    # Add Pressure contribution to accelerations
    ax = -m * np.sum((P / rho**2 + P.T / rho.T**2 + Pi) * dWx, 1).reshape((N, 1))
    ay = -m * np.sum((P / rho**2 + P.T / rho.T**2 + Pi) * dWy, 1).reshape((N, 1))

    # pack together the acceleration components
    a = np.hstack((ax, ay))

    return a


def main():
    """SPH simulation"""

    # TODO: add adaptive smoothing h
    # TODO: add compact kernel
    # TODO: add artificial viscosity
    # TODO: speed-up neighbor-search to O(N log N) instead of brute-force O(N^2)
    # WARNING: without these features, code may not be able to handle loong-term evolution of this flow

    # Simulation parameters
    N = 50**2  # Number of particles
    t = 0  # current time of the simulation
    tEnd = 1 / np.sqrt(3)  # time at which simulation ends
    Nout = 100  # number of frames to draw
    saveFrames = False  # save frames to create movie

    Nlin = int(np.sqrt(N))
    h = np.sqrt(2) / Nlin  # smoothing length

    # Generate Initial Conditions
    Lbox = 1.0  # box size
    M = 1  # total mass

    dt = 0.3 * h  # timestep

    dx = Lbox / Nlin
    xlin = np.linspace(0.5 * dx, Lbox - 0.5 * dx, Nlin)
    X, Y = np.meshgrid(xlin, xlin)

    vx = 0.5 * np.sin(2 * np.pi * Y)
    vy = 0.1 * np.sin(4 * np.pi * X) * np.cos(2 * np.pi * Y) ** 2

    m = M / N  # single particle mass
    pos = np.reshape(np.array([X.flatten(), Y.flatten()]).T, (Nlin * Nlin, 2))
    vel = np.reshape(np.array([vx.flatten(), vy.flatten()]).T, (Nlin * Nlin, 2))

    # calculate initial acceleration
    dx, dy = getPairwiseSeparations(pos, pos)
    rho = getDensity(dx, dy, m, h)
    acc = getAcc(pos, vel, rho, m, h, dx, dy)

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=100)
    cmap = plt.cm.bwr
    cmap.set_bad("LightGray")
    outputCount = 1

    # Simulation Main Loop
    while t < tEnd:
        dt = 0.3 * h
        plotThisTurn = False
        if t + dt > outputCount * tEnd / Nout:
            dt = outputCount * tEnd / Nout - t
            plotThisTurn = True

        # (1/2) kick
        vel += acc * dt / 2

        # drift
        pos += vel * dt

        # apply periodic BCs
        pos = pos % Lbox

        # update accelerations
        dx, dy = getPairwiseSeparations(pos, pos)
        rho = getDensity(dx, dy, m, h)
        acc = getAcc(pos, vel, rho, m, h, dx, dy)

        # (1/2) kick
        vel += acc * dt / 2

        # update time
        t += dt
        print(t)

        # plot in real time
        if plotThisTurn:
            plt.cla()
            cval = rho.flatten()
            plt.scatter(pos[:, 0], pos[:, 1], c=cval, cmap=cmap, s=16, alpha=1)
            plt.clim(0.85, 1.15)
            ax = plt.gca()
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            if saveFrames:
                plt.text(
                    0.5 * Lbox,
                    0.9 * Lbox,
                    "SPH",
                    fontsize=20,
                    horizontalalignment="center",
                )
                ax.set(facecolor="LightGray")
                plt.savefig(
                    "tmp/sph%03d.png" % (outputCount - 1),
                    dpi=100,
                    bbox_inches="tight",
                    pad_inches=0,
                )
            outputCount += 1
            plt.pause(0.001)

    # Save figure
    plt.savefig("sph.png", dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
