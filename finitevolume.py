import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Finite Volume Fluid Simulation (With Python)
Philip Mocz 2023, @PMocz

Simulate the compressible Euler equations

Method based on:
https://levelup.gitconnected.com/create-your-own-finite-volume-fluid-simulation-with-python-8f9eab0b8305

see link for details

"""

R = -1  # right
L = 1  # left
aX = 1  # x-axis
aY = 0  # y-axis


def getConserved(rho, vx, vy, vol):
    """
    Calculate the conserved variable from the primitive
        rho      is matrix of cell densities
        vx       is matrix of cell x-velocity
        vy       is matrix of cell y-velocity
        vol      is cell volume
        Mass     is matrix of mass in cells
        Momx     is matrix of x-momentum in cells
        Momy     is matrix of y-momentum in cells
    """
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol

    return Mass, Momx, Momy


def getPrimitive(Mass, Momx, Momy, vol):
    """
    Calculate the primitive variable from the conservative
        Mass     is matrix of mass in cells
        Momx     is matrix of x-momentum in cells
        Momy     is matrix of y-momentum in cells
        vol      is cell volume
        rho      is matrix of cell densities
        vx       is matrix of cell x-velocity
        vy       is matrix of cell y-velocity
    """
    rho = Mass / vol
    vx = Momx / rho / vol
    vy = Momy / rho / vol

    return rho, vx, vy


def getGradient(f, dx):
    """
    Calculate the gradients of a field
        f        is a matrix of the field
        dx       is the cell size
        f_dx     is a matrix of derivative of f in the x-direction
        f_dy     is a matrix of derivative of f in the y-direction
    """

    f_dx = (np.roll(f, R, axis=aX) - np.roll(f, L, axis=aX)) / (2 * dx)
    f_dy = (np.roll(f, R, axis=aY) - np.roll(f, L, axis=aY)) / (2 * dx)

    return f_dx, f_dy


def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
    """
    Calculate the gradients of a field
        f        is a matrix of the field
        f_dx     is a matrix of the field x-derivatives
        f_dy     is a matrix of the field y-derivatives
        dx       is the cell size
        f_XL     is a matrix of spatial-extrapolated values on `left' face along x-axis
        f_XR     is a matrix of spatial-extrapolated values on `right' face along x-axis
        f_YL     is a matrix of spatial-extrapolated values on `left' face along y-axis
        f_YR     is a matrix of spatial-extrapolated values on `right' face along y-axis
    """

    f_XL = f - f_dx * dx / 2
    f_XL = np.roll(f_XL, R, axis=aX)
    f_XR = f + f_dx * dx / 2

    f_YL = f - f_dy * dx / 2
    f_YL = np.roll(f_YL, R, axis=aY)
    f_YR = f + f_dy * dx / 2

    return f_XL, f_XR, f_YL, f_YR


def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """
    Apply fluxes to conserved variables
        F        is a matrix of the conserved variable field
        flux_F_X is a matrix of the x-dir fluxes
        flux_F_Y is a matrix of the y-dir fluxes
        dx       is the cell size
        dt       is the timestep
    """

    # update solution
    F += -dt * dx * flux_F_X
    F += dt * dx * np.roll(flux_F_X, L, axis=aX)
    F += -dt * dx * flux_F_Y
    F += dt * dx * np.roll(flux_F_Y, L, axis=aY)

    return F


def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R):
    """
    Calculate fluxed between 2 states with local Lax-Friedrichs/Rusanov rule
        rho_L        is a matrix of left-state  density
        rho_R        is a matrix of right-state density
        vx_L         is a matrix of left-state  x-velocity
        vx_R         is a matrix of right-state x-velocity
        vy_L         is a matrix of left-state  y-velocity
        vy_R         is a matrix of right-state y-velocity
        flux_Mass    is the matrix of mass fluxes
        flux_Momx    is the matrix of x-momentum fluxes
        flux_Momy    is the matrix of y-momentum fluxes
    """

    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)

    P_star = rho_star

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star
    flux_Momy = momx_star * momy_star / rho_star

    # find wavespeeds
    C_L = 1 + np.abs(vx_L)
    C_R = 1 + np.abs(vx_R)
    C = np.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)

    return flux_Mass, flux_Momx, flux_Momy


def main():
    """Finite Volume simulation"""

    # Simulation parameters
    N = 100  # resolution
    courant_fac = 0.5  # Courant factor
    t = 0  # current time of the simulation
    tEnd = 1 / np.sqrt(3)  # time at which simulation ends
    Nout = 100  # number of frames to draw
    saveFrames = False  # save frames to create movie

    # Mesh
    Lbox = 1.0  # box size
    dx = Lbox / N
    vol = dx**2
    xlin = np.linspace(0.5 * dx, Lbox - 0.5 * dx, N)
    X, Y = np.meshgrid(xlin, xlin)

    # Generate Initial Conditions
    rho = 0 * X + 1
    vx = 0.5 * np.sin(2 * np.pi * Y)
    vy = 0.1 * np.sin(4 * np.pi * X) * np.cos(2 * np.pi * Y) ** 2

    # Get conserved variables
    Mass, Momx, Momy = getConserved(rho, vx, vy, vol)

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=100)
    cmap = plt.cm.bwr
    cmap.set_bad("LightGray")
    outputCount = 1

    # Simulation Main Loop
    while t < tEnd:
        # get Primitive variables
        rho, vx, vy = getPrimitive(Mass, Momx, Momy, vol)

        # get time step (CFL) = dx / max signal speed
        dt = courant_fac * np.min(dx / (1 + np.sqrt(vx**2 + vy**2)))
        plotThisTurn = False
        if t + dt > outputCount * tEnd / Nout:
            dt = outputCount * tEnd / Nout - t
            plotThisTurn = True

        # calculate gradients
        rho_dx, rho_dy = getGradient(rho, dx)
        vx_dx, vx_dy = getGradient(vx, dx)
        vy_dx, vy_dy = getGradient(vy, dx)
        P_dx = rho_dx
        P_dy = rho_dy

        # extrapolate half-step in time
        rho_prime = rho - 0.5 * dt * (
            vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy
        )
        vx_prime = vx - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1 / rho) * P_dx)
        vy_prime = vy - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1 / rho) * P_dy)

        # extrapolate in space to face centers
        rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(
            rho_prime, rho_dx, rho_dy, dx
        )
        vx_XL, vx_XR, vx_YL, vx_YR = extrapolateInSpaceToFace(
            vx_prime, vx_dx, vx_dy, dx
        )
        vy_XL, vy_XR, vy_YL, vy_YR = extrapolateInSpaceToFace(
            vy_prime, vy_dx, vy_dy, dx
        )

        # compute fluxes (local Lax-Friedrichs/Rusanov)
        flux_Mass_X, flux_Momx_X, flux_Momy_X = getFlux(
            rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR
        )
        flux_Mass_Y, flux_Momy_Y, flux_Momx_Y = getFlux(
            rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR
        )

        # update solution
        Mass = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
        Momx = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
        Momy = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)

        # update time
        t += dt

        # plot in real time
        if plotThisTurn:
            plt.cla()
            plot_field = 1.0 * rho
            if saveFrames:
                # fancy plot to illustrate FV concept
                plot_field[:, 0 :: int(N / 10)] = np.nan
                plot_field[0 :: int(N / 10), :] = np.nan
            plt.imshow(plot_field, cmap=cmap)
            plt.clim(0.85, 1.15)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            if saveFrames:
                plt.text(
                    0.5 * N,
                    0.9 * N,
                    "Finite Volume",
                    fontsize=20,
                    horizontalalignment="center",
                )
                plt.savefig(
                    "tmp/fv%03d.png" % (outputCount - 1),
                    dpi=100,
                    bbox_inches="tight",
                    pad_inches=0,
                )
            outputCount += 1
            plt.pause(0.001)

    # Save figure
    plt.savefig("finitevolume.png", dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
