import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Spectral Simulation (With Python)
Philip Mocz 2023, @PMocz

Simulate the compressible Navier-Stokes equations

Method based on:
https://levelup.gitconnected.com/create-your-own-navier-stokes-spectral-method-fluid-simulation-with-python-3f37405524f4

see link for details

"""

# fancy plot
mask = [
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
]


def grad(v, kx, ky):
    """return gradient of v"""
    v_hat = np.fft.fftn(v)
    dvx = np.real(np.fft.ifftn(1j * kx * v_hat))
    dvy = np.real(np.fft.ifftn(1j * ky * v_hat))
    return dvx, dvy


def apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = dealias * np.fft.fftn(f)
    return np.real(np.fft.ifftn(f_hat))


def diffusion_solve(v, dt, nu, kSq):
    """solve the diffusion equation over a timestep dt, given viscosity nu"""
    v_hat = (np.fft.fftn(v)) / (1.0 + dt * nu * kSq)
    v = np.real(np.fft.ifftn(v_hat))
    return v


def main():
    """Spectral Simulation"""

    # Simulation parameters
    N = 100  # resolution
    courant_fac = 0.5  # Courant factor
    t = 0  # current time of the simulation
    tEnd = 1 / np.sqrt(3)  # time at which simulation ends
    Nout = 100  # number of frames to draw
    nu = 0.001  # viscosity (spectral methods need explicit viscosity)
    saveFrames = False  # save frames to create movie

    # Mesh
    Lbox = 1.0  # box size
    dx = Lbox / N
    xlin = np.linspace(0.5 * dx, Lbox - 0.5 * dx, N)
    X, Y = np.meshgrid(xlin, xlin)

    # Generate Initial Conditions
    rho = 0 * X + 1
    vx = 0.5 * np.sin(2 * np.pi * Y)
    vy = 0.1 * np.sin(4 * np.pi * X) * np.cos(2 * np.pi * Y) ** 2

    # Fourier Space Variables
    klin = 2.0 * np.pi / Lbox * np.arange(-N / 2, N / 2)
    kmax = np.max(klin)
    kx, ky = np.meshgrid(klin, klin)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kSq = kx**2 + ky**2

    # De-alias with the 2/3 rule
    dealias = (np.abs(kx) < (2.0 / 3.0) * kmax) & (np.abs(ky) < (2.0 / 3.0) * kmax)

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=100)
    cmap = plt.cm.bwr
    cmap.set_bad("LightGray")
    outputCount = 1

    # Simulation Main Loop
    while t < tEnd:
        # get time step (CFL) = dx / max signal speed
        dt = courant_fac * np.min(dx / (1 + np.sqrt(vx**2 + vy**2)))
        plotThisTurn = False
        if t + dt > outputCount * tEnd / Nout:
            dt = outputCount * tEnd / Nout - t
            plotThisTurn = True

        # Advection: rhs = -(v.grad)v
        drho_x, drho_y = grad(rho, kx, ky)
        dvx_x, dvx_y = grad(vx, kx, ky)
        dvy_x, dvy_y = grad(vy, kx, ky)

        rhs_rho = -(vx * drho_x + vy * drho_y) - rho * (dvx_x + dvy_y)
        rhs_vx = -(vx * dvx_x + vy * dvx_y)
        rhs_vy = -(vx * dvy_x + vy * dvy_y)

        rhs_rho = apply_dealias(rhs_rho, dealias)
        rhs_vx = apply_dealias(rhs_vx, dealias)
        rhs_vy = apply_dealias(rhs_vy, dealias)

        rho += dt * rhs_rho
        vx += dt * rhs_vx
        vy += dt * rhs_vy

        # Pressure
        P = rho
        dPx, dPy = grad(P, kx, ky)

        # Add pressure
        # rho += -dt * rho * (dvx_x + dvy_y)
        vx += -dt * dPx / rho  # apply_dealias(dt * dPx / rho, dealias)
        vy += -dt * dPy / rho  # apply_dealias(dt * dPy / rho, dealias)

        # Diffusion solve (implicit)
        vx = diffusion_solve(vx, dt, nu, kSq)
        vy = diffusion_solve(vy, dt, nu, kSq)

        # update time
        t += dt

        # Plot in real time
        if plotThisTurn:
            plt.cla()
            plot_field = 1.0 * rho
            if saveFrames:
                # fancy plot to illustrate FV concept
                plot_field[np.tile(mask, (int(N / 10), int(N / 10))) == 0] = np.nan
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
                    "Spectral",
                    fontsize=20,
                    horizontalalignment="center",
                )
                plt.savefig(
                    "tmp/spec%03d.png" % (outputCount - 1),
                    dpi=100,
                    bbox_inches="tight",
                    pad_inches=0,
                )
            outputCount += 1
            plt.pause(0.001)

    # Save figure
    plt.savefig("spectral.png", dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
