import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Lattice Boltzmann Simulation (With Python)
Philip Mocz 2023, @PMocz

Simulate the compressible Euler equations

Method based on:
https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c

see link for details

"""

# fancy plot
mask = [
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
]


def main():
    """Lattice Boltzmann Simulation"""

    # Simulation parameters
    N = 100  # resolution
    tau = 0.51  # collision timescale (> 0.5)
    Nt = 1 * N  # number of timesteps. Nt = N corresponds to tEnd = 1/np.sqrt(3)
    saveFrames = False  # save frames to create movie

    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array(
        [4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36]
    )  # sums to 1

    # Initial Conditions
    # units: dx = dt = 1, c^2 = 1/3
    # other codes use: Lbox = c = <rho> = 1
    # other code ICs:
    #   rho_other = 0*X + 1 + 0.1*np.sin(2*np.pi*Y)
    #   vx_other = np.sin(2*np.pi*Y)
    #   vy_other =  0.1*np.sin(4*np.pi*X)*np.cos(2*np.pi*Y)**2
    # here, we need to multiply:
    #   vx = vx_other * sqrt(1/3)

    F = np.zeros((N, N, NL))
    Lbox = 1
    dx = Lbox / N
    vol = dx**2
    xlin = np.linspace(0.5 * dx, Lbox - 0.5 * dx, N)
    X, Y = np.meshgrid(xlin, xlin)
    rho = 0 * X + 1
    ux = 0.5 * np.sin(2 * np.pi * Y) * np.sqrt(1 / 3)
    uy = 0.1 * np.sin(4 * np.pi * X) * np.cos(2 * np.pi * Y) ** 2 * np.sqrt(1 / 3)

    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        F[:, :, i] = (
            rho
            * w
            * (
                1
                + 3 * (cx * ux + cy * uy)
                + 9 * (cx * ux + cy * uy) ** 2 / 2
                - 3 * (ux**2 + uy**2) / 2
            )
        )

    # Prep figure
    fig = plt.figure(figsize=(4, 4), dpi=100)
    cmap = plt.cm.bwr
    cmap.set_bad("LightGray")

    # Simulation Main Loop
    for it in range(Nt):
        print(it)

        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # Calculate fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = (
                rho
                * w
                * (
                    1
                    + 3 * (cx * ux + cy * uy)
                    + 9 * (cx * ux + cy * uy) ** 2 / 2
                    - 3 * (ux**2 + uy**2) / 2
                )
            )

        F += -(1.0 / tau) * (F - Feq)

        # Plot in real time
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
                "Lattice-Boltzmann",
                fontsize=20,
                horizontalalignment="center",
            )
            plt.savefig(
                "tmp/lb%03d.png" % it, dpi=100, bbox_inches="tight", pad_inches=0
            )
        plt.pause(0.001)

    # Save figure
    plt.savefig("latticeboltzmann.png", dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
