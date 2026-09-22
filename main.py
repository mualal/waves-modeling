from chains import ChainChainStructure
from lattices import LatticeLatticeStructure
import numpy as np
from waves_vis_utils import monitor_energy, animate_lattices, animate_chains


if __name__ == "__main__":

    chain_chain = ChainChainStructure(m1=0.5, m2=1.0,
                                      c1=0.1, c2=0.1, c12=0.1,
                                      d1=0.0, d2=0.2,
                                      cnt=401, a=1)
    chain_chain.specify_initial_and_boundary(beta=0.035, u0=1, omega_undim=np.sqrt(0.5))
    chain_chain.plot_field()
    chain_chain.solve()
    chain_chain.plot_field()
    monitor_energy(chain_chain)
    animate_chains(chain_chain)

    lattice_lattice = LatticeLatticeStructure(m1=0.5, m2=1.0,
                                              c1=0.1, c2=0.1, c12=0.1,
                                              d1=0.0, d2=0.2,
                                              cnt_x=401, cnt_y=401, a=1)
    lattice_lattice.specify_initial_and_boundary(gamma=np.radians(0), beta_x=0.035, beta_y=0.035,
                                                 u0=1, omega_undim=np.sqrt(0.5))
    lattice_lattice.plot_field()
    lattice_lattice.solve()
    lattice_lattice.plot_field()
    monitor_energy(lattice_lattice)
    animate_lattices(lattice_lattice)
