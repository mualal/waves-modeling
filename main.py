from chains import ChainChainStructure
from lattices import LatticeLatticeStructure
import numpy as np
from waves_vis_utils import monitor_energy, animate_lattices, animate_chains


if __name__ == "__main__":

    chain_chain = ChainChainStructure(m_1=0.5, m_2=1.0,
                                      c_1=0.1, c_2=0.1, c_12=0.1,
                                      d_1=0.0, d_2=0.2,
                                      cnt=401, a=1)
    chain_chain.specify_initial_and_boundary(beta=0.035, u_0=1, omega_undim=np.sqrt(0.5))
    chain_chain.plot_field()
    chain_chain.solve()
    chain_chain.plot_field()
    monitor_energy(chain_chain)
    animate_chains(chain_chain)

    lattice_lattice = LatticeLatticeStructure(m_1=0.5, m_2=1.0,
                                              c_1=0.1, c_2=0.1, c_12=0.1,
                                              d_1=0.0, d_2=0.2,
                                              cnt_x=401, cnt_y=401, a=1)
    lattice_lattice.specify_initial_and_boundary(gamma=np.radians(0), beta_x=0.035, beta_y=0.035,
                                                 u_0=1, omega_undim=np.sqrt(0.5))
    lattice_lattice.plot_field()
    lattice_lattice.solve()
    lattice_lattice.plot_field()
    monitor_energy(lattice_lattice)
    animate_lattices(lattice_lattice)
