from chains import ChainChainStructure
from lattices import LatticeLatticeStructure
import numpy as np
from waves_vis_utils import monitor_energy, lattices_animation, chains_animation


if __name__ == "__main__":

    chain_chain = ChainChainStructure(m_1=0.5, m_2=1.0,
                                      c_1=0.1, c_2=0.1, c_12=0.1,
                                      d_1=0.0, d_2=0.2,
                                      cnt=301, a=1)
    chain_chain.specify_initial_and_boundary(beta=0.035, n_0=-70, u_0=1, omega_undim=np.sqrt(0.5))
    chain_chain.solve(dt=0.05, t_max=400, save_time=15)
    monitor_energy(chain_chain)

    lattice_lattice = LatticeLatticeStructure(m_1=0.5, m_2=1.0,
                                              c_1=0.1, c_2=0.1, c_12=0.1,
                                              d_1=0.0, d_2=0.2,
                                              cnt_x=301, cnt_y=301, a=1)
    lattice_lattice.specify_initial_and_boundary(gamma=np.radians(0), beta_x=0.035, beta_y=0.035,
                                                 shift_x=-70, shift_y=-35, u_0=1, omega_undim=np.sqrt(0.5))
    lattice_lattice.solve(dt=0.05, t_max=400, save_time=15)
    monitor_energy(lattice_lattice)
