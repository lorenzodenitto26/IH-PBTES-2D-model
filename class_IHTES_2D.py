# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:46:00 2026

@author: lolle
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

from class_IHTES import IHTES, mu_0


class IHTES2D(IHTES):
    """
    Estensione 2D assialsimmetrica della classe IHTES.
    Per ora costruisce solo la mappa del campo magnetico B(r,z), H(r,z).
    NON modifica ancora il charging termico.
    """

    def __init__(self, TH_pro, EM_prop, hys_para, geom, mesh=(40, 80)):
        # inizializza tutta la classe vecchia
        super().__init__(TH_pro, EM_prop, hys_para, geom)

        # mesh 2D
        self.Nr, self.Nz = mesh

        # dominio assialsimmetrico: r in [0, R], z in [0, H]
        self.R_TES = self.D_TES / 2.0
        self.Z_TES = self.H_TES

        self.r = np.linspace(0.0, self.R_TES, self.Nr)
        self.z = np.linspace(0.0, self.Z_TES, self.Nz)

        self.dr = self.r[1] - self.r[0] if self.Nr > 1 else self.R_TES
        self.dz = self.z[1] - self.z[0] if self.Nz > 1 else self.Z_TES

        # griglia 2D
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing="ij")

        # numero reale di spire -> numero intero di spire per il modello 2D
        self.N_turns = max(1, int(round(self.N)))

        # raggio medio coil
        self.a_coil = self.D_coil / 2.0

        # posizione assiale delle spire
        self.z_turns = self._build_turn_positions()

        # volumi celle assialsimmetriche (serviranno dopo per il termico)
        self.cell_volumes = self._build_cell_volumes()

    def _build_turn_positions(self):
        """
        Posizioni assiali delle spire distribuite lungo l'altezza della coil.
        """
        if self.N_turns == 1:
            return np.array([self.H_coil / 2.0])

        return np.linspace(0.0, self.H_coil, self.N_turns)

    def _build_cell_volumes(self):
        """
        Volume della cella assialsimmetrica:
        V_ij = 2*pi*r_i*dr*dz
        """
        volumes = np.zeros((self.Nr, self.Nz))
        for i, r_i in enumerate(self.r):
            volumes[i, :] = 2.0 * np.pi * r_i * self.dr * self.dz
        return volumes

    def single_turn_field(self, r, z, a, z_turn, I):
        """
        Campo magnetico di una singola spira circolare di raggio a,
        posta alla quota z_turn, in coordinate cilindriche (r,z).

        Restituisce:
        Br, Bz
        """
        r = np.asarray(r, dtype=float)
        z = np.asarray(z, dtype=float)

        z_rel = z - z_turn

        Br = np.zeros_like(r, dtype=float)
        Bz = np.zeros_like(r, dtype=float)

        # punti sull'asse r = 0
        axis_mask = np.isclose(r, 0.0)

        if np.any(axis_mask):
            Bz[axis_mask] = mu_0 * I * a**2 / (2.0 * (a**2 + z_rel[axis_mask]**2)**1.5)
            Br[axis_mask] = 0.0

        # punti fuori asse
        off_mask = ~axis_mask
        if np.any(off_mask):
            rr = r[off_mask]
            zz = z_rel[off_mask]

            alpha2 = (a + rr)**2 + zz**2
            beta2  = (a - rr)**2 + zz**2

            k2 = 4.0 * a * rr / alpha2
            k2 = np.clip(k2, 0.0, 1.0 - 1e-12)

            K = ellipk(k2)
            E = ellipe(k2)

            alpha = np.sqrt(alpha2)

            common = mu_0 * I / (2.0 * np.pi * alpha)

            Br_val = common * (zz / rr) * (
                -K + E * (a**2 + rr**2 + zz**2) / beta2
            )

            Bz_val = common * (
                K + E * (a**2 - rr**2 - zz**2) / beta2
            )

            Br[off_mask] = Br_val
            Bz[off_mask] = Bz_val

        return Br, Bz

    def field_map(self, I):
        """
        Somma il contributo di tutte le spire.
        Restituisce:
        Br_tot, Bz_tot, B_tot, H_tot
        """
        Br_tot = np.zeros_like(self.R)
        Bz_tot = np.zeros_like(self.Z)

        for zt in self.z_turns:
            Br_i, Bz_i = self.single_turn_field(self.R, self.Z, self.a_coil, zt, I)
            Br_tot += Br_i
            Bz_tot += Bz_i

        B_tot = np.sqrt(Br_tot**2 + Bz_tot**2)

        # prima approssimazione: H = B / mu0
        H_tot = B_tot / mu_0

        return Br_tot, Bz_tot, B_tot, H_tot
    def plot_H_map(self, I, levels=40):
         _, _, _, H = self.field_map(I)

         plt.figure(figsize=(5, 8))
         cp = plt.contourf(self.r, self.z, H.T, levels=levels)
         plt.colorbar(cp, label="H [A/m]")
         plt.xlabel("r [m]")
         plt.ylabel("z [m]")
         plt.title("Mappa 2D del campo magnetico H(r,z)")
         plt.tight_layout()
         plt.show()
    
    def axial_profile(self, I, r_index=0):
        """
        Profilo lungo z per un dato indice radiale.
        """
        _, _, _, H = self.field_map(I)
        return self.z, H[r_index, :]

    def radial_profile(self, I, z_index=None):
        """
        Profilo lungo r a una quota z.
        """
        _, _, _, H = self.field_map(I)

        if z_index is None:
            z_index = self.Nz // 2

        return self.r, H[:, z_index]
    def delta_map(self, freq):
        """
        Mappa 2D della skin depth del pebble.
        Per ora è uguale in tutte le celle, perché mu_r_p e sigma_p sono costanti.
        """
        delta = self.depth_p(freq)
        return np.full_like(self.R, delta, dtype=float)

    def F_map(self, freq):
        """
        Mappa 2D del fattore F del modello del PhD.
        Per ora è uguale in tutte le celle, perché delta_p è uguale in tutte le celle.
        """
        Fval = self.F(freq)
        return np.full_like(self.R, Fval, dtype=float)

    def volume_weights(self):
        """
        Pesi volumetrici assialsimmetrici.
        """
        r_eff = np.where(np.isclose(self.R, 0.0), self.dr / 2.0, self.R)
        return 2.0 * np.pi * r_eff * self.dr * self.dz

    def volume_average(self, M):
        """
        Media volumetrica pesata di una mappa 2D.
        """
        W = self.volume_weights()
        return np.sum(M * W) / np.sum(W)

    def qhys_map(self, freq, I, T):
        """
        Densità di potenza isteretica locale [W/m^3] per ogni cella.
        Usa la stessa Phys_sp del PhD, ma con H locale.
        """
        _, _, _, H_rms = self.field_map(I)
        H_peak = np.sqrt(2.0) * H_rms

        # Phys_sp -> W/kg
        # conversione a W/m^3 di packed bed
        q_hys = self.Phys_sp(H_peak, T, freq) * self.rho_s * (1.0 - self.e)

        return q_hys

    def qeddy_map(self, freq, I):
        """
        Densità di potenza eddy locale [W/m^3] per ogni cella.

        Mantiene la potenza totale del modello del PhD:
        Peddy = Req(freq) * I^2

        e la distribuisce nelle celle con una forma spaziale coerente con:
        Peddy ~ H^2
        usando anche F(freq) del modello del PhD.
        """
        _, _, _, H_rms = self.field_map(I)
        F_loc = self.F_map(freq)

        # forma spaziale locale
        phi = F_loc * H_rms**2

        # densità di potenza media globale del vecchio modello
        qeddy_old = self.qv_eddy(freq, I)

        # media volumetrica della forma spaziale
        phi_avg = self.volume_average(phi)

        # normalizzazione: la media torna uguale al vecchio modello
        q_eddy = qeddy_old * phi / phi_avg

        return q_eddy

    def qind_map(self, freq, I, T):
        """
        Densità di potenza totale indotta [W/m^3].
        """
        q_hys = self.qhys_map(freq, I, T)
        q_eddy = self.qeddy_map(freq, I)
        return q_hys + q_eddy

    def Phys_cells(self, freq, I, T):
        """
        Potenza isteretica di ogni cella [W].
        """
        return self.qhys_map(freq, I, T) * self.cell_volumes

    def Peddy_cells(self, freq, I):
        """
        Potenza eddy di ogni cella [W].
        """
        return self.qeddy_map(freq, I) * self.cell_volumes

    def Pind_cells(self, freq, I, T):
        """
        Potenza indotta totale di ogni cella [W].
        """
        return self.qind_map(freq, I, T) * self.cell_volumes

    def plot_scalar_map(self, M, title="Mappa 2D", cbar_label="", levels=40):
        
        """
        Plot generico di una mappa 2D nel dominio (r,z).
        """
        plt.figure(figsize=(5, 8))
        cp = plt.contourf(self.r, self.z, M.T, levels=levels)
        plt.colorbar(cp, label=cbar_label)
        plt.xlabel("r [m]")
        plt.ylabel("z [m]")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    def rhoCp_eff(self):
        """
        Capacità termica volumetrica efficace del packed bed [J/m^3/K].
        Per ora uso solo la fase solida, coerentemente con il vecchio charging.
        """
        return (1.0 - self.e) * self.rho_s * self.cp_s

    def heq_from_Pamb(self, T_ref):
        """
        Coefficiente equivalente di scambio verso l'ambiente [W/m^2/K],
        ricavato dalla stessa logica del Pamb del modello del PhD.
        """
        k_ins = 0.035 + 1e-4 * (T_ref - 300.0)
        h_ext = 10.0

        D_eff = self.D_TES + 2.0 * self.e_ins

        R_cond = np.log(D_eff / self.D_TES) / (2.0 * np.pi * k_ins * self.H_TES)
        R_conv = 1.0 / (h_ext * np.pi * D_eff * self.H_TES)
        R_tot = R_cond + R_conv

        A_side_in = np.pi * self.D_TES * self.H_TES

        return 1.0 / (A_side_in * R_tot)

    def explicit_dt_limit(self, k_eff=None):
        """
        Stima del dt massimo per lo schema esplicito.
        """
        if k_eff is None:
            k_eff = self.k_s

        alpha = k_eff / self.rhoCp_eff()

        return 1.0 / (2.0 * alpha * (1.0 / self.dr**2 + 1.0 / self.dz**2))

    def charging_2D(self, freq, I, T_init, T_amb, ch_time, dt,
                T_ref_source=None, k_eff=None, save_every=100):
      """
    Charging termico 2D del packed bed.

    - sorgente termica uniforme nel tempo e nello spazio
    - proprietà termiche costanti
    - asse r=0: simmetria
    - parete laterale r=R: perdita verso ambiente con h_eq
    - top e bottom: adiabatici
    """

      if k_eff is None:
        k_eff = self.k_s

      if T_ref_source is None:
        T_ref_source = T_init

      rhoCp = self.rhoCp_eff()
      h_eq = self.heq_from_Pamb(T_ref_source)

      n_steps = int(np.ceil(ch_time / dt))
      time = np.linspace(0.0, n_steps * dt, n_steps + 1)

      T = np.full((self.Nr, self.Nz), float(T_init))

      T_mean = np.zeros(n_steps + 1)
      T_max = np.zeros(n_steps + 1)

      T_mean[0] = self.volume_average(T)
      T_max[0] = np.max(T)

      snapshots = [T.copy()]
      snapshot_times = [0.0]

      dt_lim = self.explicit_dt_limit(k_eff)
      if dt > 0.8 * dt_lim:
        print(f"Warning: dt={dt:.4g} s is rather large for the explicit scheme.")
        print(f"Suggested dt <= {0.8 * dt_lim:.4g} s")

      for n in range(n_steps):
        T_bulk = self.volume_average(T)
        q_hys_t = self.qv_hys(freq, I, T_bulk)
        q_eddy_t = self.qv_eddy(freq, I)
        q_src = np.full((self.Nr, self.Nz), q_hys_t + q_eddy_t)

        Tnew = T.copy()

        for i in range(self.Nr):
            r_i = self.r[i]

            for j in range(self.Nz):

                # ===== derivata seconda lungo z =====
                if j == 0:
                    T_jm1 = T[i, 1]
                    T_jp1 = T[i, 1]
                elif j == self.Nz - 1:
                    T_jm1 = T[i, self.Nz - 2]
                    T_jp1 = T[i, self.Nz - 2]
                else:
                    T_jm1 = T[i, j - 1]
                    T_jp1 = T[i, j + 1]

                d2z = (T_jp1 - 2.0 * T[i, j] + T_jm1) / self.dz**2

                # ===== operatore radiale assialsimmetrico =====
                if i == 0:
                    radial = 4.0 * (T[1, j] - T[0, j]) / self.dr**2

                elif i == self.Nr - 1:
                    T_ghost = T[i - 1, j] - 2.0 * self.dr * (h_eq / k_eff) * (T[i, j] - T_amb)

                    d2r = (T_ghost - 2.0 * T[i, j] + T[i - 1, j]) / self.dr**2
                    d1r = (T_ghost - T[i - 1, j]) / (2.0 * self.dr)

                    radial = d2r + d1r / r_i

                else:
                    d2r = (T[i + 1, j] - 2.0 * T[i, j] + T[i - 1, j]) / self.dr**2
                    d1r = (T[i + 1, j] - T[i - 1, j]) / (2.0 * self.dr)

                    radial = d2r + d1r / r_i

                laplacian = radial + d2z

                Tnew[i, j] = T[i, j] + (dt / rhoCp) * (k_eff * laplacian + q_src[i, j])

        T = Tnew

        T_mean[n + 1] = self.volume_average(T)
        T_max[n + 1] = np.max(T)

        if ((n + 1) % save_every == 0) or (n == n_steps - 1):
            snapshots.append(T.copy())
            snapshot_times.append((n + 1) * dt)

      return time, T_mean, T_max, T, np.array(snapshot_times), snapshots, q_src, h_eq
    def charging_2D_nonuniform(self, freq, I, T_init, T_amb, ch_time, dt,
                           T_ref_source=None, k_eff=None, save_every=100):
      """
    Charging termico 2D del packed bed con sorgente non uniforme basata su H(r,z).

    - sorgente termica spazialmente non uniforme
    - dipendenza termica valutata sulla temperatura media bulk
    - asse r=0: simmetria
    - parete laterale r=R: perdita verso ambiente con h_eq
    - top e bottom: adiabatici
    """

      if k_eff is None:
        k_eff = self.k_s

      if T_ref_source is None:
        T_ref_source = T_init

      rhoCp = self.rhoCp_eff()
      h_eq = self.heq_from_Pamb(T_ref_source)

      n_steps = int(np.ceil(ch_time / dt))
      time = np.linspace(0.0, n_steps * dt, n_steps + 1)

      T = np.full((self.Nr, self.Nz), float(T_init))

      T_mean = np.zeros(n_steps + 1)
      T_max = np.zeros(n_steps + 1)

      T_mean[0] = self.volume_average(T)
      T_max[0] = np.max(T)

      snapshots = [T.copy()]
      snapshot_times = [0.0]

      dt_lim = self.explicit_dt_limit(k_eff)
      if dt > 0.8 * dt_lim:
        print(f"Warning: dt={dt:.4g} s is rather large for the explicit scheme.")
        print(f"Suggested dt <= {0.8 * dt_lim:.4g} s")

      for n in range(n_steps):
        T_bulk = self.volume_average(T)

        # sorgente termica NON uniforme
        q_src = self.qind_map(freq, I, T_bulk)

        Tnew = T.copy()

        for i in range(self.Nr):
            r_i = self.r[i]

            for j in range(self.Nz):

                if j == 0:
                    T_jm1 = T[i, 1]
                    T_jp1 = T[i, 1]
                elif j == self.Nz - 1:
                    T_jm1 = T[i, self.Nz - 2]
                    T_jp1 = T[i, self.Nz - 2]
                else:
                    T_jm1 = T[i, j - 1]
                    T_jp1 = T[i, j + 1]

                d2z = (T_jp1 - 2.0 * T[i, j] + T_jm1) / self.dz**2

                if i == 0:
                    radial = 4.0 * (T[1, j] - T[0, j]) / self.dr**2

                elif i == self.Nr - 1:
                    T_ghost = T[i - 1, j] - 2.0 * self.dr * (h_eq / k_eff) * (T[i, j] - T_amb)

                    d2r = (T_ghost - 2.0 * T[i, j] + T[i - 1, j]) / self.dr**2
                    d1r = (T_ghost - T[i - 1, j]) / (2.0 * self.dr)

                    radial = d2r + d1r / r_i

                else:
                    d2r = (T[i + 1, j] - 2.0 * T[i, j] + T[i - 1, j]) / self.dr**2
                    d1r = (T[i + 1, j] - T[i - 1, j]) / (2.0 * self.dr)

                    radial = d2r + d1r / r_i

                laplacian = radial + d2z

                Tnew[i, j] = T[i, j] + (dt / rhoCp) * (k_eff * laplacian + q_src[i, j])

        T = Tnew

        T_mean[n + 1] = self.volume_average(T)
        T_max[n + 1] = np.max(T)

        if ((n + 1) % save_every == 0) or (n == n_steps - 1):
            snapshots.append(T.copy())
            snapshot_times.append((n + 1) * dt)

      return time, T_mean, T_max, T, np.array(snapshot_times), snapshots, q_src, h_eq
    