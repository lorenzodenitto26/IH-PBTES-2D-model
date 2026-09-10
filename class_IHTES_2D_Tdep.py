# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:46:00 2026

@author: lolle
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

from class_IHTES import IHTES, mu_0


class IHTES2D_Tdep(IHTES):
    """
    Estensione 2D assialsimmetrica della classe IHTES.
    Per ora costruisce solo la mappa del campo magnetico B(r,z), H(r,z).
    NON modifica ancora il charging termico.
    """

    # ============================================================
    # Electrical resistivity / conductivity of magnetite
    # Botrous El Badramany et al. 1979, Fig. 1, DC resistivity
    # rho_e in ohm*cm
    # sigma [S/m] = 100 / rho_e[ohm*cm]
    # ============================================================
    _T_sigma_K = np.array(
        [300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800],
        dtype=float
    )

    _rho_e_ohm_cm_botrous = np.array(
        [
            4.50e-3,
            3.14e-3,
            2.40e-3,
            1.95e-3,
            1.65e-3,
            1.44e-3,
            1.28e-3,
            1.16e-3,
            1.07e-3,
            9.96e-4,
            9.35e-4
        ],
        dtype=float
    )

    _sigma_botrous = 100.0 / _rho_e_ohm_cm_botrous

    # ============================================================
    # Effective magnetic permeability temperature scaling
    # Dunlop 2014, Fig. 3a, 110 um magnetite heating curve
    # f_mu(T) = k0(T) / k0(20 C)
    #
    # This is NOT an absolute permeability of centimetric pebbles.
    # It is used only as an effective temperature scaling.
    # ============================================================
    _T_mu_C = np.array(
        [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 525, 550, 575],
        dtype=float
    )

    _f_mu_110 = np.array(
        [
            1.002,
            0.994,
            0.984,
            0.976,
            0.962,
            0.958,
            0.958,
            0.958,
            0.949,
            0.949,
            0.944,
            0.947,
            0.945,
            0.947
        ],
        dtype=float
    )

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
    def qeddy_map_Tlocal(self, freq, I, T_map):
        """
    Densità di potenza eddy locale [W/m3] con dipendenza termica relativa
    di sigma(T) e mu_r(T).

    IMPORTANTE:
    qeddy_map(freq, I) è la mappa eddy del modello già validato.

    Per non rompere la validazione, non sostituiamo il valore assoluto
    della conducibilità del modello con quello del paper.

    Usiamo solo i rapporti relativi:

        sigma_scale = sigma(T) / sigma(T_ref)
        mu_scale    = [mu_r(T) / mu_r(T_ref)]^2

    così a T_ref la nuova q_eddy coincide con quella validata.
    """
        T_map = np.asarray(T_map, dtype=float)

    # mappa eddy del modello validato
        q_eddy_ref = self.qeddy_map(freq, I)

    # temperatura di riferimento: temperatura iniziale del charging
        T_ref = 293.15

    # variazione relativa della conducibilità elettrica
        sigma_scale = self.sigma_magnetite_T(T_map) / self.sigma_magnetite_T(T_ref)

    # variazione relativa della permeabilità magnetica efficace
        mu_scale = (self.mu_r_magnetite_T(T_map) / self.mu_r_magnetite_T(T_ref))**2

    # nuova q_eddy: stessa q_eddy validata, ma corretta solo per la variazione con T
        q_eddy_T = q_eddy_ref * sigma_scale * mu_scale

        return q_eddy_T

    def qind_map(self, freq, I, T):
        """
        Densità di potenza totale indotta [W/m^3].
        """
        q_hys = self.qhys_map(freq, I, T)
        q_eddy = self.qeddy_map(freq, I)
        return q_hys + q_eddy

    def qhys_map_Tlocal(self, freq, I, T_map):
        """
        Densità di potenza isteretica locale [W/m3] usando:
        - H locale da field_map(I)
        - T locale della cella T_map[i,j]

        Questa versione chiama Phys_sp con scalari, evitando problemi
        quando Phys_sp riceve array 2D.
        """
        _, _, _, H_rms = self.field_map(I)
        H_peak = np.sqrt(2.0) * H_rms

        T_map = np.asarray(T_map, dtype=float)
        q_hys = np.zeros_like(T_map, dtype=float)

        for i in range(self.Nr):
            for j in range(self.Nz):
                q_hys[i, j] = (
                    self.Phys_sp(H_peak[i, j], T_map[i, j], freq)
                    * self.rho_s
                    * (1.0 - self.e)
                )

        return q_hys

    def qind_map_Tlocal(self, freq, I, T_map):
        """
        Densità di potenza totale indotta [W/m3] con proprietà locali.

        q_hys:
            dipende da H(r,z) e T(r,z), tramite LangArc/Phys_sp.

        q_eddy:
            dipende da H(r,z), sigma(T) e mu_r(T).
        """
        q_hys = self.qhys_map_Tlocal(freq, I, T_map)
        q_eddy = self.qeddy_map_Tlocal(freq, I, T_map)

        return q_hys + q_eddy
    def qsource_map_Tlocal(self, freq, I, T_map, source_mode="total"):
        """
    Seleziona la sorgente termica da usare nel charging.

    source_mode:
        "total"     -> q_hys + q_eddy
        "hys_only"  -> solo q_hys
        "eddy_only" -> solo q_eddy
    """
        if source_mode == "total":
            return self.qind_map_Tlocal(freq, I, T_map)

        elif source_mode == "hys_only":
            return self.qhys_map_Tlocal(freq, I, T_map)

        elif source_mode == "eddy_only":
            return self.qeddy_map_Tlocal(freq, I, T_map)

        else:
            raise ValueError(
            "source_mode must be 'total', 'hys_only', or 'eddy_only'"
        )

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

    def cp_magnetite_raw(self, T):
        """
        Calore specifico della raw magnetite [J/kg/K].
        T in K.
        """
        T_C = np.asarray(T, dtype=float) - 273.15
        T_C_clip = np.clip(T_C, 50.0, 500.0)

        T_data = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500], dtype=float)
        cp_data = np.array([640, 690, 730, 770, 810, 860, 900, 940, 970, 1010], dtype=float)

        return np.interp(T_C_clip, T_data, cp_data)

    def k_magnetite_raw(self, T):
        """
        Conducibilità termica della raw magnetite [W/m/K].
        T in K.
        """
        T_C = np.asarray(T, dtype=float) - 273.15
        T_C_clip = np.clip(T_C, 50.0, 550.0)

        T_data = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550], dtype=float)
        k_data = np.array([4.5, 4.4, 4.25, 4.15, 3.75, 3.55, 3.40, 3.15, 2.85, 2.55, 2.20], dtype=float)

        return np.interp(T_C_clip, T_data, k_data)

    def rhoCp_eff_T(self, T):
        """
        Capacità termica volumetrica efficace del packed bed [J/m3/K],
        usando cp_s(T) della raw magnetite.
        """
        cp_loc = self.cp_magnetite_raw(T)
        return (1.0 - self.e) * self.rho_s * cp_loc
    def rho_e_magnetite_T(self, T):
        """
    Resistività elettrica della magnetite [ohm*m].
    T in K.

    Dati da Botrous El Badramany et al. 1979, Fig. 1.
    I dati originali sono in ohm*cm; qui vengono convertiti in ohm*m.

    Nota:
    valori fuori dal range 300-800 K vengono clippati.
    """
        T_clip = np.clip(np.asarray(T, dtype=float),
                     self._T_sigma_K[0],
                     self._T_sigma_K[-1])

        rho_ohm_cm = np.interp(T_clip, self._T_sigma_K, self._rho_e_ohm_cm_botrous)

    # conversione: 1 ohm*cm = 0.01 ohm*m
        return rho_ohm_cm * 0.01


    def sigma_magnetite_T(self, T):
        """
    Conducibilità elettrica della magnetite [S/m].
    T in K.

    sigma(T) = 1 / rho_e(T)
    """
        return 1.0 / self.rho_e_magnetite_T(T)


    def sigma_p_ref(self):
        """
    Conducibilità elettrica di riferimento usata nel modello originale [S/m].

    Nel tuo notebook:
    EM_mag = (100, 10, 1.0, 5200)

    quindi sigma_ref = 100 S/m.
    """
        return 100.0


    def mu_r_p_ref(self):
        """
    Permeabilità magnetica relativa di riferimento usata nel modello originale.

    Nel tuo notebook:
    EM_mag = (100, 10, 1.0, 5200)

    quindi mu_r_ref = 10.
    """
        return 10.0


    def f_mu_magnetite_T(self, T):
        """
    Scaling termico efficace della suscettibilità magnetica.

    f_mu(T) = k0(T) / k0(20 C)

    Dati da Dunlop 2014, curva 110 um.
    T in K.

    Nota:
    valori fuori dal range sperimentale vengono clippati.
    """
        T_C = np.asarray(T, dtype=float) - 273.15

        T_C_clip = np.clip(T_C,
                       self._T_mu_C[0],
                       self._T_mu_C[-1])

        return np.interp(T_C_clip, self._T_mu_C, self._f_mu_110)


    def mu_r_magnetite_T(self, T):
        """
    Permeabilità magnetica relativa efficace della magnetite.

    Non usiamo Dunlop come valore assoluto di mu_r.
    Usiamo solo uno scaling:

        chi(T) = chi_ref * f_mu(T)
        mu_r(T) = 1 + chi(T)

    dove:
        chi_ref = mu_r_ref - 1
    """
        mu_ref = self.mu_r_p_ref()
        chi_ref = mu_ref - 1.0

        return 1.0 + chi_ref * self.f_mu_magnetite_T(T)


    def skin_depth_magnetite_T(self, freq, T):
        """
    Skin depth locale [m] usando sigma(T) e mu_r(T).
    """
        omega = 2.0 * np.pi * freq

        sigma_loc = self.sigma_magnetite_T(T)
        mu_r_loc = self.mu_r_magnetite_T(T)

        return np.sqrt(2.0 / (omega * mu_0 * mu_r_loc * sigma_loc))

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
    
    def heq_top_bottom_from_Pamb(self, T_ref):
        """
    Coefficiente equivalente di scambio verso l'ambiente [W/m2/K]
    per le superfici top e bottom.

    Modello semplificato piano:
        R'' = e_ins/k_ins + 1/h_ext

    dove:
        e_ins = spessore isolamento
        k_ins = conducibilità isolamento
        h_ext = coefficiente convettivo esterno
    """
        k_ins = 0.035 + 1e-4 * (T_ref - 300.0)
        h_ext = 10.0

        R_area = self.e_ins / k_ins + 1.0 / h_ext

        return 1.0 / R_area

    def explicit_dt_limit(self, k_eff=None, T_ref=None):
        """
        Stima del dt massimo per lo schema esplicito.

        Se T_ref è fornita, usa cp_s(T) e k_s(T) della raw magnetite.
        """
        if T_ref is None:
            if k_eff is None:
                k_eff = self.k_s
            alpha = k_eff / self.rhoCp_eff()
        else:
            k_loc = self.k_magnetite_raw(T_ref)
            rhoCp_loc = self.rhoCp_eff_T(T_ref)
            alpha = np.max(k_loc / rhoCp_loc)

        return 1.0 / (2.0 * alpha * (1.0 / self.dr**2 + 1.0 / self.dz**2))

    def charging_2D(self, freq, I, T_init, T_amb, ch_time, dt,
                    T_ref_source=None, k_eff=None, save_every=100):
        """
        Charging termico 2D del packed bed.
        - sorgente termica uniforme nel tempo e nello spazio
        - proprietà termiche costanti
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

    def charging_2D_nonuniform(self, freq, I, T_init, T_amb, ch_time, dt,
                               T_ref_source=None, k_eff=None, save_every=100):
        """
        Charging termico 2D del packed bed con sorgente non uniforme basata su H(r,z).
        Versione validata con proprietà termiche costanti e dipendenza termica su T_bulk.
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

    def charging_2D_nonuniform_Tdep(self, freq, I, T_init, T_amb, ch_time, dt,
                                    T_ref_source=None, save_every=100,
                                    source_mode="total"):
        """
        Charging termico 2D del packed bed con:
        - sorgente non uniforme basata su H(r,z)
        - cp_s(T) della raw magnetite
        - k_s(T) della raw magnetite
        - sorgente isteretica valutata con temperatura locale T(r,z)

        Nota: la parte eddy resta per ora coerente col modello validato.
        """
        if T_ref_source is None:
            T_ref_source = T_init

        # perdita laterale equivalente
        h_eq = self.heq_from_Pamb(T_ref_source)

        # perdita equivalente da top e bottom
        h_eq_tb = self.heq_top_bottom_from_Pamb(T_ref_source)

        # salvo i valori per poterli controllare da Jupyter
        self.last_h_eq_side = h_eq
        self.last_h_eq_top_bottom = h_eq_tb

        n_steps = int(np.ceil(ch_time / dt))
        time = np.linspace(0.0, n_steps * dt, n_steps + 1)

        T = np.full((self.Nr, self.Nz), float(T_init))

        T_mean = np.zeros(n_steps + 1)
        T_max = np.zeros(n_steps + 1)
        q_mean = np.zeros(n_steps + 1)
        q_max = np.zeros(n_steps + 1)

        T_mean[0] = self.volume_average(T)
        T_max[0] = np.max(T)

        snapshots = [T.copy()]
        snapshot_times = [0.0]

        dt_lim = self.explicit_dt_limit(T_ref=T)
        if dt > 0.8 * dt_lim:
            print(f"Warning: dt={dt:.4g} s is rather large for the explicit scheme.")
            print(f"Suggested dt <= {0.8 * dt_lim:.4g} s")

        q_src = self.qsource_map_Tlocal(freq, I, T, source_mode=source_mode)
        q_mean[0] = self.volume_average(q_src)
        q_max[0] = np.max(q_src)

        for n in range(n_steps):
            q_src = self.qsource_map_Tlocal(freq, I, T, source_mode=source_mode)
            Tnew = T.copy()

            for i in range(self.Nr):
                r_i = self.r[i]

                for j in range(self.Nz):
                    rhoCp_ij = self.rhoCp_eff_T(T[i, j])
                    k_ij = self.k_magnetite_raw(T[i, j])

                    # =====================================================
                    # Bordo assiale bottom/top con perdite verso ambiente
                    # =====================================================
                    if j == 0:
                        # bottom surface, z = 0
                        # ghost node esterno sotto il packed bed
                        T_jp1 = T[i, 1]
                        T_jm1 = T_jp1 - 2.0 * self.dz * (h_eq_tb / k_ij) * (T[i, j] - T_amb)

                    elif j == self.Nz - 1:
                        # top surface, z = H
                        # ghost node esterno sopra il packed bed
                        T_jm1 = T[i, self.Nz - 2]
                        T_jp1 = T_jm1 - 2.0 * self.dz * (h_eq_tb / k_ij) * (T[i, j] - T_amb)

                    else:
                        # celle interne lungo z
                        T_jm1 = T[i, j - 1]
                        T_jp1 = T[i, j + 1]

                    d2z = (T_jp1 - 2.0 * T[i, j] + T_jm1) / self.dz**2

                    if i == 0:
                        radial = 4.0 * (T[1, j] - T[0, j]) / self.dr**2
                    elif i == self.Nr - 1:
                        T_ghost = T[i - 1, j] - 2.0 * self.dr * (h_eq / k_ij) * (T[i, j] - T_amb)
                        d2r = (T_ghost - 2.0 * T[i, j] + T[i - 1, j]) / self.dr**2
                        d1r = (T_ghost - T[i - 1, j]) / (2.0 * self.dr)
                        radial = d2r + d1r / r_i
                    else:
                        d2r = (T[i + 1, j] - 2.0 * T[i, j] + T[i - 1, j]) / self.dr**2
                        d1r = (T[i + 1, j] - T[i - 1, j]) / (2.0 * self.dr)
                        radial = d2r + d1r / r_i

                    laplacian = radial + d2z
                    Tnew[i, j] = T[i, j] + (dt / rhoCp_ij) * (k_ij * laplacian + q_src[i, j])

            T = Tnew

            T_mean[n + 1] = self.volume_average(T)
            T_max[n + 1] = np.max(T)
            q_mean[n + 1] = self.volume_average(q_src)
            q_max[n + 1] = np.max(q_src)

            if ((n + 1) % save_every == 0) or (n == n_steps - 1):
                snapshots.append(T.copy())
                snapshot_times.append((n + 1) * dt)

        return time, T_mean, T_max, T, np.array(snapshot_times), snapshots, q_src, h_eq, q_mean, q_max
