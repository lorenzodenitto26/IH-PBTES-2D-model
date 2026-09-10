# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:09:37 2026

@author: lolle
"""
import numpy as np

try:
    from CoolProp.CoolProp import PropsSI
except ImportError as exc:
    raise ImportError(
        "CoolProp is required for the final air-enthalpy discharge model. "
        "Install it with: pip install CoolProp"
    ) from exc


# ============================================================
# AIR PROPERTY TABLES
# ============================================================

P_AIR = 101325.0  # [Pa]

T_AIR_GRID = np.linspace(250.0, 1200.0, 4000)  # [K]

CP_AIR_GRID = PropsSI("C", "T", T_AIR_GRID, "P", P_AIR, "Air")   # [J/kg/K]
RHO_AIR_GRID = PropsSI("D", "T", T_AIR_GRID, "P", P_AIR, "Air") # [kg/m3]
MU_AIR_GRID = PropsSI("V", "T", T_AIR_GRID, "P", P_AIR, "Air")  # [Pa s]
K_AIR_GRID = PropsSI("L", "T", T_AIR_GRID, "P", P_AIR, "Air")   # [W/m/K]
H_AIR_GRID = PropsSI("H", "T", T_AIR_GRID, "P", P_AIR, "Air")   # [J/kg]


def _clip_T_air(T):
    return np.clip(np.asarray(T, dtype=float), T_AIR_GRID[0], T_AIR_GRID[-1])


def cp_air(T):
    """Air specific heat [J/kg/K]."""
    return np.interp(_clip_T_air(T), T_AIR_GRID, CP_AIR_GRID)


def rho_air(T):
    """Air density [kg/m3]."""
    return np.interp(_clip_T_air(T), T_AIR_GRID, RHO_AIR_GRID)


def mu_air(T):
    """Air dynamic viscosity [Pa s]."""
    return np.interp(_clip_T_air(T), T_AIR_GRID, MU_AIR_GRID)


def k_air(T):
    """Air thermal conductivity [W/m/K]."""
    return np.interp(_clip_T_air(T), T_AIR_GRID, K_AIR_GRID)


def h_air(T):
    """
    Air specific enthalpy [J/kg].

    The absolute value is not important.
    We only use enthalpy differences:
        h_air(T_out) - h_air(T_in)
    """
    return np.interp(_clip_T_air(T), T_AIR_GRID, H_AIR_GRID)


def T_from_h_air(h):
    """
    Inverse function:
    given air enthalpy [J/kg], return air temperature [K].
    """
    h_clip = np.clip(np.asarray(h, dtype=float), H_AIR_GRID[0], H_AIR_GRID[-1])
    return np.interp(h_clip, H_AIR_GRID, T_AIR_GRID)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _eval_array(func, X):
    """
    Safely evaluate a class method on an array.
    Works whether func already accepts arrays or only scalars.
    """
    try:
        return func(X)
    except Exception:
        return np.vectorize(func)(X)


def _area_weights_radial(self):
    """
    Area weights for radial integration over a circular section.
    Used to average outlet air temperature/enthalpy.
    """
    r_eff = np.where(np.isclose(self.r, 0.0), self.dr / 2.0, self.r)
    return 2.0 * np.pi * r_eff * self.dr


def _volume_weights_half(self, Nz_half, dz_half):
    """
    Volume weights for the 2D axisymmetric half-domain.
    """
    A_w = self._area_weights_radial()
    return A_w[:, None] * dz_half * np.ones((self.Nr, Nz_half))


def _area_average_outlet(self, values_r):
    """
    Area-weighted average over outlet section.
    values_r has shape (Nr,).
    """
    A_w = self._area_weights_radial()
    return np.sum(values_r * A_w) / np.sum(A_w)


def _area_average_profile_z(self, M):
    """
    Area-weighted radial average for each axial position.
    M shape: (Nr, Nz_half)
    returns shape: (Nz_half,)
    """
    A_w = self._area_weights_radial()
    return np.sum(M * A_w[:, None], axis=0) / np.sum(A_w)


def _prepare_half_tank_initial(self, T_full, method="average_two_ends"):
    """
    Prepare half packed-bed initial condition.

    Full PB:
        end 1 ---- center ---- end 2

    During discharging:
        air enters from both ends and exits from the center.

    Therefore, we simulate only half PB:
        end ---- center

    method:
        "left"             -> use first half
        "right"            -> use second half, flipped
        "average_two_ends" -> average the two symmetric halves
    """
    T_full = np.asarray(T_full, dtype=float)

    if T_full.shape != (self.Nr, self.Nz):
        raise ValueError(f"T_full must have shape {(self.Nr, self.Nz)}, got {T_full.shape}")

    Nz_half = self.Nz // 2

    left = T_full[:, :Nz_half]
    right = T_full[:, -Nz_half:][:, ::-1]

    if method == "left":
        return left.copy()

    if method == "right":
        return right.copy()

    if method == "average_two_ends":
        return 0.5 * (left + right)

    raise ValueError("method must be 'left', 'right', or 'average_two_ends'")


def _gunn_hv(self, Ts, Tf, m_dot_half, d_p=0.02):
    """
    Compute solid-fluid heat transfer coefficient.

    Gunn correlation:

    Nu =
    (7 - 10 eps + 5 eps^2)(1 + 0.7 Re^0.2 Pr^(1/3))
    +
    (1 - 2.4 eps + 1.2 eps^2) Re^0.7 Pr^(1/3)

    Then:
        h_sf = Nu k_f / d_p
        h_v  = h_sf * 6(1-eps)/d_p

    Here:
        h_sf [W/m2/K]
        h_v  [W/m3/K]
    """
    eps = self.e

    A_cross = np.pi * self.R_TES**2

    # Mass flux through one half-tank
    G = m_dot_half / A_cross  # [kg/m2/s]

    # Film temperature for air properties
    Tfilm = 0.5 * (Ts + Tf)

    cp_f = cp_air(Tfilm)
    mu_f = mu_air(Tfilm)
    k_f = k_air(Tfilm)

    # Reynolds based on superficial mass flux:
    # Re = rho*u_D*d_p/mu = G*d_p/mu
    Re = G * d_p / mu_f

    Pr = cp_f * mu_f / k_f

    Nu = (
        (7.0 - 10.0 * eps + 5.0 * eps**2)
        * (1.0 + 0.7 * Re**0.2 * Pr**(1.0 / 3.0))
        +
        (1.0 - 2.4 * eps + 1.2 * eps**2)
        * Re**0.7 * Pr**(1.0 / 3.0)
    )

    h_sf = Nu * k_f / d_p

    a_s = 6.0 * (1.0 - eps) / d_p

    h_v = h_sf * a_s

    return h_sf, h_v, Re, Pr, Nu

def _z_target_from_profile(z_half, T_profile_K, T_target, T_inlet=None):
    """
    Find first axial position where air reaches T_target.

    If T_inlet is provided, the inlet boundary point z = 0 is added.
    This avoids saying that the target is reached at 0 cm when actually
    it is reached inside the first cell.
    """
    T_profile_K = np.asarray(T_profile_K, dtype=float)

    if T_inlet is not None:
        z_eval = np.concatenate(([0.0], np.asarray(z_half, dtype=float)))
        T_eval = np.concatenate(([float(T_inlet)], T_profile_K))
    else:
        z_eval = np.asarray(z_half, dtype=float)
        T_eval = T_profile_K

    idx = np.where(T_eval >= T_target)[0]

    if len(idx) == 0:
        return np.nan

    j = idx[0]

    if j == 0:
        return z_eval[0]

    T0 = T_eval[j - 1]
    T1 = T_eval[j]
    z0 = z_eval[j - 1]
    z1 = z_eval[j]

    if np.isclose(T1, T0):
        return z1

    return z0 + (T_target - T0) * (z1 - z0) / (T1 - T0)


def _z_target_from_edges(z_edges, T_edges_K, T_target):
    """
    Find first axial position where air reaches T_target.

    z_edges are the physical positions of the cell boundaries:
        z = 0       inlet
        z = H_half  outlet

    T_edges_K is the air bulk temperature at those positions.
    """
    z_edges = np.asarray(z_edges, dtype=float)
    T_edges_K = np.asarray(T_edges_K, dtype=float)

    idx = np.where(T_edges_K >= T_target)[0]

    if len(idx) == 0:
        return np.nan

    j = idx[0]

    if j == 0:
        return z_edges[0]

    T0 = T_edges_K[j - 1]
    T1 = T_edges_K[j]
    z0 = z_edges[j - 1]
    z1 = z_edges[j]

    if np.isclose(T1, T0):
        return z1

    return z0 + (T_target - T0) * (z1 - z0) / (T1 - T0)
# ============================================================
# FINAL DISCHARGING MODEL
# ============================================================

def discharging_2D_air_final(
    self,
    T_s_init_full,
    m_dot_total,
    T_air_in=293.15,
    T_amb=293.15,
    T_target=573.15,
    dis_time=2 * 3600,
    dt=5.0,
    d_p=0.02,
    half_method="average_two_ends",
    save_every=60,
    verbose=True,
):
    """
    Final 2D discharging model with air marching cell-by-cell along z.

    The air temperature rise in each cell is computed with a local NTU:
        NTU_cell = h_v * dz / (G * cp_air)

    where G = m_dot_half / A_cross is the superficial mass flux.
    Therefore, increasing mass flow rate reduces the temperature rise
    per cell, as expected physically.
    """

    eps = self.e
    m_dot_half = 0.5 * m_dot_total

    if m_dot_total <= 0:
        raise ValueError("m_dot_total must be positive.")

    # Initial solid temperature from final charging map
    Ts = self._prepare_half_tank_initial(T_s_init_full, method=half_method)
    Nz_half = Ts.shape[1]

    H_half = self.H_TES / 2.0
    dz_half = H_half / Nz_half

    # Cell centers for solid/cell-average air
    z_half = (np.arange(Nz_half) + 0.5) * dz_half

    # Cell edges for the flowing air: inlet z=0, outlet z=H_half
    z_edges = np.linspace(0.0, H_half, Nz_half + 1)

    # Geometry
    A_cross = np.pi * self.R_TES**2
    W_half = self._volume_weights_half(Nz_half, dz_half)

    # External losses
    h_eq_side = self.heq_from_Pamb(T_amb)
    h_eq_end = self.heq_top_bottom_from_Pamb(T_amb)

    # Time
    n_steps = int(np.ceil(dis_time / dt))
    time = np.arange(n_steps + 1) * dt

    # Histories
    T_s_mean = np.zeros(n_steps + 1)
    T_s_max = np.zeros(n_steps + 1)
    T_s_min = np.zeros(n_steps + 1)

    T_out = np.zeros(n_steps + 1)
    h_out = np.zeros(n_steps + 1)

    P_rec = np.zeros(n_steps + 1)
    E_rec = np.zeros(n_steps + 1)

    z_target = np.full(n_steps + 1, np.nan)

    qsf_mean = np.zeros(n_steps + 1)
    qsf_max = np.zeros(n_steps + 1)

    h_in = float(h_air(T_air_in))

    # Initial air fields, only for storage/plotting
    Tf = np.full_like(Ts, float(T_air_in))
    hf = h_air(Tf)

    Tf_edges_initial = np.full((self.Nr, Nz_half + 1), float(T_air_in))

    snapshots_Ts = [Ts.copy()]
    snapshots_Tf = [Tf.copy()]
    snapshots_Tf_edges = [Tf_edges_initial.copy()]
    snapshots_qsf = []
    snapshot_times = [0.0]

    # Initial values before discharge starts
    T_s_mean[0] = np.sum(Ts * W_half) / np.sum(W_half)
    T_s_max[0] = np.max(Ts)
    T_s_min[0] = np.min(Ts)

    h_out[0] = h_in
    T_out[0] = T_air_in
    P_rec[0] = 0.0
    E_rec[0] = 0.0

    G = m_dot_half / A_cross  # [kg/m2/s]

    if verbose:
        rho_ref = float(rho_air(T_air_in))
        u_D_ref = m_dot_half / (rho_ref * A_cross)
        u_int_ref = u_D_ref / eps

        print("========== DISCHARGE SETUP ==========")
        print(f"H_full  = {self.H_TES:.3f} m = {100*self.H_TES:.1f} cm")
        print(f"H_half  = {H_half:.3f} m = {100*H_half:.1f} cm")
        print(f"Nz_half = {Nz_half}")
        print(f"dz_half = {dz_half:.4f} m = {100*dz_half:.2f} cm")
        print(f"m_dot_total = {m_dot_total:.4f} kg/s")
        print(f"m_dot_half  = {m_dot_half:.4f} kg/s")
        print(f"G = {G:.4f} kg/m2/s")
        print(f"u_D_ref     = {u_D_ref:.4f} m/s")
        print(f"u_int_ref   = {u_int_ref:.4f} m/s")
        print("Air is marched cell-by-cell along z using local NTU.")
        print("=====================================")

    for n in range(n_steps):

        # ----------------------------------------------------
        # 1) Solid conduction + ambient losses
        # ----------------------------------------------------
        Ts_star = Ts.copy()

        for i in range(self.Nr):
            r_i = self.r[i]

            for j in range(Nz_half):

                cp_s_ij = float(self.cp_magnetite_raw(Ts[i, j]))
                k_s_ij = float(self.k_magnetite_raw(Ts[i, j]))
                rhoCp_s_ij = (1.0 - eps) * self.rho_s * cp_s_ij

                # Axial boundary conditions
                if j == 0:
                    # Physical tank end: loss to ambient
                    T_jp1 = Ts[i, 1]
                    T_jm1 = T_jp1 - 2.0 * dz_half * (h_eq_end / k_s_ij) * (Ts[i, j] - T_amb)

                elif j == Nz_half - 1:
                    # Center plane: symmetry
                    T_jm1 = Ts[i, Nz_half - 2]
                    T_jp1 = T_jm1

                else:
                    T_jm1 = Ts[i, j - 1]
                    T_jp1 = Ts[i, j + 1]

                d2z = (T_jp1 - 2.0 * Ts[i, j] + T_jm1) / dz_half**2

                # Radial boundary conditions
                if i == 0:
                    radial = 4.0 * (Ts[1, j] - Ts[0, j]) / self.dr**2

                elif i == self.Nr - 1:
                    T_ghost = Ts[i - 1, j] - 2.0 * self.dr * (h_eq_side / k_s_ij) * (Ts[i, j] - T_amb)

                    d2r = (T_ghost - 2.0 * Ts[i, j] + Ts[i - 1, j]) / self.dr**2
                    d1r = (T_ghost - Ts[i - 1, j]) / (2.0 * self.dr)

                    radial = d2r + d1r / r_i

                else:
                    d2r = (Ts[i + 1, j] - 2.0 * Ts[i, j] + Ts[i - 1, j]) / self.dr**2
                    d1r = (Ts[i + 1, j] - Ts[i - 1, j]) / (2.0 * self.dr)

                    radial = d2r + d1r / r_i

                laplacian = radial + d2z
                Ts_star[i, j] = Ts[i, j] + dt / rhoCp_s_ij * (k_s_ij * laplacian)

        # ----------------------------------------------------
        # 2) Air marching along z using local NTU
        # ----------------------------------------------------
        Ts_new = Ts_star.copy()

        Tf_cell = np.zeros_like(Ts_star)
        hf_cell = np.zeros_like(Ts_star)

        Tf_edges = np.zeros((self.Nr, Nz_half + 1))
        hf_edges = np.zeros((self.Nr, Nz_half + 1))

        qsf = np.zeros_like(Ts_star)

        Tf_edges[:, 0] = T_air_in
        hf_edges[:, 0] = h_in

        for i in range(self.Nr):

            T_air_in_cell = float(T_air_in)
            h_air_in_cell = float(h_in)

            for j in range(Nz_half):

                Ts_ij = float(Ts_star[i, j])
                T_air_out_guess = T_air_in_cell

                # Iterate because properties depend on air temperature
                for _ in range(3):

                    T_air_mean = 0.5 * (T_air_in_cell + T_air_out_guess)

                    Ts_tmp = np.array([[Ts_ij]])
                    Tf_tmp = np.array([[T_air_mean]])

                    _, h_v_tmp, _, _, _ = self._gunn_hv(
                        Ts_tmp,
                        Tf_tmp,
                        m_dot_half=m_dot_half,
                        d_p=d_p,
                    )

                    h_v_ij = float(h_v_tmp[0, 0])
                    cp_flow = float(cp_air(T_air_mean))

                    NTU_cell = h_v_ij * dz_half / (G * cp_flow)

                    T_air_out_guess = Ts_ij - (Ts_ij - T_air_in_cell) * np.exp(-NTU_cell)

                T_air_out_cell = float(T_air_out_guess)
                h_air_out_cell = float(h_air(T_air_out_cell))

                # Heat transferred to air per packed-bed volume [W/m3]
                q_to_air = G * (h_air_out_cell - h_air_in_cell) / dz_half

                # Solid update due to heat removed by air
                cp_s_ij = float(self.cp_magnetite_raw(Ts_ij))
                Cs_vol = (1.0 - eps) * self.rho_s * cp_s_ij
                Ts_new[i, j] = Ts_ij - dt * q_to_air / Cs_vol

                # Store average air state inside the cell
                hf_cell[i, j] = 0.5 * (h_air_in_cell + h_air_out_cell)
                Tf_cell[i, j] = float(T_from_h_air(hf_cell[i, j]))

                # Store edge state
                hf_edges[i, j + 1] = h_air_out_cell
                Tf_edges[i, j + 1] = T_air_out_cell

                qsf[i, j] = q_to_air

                # March to next cell
                T_air_in_cell = T_air_out_cell
                h_air_in_cell = h_air_out_cell

        Ts = Ts_new
        Tf = Tf_cell
        hf = hf_cell

        # ----------------------------------------------------
        # 3) Integral outputs
        # ----------------------------------------------------
        T_s_mean[n + 1] = np.sum(Ts * W_half) / np.sum(W_half)
        T_s_max[n + 1] = np.max(Ts)
        T_s_min[n + 1] = np.min(Ts)

        h_out[n + 1] = self._area_average_outlet(hf_edges[:, -1])
        T_out[n + 1] = T_from_h_air(h_out[n + 1])

        P_rec[n + 1] = m_dot_total * (h_out[n + 1] - h_in)
        E_rec[n + 1] = E_rec[n] + P_rec[n + 1] * dt

        # Target distance from air edge-temperature profile
        h_edges_profile = self._area_average_profile_z(hf_edges)
        T_edges_profile = T_from_h_air(h_edges_profile)

        z_target[n + 1] = _z_target_from_edges(
            z_edges,
            T_edges_profile,
            T_target,
        )

        qsf_mean[n + 1] = np.sum(qsf * W_half) / np.sum(W_half)
        qsf_max[n + 1] = np.max(qsf)

        if ((n + 1) % save_every == 0) or (n == n_steps - 1):
            snapshots_Ts.append(Ts.copy())
            snapshots_Tf.append(Tf.copy())
            snapshots_Tf_edges.append(Tf_edges.copy())
            snapshots_qsf.append(qsf.copy())
            snapshot_times.append((n + 1) * dt)

    return {
        "time": time,
        "z_half": z_half,
        "z_edges": z_edges,

        "T_s_mean": T_s_mean,
        "T_s_max": T_s_max,
        "T_s_min": T_s_min,

        "T_out": T_out,
        "h_out": h_out,

        "P_rec": P_rec,
        "E_rec": E_rec,

        "z_target": z_target,
        "z_target_cm": z_target * 100.0,

        "qsf_mean": qsf_mean,
        "qsf_max": qsf_max,

        "T_s_final": Ts,
        "T_f_final": Tf,
        "h_f_final": hf,

        "snapshots_Ts": snapshots_Ts,
        "snapshots_Tf": snapshots_Tf,
        "snapshots_Tf_edges": snapshots_Tf_edges,
        "snapshots_qsf": snapshots_qsf,
        "snapshot_times": np.array(snapshot_times),

        "m_dot_total": m_dot_total,
        "m_dot_half": m_dot_half,
        "T_air_in": T_air_in,
        "T_target": T_target,

        "h_eq_side": h_eq_side,
        "h_eq_end": h_eq_end,
        "half_method": half_method,
    }


# ============================================================
# ATTACH FUNCTION TO YOUR EXISTING CLASS
# ============================================================

def attach_discharging_model(IHTES2D_Tdep):
    """
    Attach the discharging model to the existing charging class.
    """

    IHTES2D_Tdep._area_weights_radial = _area_weights_radial
    IHTES2D_Tdep._volume_weights_half = _volume_weights_half
    IHTES2D_Tdep._area_average_outlet = _area_average_outlet
    IHTES2D_Tdep._area_average_profile_z = _area_average_profile_z
    IHTES2D_Tdep._prepare_half_tank_initial = _prepare_half_tank_initial
    IHTES2D_Tdep._gunn_hv = _gunn_hv

    IHTES2D_Tdep.discharging_2D_air_final = discharging_2D_air_final

    print("Discharging model attached to IHTES2D_Tdep.")
