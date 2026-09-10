# -*- coding: utf-8 -*-
"""
cycles_same_Tmean_target.py

Charge/discharge cycle model for the SAME FINAL MEAN PACKED-BED TEMPERATURE comparison.

This file does NOT modify:
    - class_IHTES.py
    - class_IHTES_2D.py
    - class_IHTES_2D_Tdep.py
    - discharging_model.py

It only attaches additional functions to IHTES2D_Tdep.

Main idea:
    1. Charge until the volume-averaged packed-bed temperature reaches a target value.
    2. Discharge until outlet air temperature drops below a chosen threshold.
    3. Rebuild the full packed-bed temperature map from the discharged half-bed.
    4. Repeat for several cycles.

Author: Lorenzo de Nitto
"""

import numpy as np

from discharging_model import attach_discharging_model


J_PER_KWH = 3.6e6


# ============================================================
# BASIC HELPERS
# ============================================================

def _as_full_temperature_map(self, T_init):
    """
    Convert initial condition into a full 2D temperature map.

    Accepted inputs:
        - scalar temperature [K]
        - full 2D map with shape (Nr, Nz)
    """
    if np.isscalar(T_init):
        return np.full((self.Nr, self.Nz), float(T_init))

    T_init = np.asarray(T_init, dtype=float)

    if T_init.shape != (self.Nr, self.Nz):
        raise ValueError(
            f"T_init must be a scalar or a full map with shape {(self.Nr, self.Nz)}, "
            f"but got {T_init.shape}"
        )

    return T_init.copy()


def rebuild_full_map_from_half(self, T_half):
    """
    Rebuild the full packed-bed temperature map from the discharged half-domain.

    During discharge, the simulated half-domain is:
        physical end -> center plane

    The full packed bed for the following charging phase is reconstructed as:
        end -> center -> end

    Therefore, the second half is obtained by mirroring the first one.
    """
    T_half = np.asarray(T_half, dtype=float)

    if T_half.shape[0] != self.Nr:
        raise ValueError(
            f"T_half must have Nr={self.Nr} rows, but got {T_half.shape[0]}"
        )

    T_full = np.concatenate([T_half, T_half[:, ::-1]], axis=1)

    if T_full.shape != (self.Nr, self.Nz):
        raise ValueError(
            f"Rebuilt full map has shape {T_full.shape}, "
            f"but expected {(self.Nr, self.Nz)}. "
            "This requires an even number of axial cells Nz."
        )

    return T_full


def _volume_integral_full(self, M):
    """
    Axisymmetric volume integral over the full packed-bed domain.
    """
    W = self.volume_weights()
    return float(np.sum(np.asarray(M, dtype=float) * W))


def _compute_charging_powers_same_Tmean(
    self,
    freq,
    I,
    q_src,
    T_coil=333.15,
    electrical_input="coil",
):
    """
    Compute charging powers.

    P_ind:
        power actually generated inside the packed bed [W]

    P_coil:
        Joule losses in the coil [W]

    P_conv:
        converter losses [W], only if electrical_input = "grid"

    P_input:
        electrical input power used for efficiency calculations [W]

    Options:
        electrical_input = "coil"
            P_input = P_ind + P_coil

        electrical_input = "grid"
            P_input = (P_ind + P_coil) / eta_conv(freq)
    """
    P_ind = self._volume_integral_full(q_src)
    P_coil = float(self.Pcoil(freq, I, T_coil))

    P_el_coil = P_ind + P_coil

    if electrical_input == "coil":
        P_conv = 0.0
        P_input = P_el_coil

    elif electrical_input == "grid":
        eta_conv = float(self.eta_conv(freq))
        if eta_conv <= 0:
            raise ValueError(
                f"eta_conv(freq={freq}) is not positive. "
                "Cannot compute grid electrical input."
            )
        P_input = P_el_coil / eta_conv
        P_conv = P_input - P_el_coil

    else:
        raise ValueError("electrical_input must be 'coil' or 'grid'")

    return P_ind, P_coil, P_conv, P_input


# ============================================================
# CHARGING UNTIL SAME MEAN PACKED-BED TEMPERATURE
# ============================================================

def charging_2D_until_Tmean_from_map(
    self,
    freq,
    I,
    T_init_full,
    T_mean_target_C,
    T_amb=293.15,
    T_coil=333.15,
    dt=10.0,
    max_ch_time=8 * 3600,
    source_mode="total",
    electrical_input="coil",
    T_ref_source=None,
    Tmax_safety_C=None,
    save_every=100,
    verbose=True,
):
    """
    Charging simulation starting from a full 2D temperature map.

    Stop criterion:
        volume-averaged packed-bed temperature reaches T_mean_target_C.

    The local heat source is selected through source_mode:
        "hys_only"  -> hysteresis contribution only
        "eddy_only" -> eddy current contribution only
        "total"     -> hysteresis + eddy

    The electrical input used for efficiency calculations can be:
        electrical_input = "coil"
            E_input = integral(P_ind + P_coil) dt

        electrical_input = "grid"
            E_input = integral((P_ind + P_coil)/eta_conv) dt
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    if max_ch_time <= 0:
        raise ValueError("max_ch_time must be positive.")

    T = self._as_full_temperature_map(T_init_full)

    T_mean_target_K = float(T_mean_target_C) + 273.15

    if T_ref_source is None:
        T_ref_source = self.volume_average(T)

    if Tmax_safety_C is None:
        Tmax_safety_K = None
    else:
        Tmax_safety_K = float(Tmax_safety_C) + 273.15

    # External losses, same approach as charging_2D_nonuniform_Tdep
    h_eq_side = self.heq_from_Pamb(T_ref_source)
    h_eq_tb = self.heq_top_bottom_from_Pamb(T_ref_source)

    self.last_h_eq_side = h_eq_side
    self.last_h_eq_top_bottom = h_eq_tb

    # Initial source and powers
    q_src = self.qsource_map_Tlocal(freq=freq, I=I, T_map=T, source_mode=source_mode)

    P_ind, P_coil, P_conv, P_input = self._compute_charging_powers_same_Tmean(
        freq=freq,
        I=I,
        q_src=q_src,
        T_coil=T_coil,
        electrical_input=electrical_input,
    )

    # Histories
    time = [0.0]

    T_mean = [float(self.volume_average(T))]
    T_max = [float(np.max(T))]
    T_min = [float(np.min(T))]

    q_mean = [float(self.volume_average(q_src))]
    q_max = [float(np.max(q_src))]

    P_ind_hist = [P_ind]
    P_coil_hist = [P_coil]
    P_conv_hist = [P_conv]
    P_input_hist = [P_input]

    E_ind_hist = [0.0]
    E_coil_hist = [0.0]
    E_conv_hist = [0.0]
    E_input_hist = [0.0]

    snapshots = [T.copy()]
    snapshot_times = [0.0]

    n_steps_max = int(np.ceil(max_ch_time / dt))
    stop_reason = "max_ch_time_reached"

    if verbose:
        print("========== CHARGING: SAME TMEAN TARGET ==========")
        print(f"source_mode             = {source_mode}")
        print(f"freq                    = {freq:.3f} Hz")
        print(f"I                       = {I:.3f} A")
        print(f"electrical_input        = {electrical_input}")
        print(f"T_mean_target           = {T_mean_target_C:.2f} °C")
        print(f"T_mean initial          = {T_mean[0] - 273.15:.2f} °C")
        print(f"T_max initial           = {T_max[0] - 273.15:.2f} °C")
        print("=================================================")

    # If already above target at the start, no charging is needed
    if T_mean[0] >= T_mean_target_K:
        stop_reason = "T_mean_target_already_reached"

    else:
        for n in range(n_steps_max):

            T_old = T.copy()
            T_mean_old = T_mean[-1]
            T_max_old = T_max[-1]

            q_src = self.qsource_map_Tlocal(
                freq=freq,
                I=I,
                T_map=T,
                source_mode=source_mode,
            )

            P_ind, P_coil, P_conv, P_input = self._compute_charging_powers_same_Tmean(
                freq=freq,
                I=I,
                q_src=q_src,
                T_coil=T_coil,
                electrical_input=electrical_input,
            )

            if P_input <= 0:
                stop_reason = "P_input_non_positive"
                break

            dt_eff = float(dt)
            Tnew = T.copy()

            for i in range(self.Nr):
                r_i = self.r[i]

                for j in range(self.Nz):

                    rhoCp_ij = float(self.rhoCp_eff_T(T[i, j]))
                    k_ij = float(self.k_magnetite_raw(T[i, j]))

                    # ----------------------------------------
                    # Axial conduction + top/bottom losses
                    # ----------------------------------------
                    if j == 0:
                        T_jp1 = T[i, 1]
                        T_jm1 = T_jp1 - 2.0 * self.dz * (h_eq_tb / k_ij) * (T[i, j] - T_amb)

                    elif j == self.Nz - 1:
                        T_jm1 = T[i, self.Nz - 2]
                        T_jp1 = T_jm1 - 2.0 * self.dz * (h_eq_tb / k_ij) * (T[i, j] - T_amb)

                    else:
                        T_jm1 = T[i, j - 1]
                        T_jp1 = T[i, j + 1]

                    d2z = (T_jp1 - 2.0 * T[i, j] + T_jm1) / self.dz**2

                    # ----------------------------------------
                    # Radial conduction + side losses
                    # ----------------------------------------
                    if i == 0:
                        radial = 4.0 * (T[1, j] - T[0, j]) / self.dr**2

                    elif i == self.Nr - 1:
                        T_ghost = T[i - 1, j] - 2.0 * self.dr * (h_eq_side / k_ij) * (T[i, j] - T_amb)

                        d2r = (T_ghost - 2.0 * T[i, j] + T[i - 1, j]) / self.dr**2
                        d1r = (T_ghost - T[i - 1, j]) / (2.0 * self.dr)

                        radial = d2r + d1r / r_i

                    else:
                        d2r = (T[i + 1, j] - 2.0 * T[i, j] + T[i - 1, j]) / self.dr**2
                        d1r = (T[i + 1, j] - T[i - 1, j]) / (2.0 * self.dr)

                        radial = d2r + d1r / r_i

                    laplacian = radial + d2z

                    Tnew[i, j] = T[i, j] + (dt_eff / rhoCp_ij) * (
                        k_ij * laplacian + q_src[i, j]
                    )

            T_mean_new_full_step = float(self.volume_average(Tnew))
            T_max_new_full_step = float(np.max(Tnew))

            # ------------------------------------------------
            # Decide if the step must be shortened.
            # We linearly interpolate the temperature map inside the step.
            # ------------------------------------------------
            alpha_stop = 1.0
            step_stop_reason = None

            # Safety crossing fraction
            if Tmax_safety_K is not None and T_max_new_full_step >= Tmax_safety_K:
                if T_max_new_full_step > T_max_old:
                    alpha_safety = (Tmax_safety_K - T_max_old) / (T_max_new_full_step - T_max_old)
                    alpha_safety = float(np.clip(alpha_safety, 0.0, 1.0))
                else:
                    alpha_safety = 1.0
            else:
                alpha_safety = None

            # Target crossing fraction
            if T_mean_new_full_step >= T_mean_target_K:
                if T_mean_new_full_step > T_mean_old:
                    alpha_target = (T_mean_target_K - T_mean_old) / (T_mean_new_full_step - T_mean_old)
                    alpha_target = float(np.clip(alpha_target, 0.0, 1.0))
                else:
                    alpha_target = 1.0
            else:
                alpha_target = None

            if alpha_safety is not None and alpha_target is not None:
                if alpha_safety < alpha_target:
                    alpha_stop = alpha_safety
                    step_stop_reason = "Tmax_safety_reached"
                else:
                    alpha_stop = alpha_target
                    step_stop_reason = "T_mean_target_reached"

            elif alpha_safety is not None:
                alpha_stop = alpha_safety
                step_stop_reason = "Tmax_safety_reached"

            elif alpha_target is not None:
                alpha_stop = alpha_target
                step_stop_reason = "T_mean_target_reached"

            dt_used = dt_eff * alpha_stop
            T = T_old + alpha_stop * (Tnew - T_old)

            # Update energy counters using dt_used
            E_ind_new = E_ind_hist[-1] + P_ind * dt_used
            E_coil_new = E_coil_hist[-1] + P_coil * dt_used
            E_conv_new = E_conv_hist[-1] + P_conv * dt_used
            E_input_new = E_input_hist[-1] + P_input * dt_used

            t_new = time[-1] + dt_used

            # Recompute q for stored final state after update
            q_src_after = self.qsource_map_Tlocal(
                freq=freq,
                I=I,
                T_map=T,
                source_mode=source_mode,
            )

            time.append(t_new)

            T_mean.append(float(self.volume_average(T)))
            T_max.append(float(np.max(T)))
            T_min.append(float(np.min(T)))

            q_mean.append(float(self.volume_average(q_src_after)))
            q_max.append(float(np.max(q_src_after)))

            P_ind_hist.append(P_ind)
            P_coil_hist.append(P_coil)
            P_conv_hist.append(P_conv)
            P_input_hist.append(P_input)

            E_ind_hist.append(E_ind_new)
            E_coil_hist.append(E_coil_new)
            E_conv_hist.append(E_conv_new)
            E_input_hist.append(E_input_new)

            if ((n + 1) % save_every == 0) or (step_stop_reason is not None):
                snapshots.append(T.copy())
                snapshot_times.append(t_new)

            if step_stop_reason is not None:
                stop_reason = step_stop_reason
                break

    result = {
        "time": np.array(time),

        "T_mean": np.array(T_mean),
        "T_max": np.array(T_max),
        "T_min": np.array(T_min),

        "q_mean": np.array(q_mean),
        "q_max": np.array(q_max),

        "P_ind": np.array(P_ind_hist),
        "P_coil": np.array(P_coil_hist),
        "P_conv": np.array(P_conv_hist),
        "P_input": np.array(P_input_hist),

        "E_ind": np.array(E_ind_hist),
        "E_coil": np.array(E_coil_hist),
        "E_conv": np.array(E_conv_hist),
        "E_input": np.array(E_input_hist),

        "E_ind_kWh": np.array(E_ind_hist) / J_PER_KWH,
        "E_coil_kWh": np.array(E_coil_hist) / J_PER_KWH,
        "E_conv_kWh": np.array(E_conv_hist) / J_PER_KWH,
        "E_input_kWh": np.array(E_input_hist) / J_PER_KWH,

        "T_final": T,

        "snapshots": snapshots,
        "snapshot_times": np.array(snapshot_times),

        "freq": freq,
        "I": I,
        "source_mode": source_mode,
        "electrical_input": electrical_input,
        "T_mean_target_C": T_mean_target_C,

        "stop_reason": stop_reason,
        "h_eq_side": h_eq_side,
        "h_eq_top_bottom": h_eq_tb,
    }

    if verbose:
        print("========== CHARGING RESULT ==========")
        print(f"stop reason         = {stop_reason}")
        print(f"charging time       = {result['time'][-1] / 3600:.3f} h")
        print(f"E_input_charge      = {result['E_input_kWh'][-1]:.3f} kWh")
        print(f"E_ind_charge        = {result['E_ind_kWh'][-1]:.3f} kWh")
        print(f"E_coil_loss         = {result['E_coil_kWh'][-1]:.3f} kWh")
        print(f"E_conv_loss         = {result['E_conv_kWh'][-1]:.3f} kWh")
        print(f"T_mean final        = {result['T_mean'][-1] - 273.15:.2f} °C")
        print(f"T_max final         = {result['T_max'][-1] - 273.15:.2f} °C")
        print("=====================================")

    return result


# ============================================================
# DISCHARGE UNTIL OUTLET TEMPERATURE THRESHOLD
# ============================================================

def _trim_discharge_result_to_Tout_stop(result, T_out_stop):
    """
    Trim the output of discharging_2D_air_final at the first time instant
    after T_out has fallen below the selected threshold.

    The original discharging model runs for a fixed maximum time.
    Here we use it as engine and then keep only the useful part.
    """
    T_out = np.asarray(result["T_out"], dtype=float)

    # At t=0 the outlet is set equal to inlet temperature.
    # The useful discharge starts only after T_out has reached the threshold at least once.
    idx_reached = np.where(T_out >= T_out_stop)[0]

    if len(idx_reached) == 0:
        stop_idx = 0
        stop_reason = "T_out_never_reached_threshold"

    else:
        first_reached = int(idx_reached[0])
        idx_below_after = np.where(T_out[first_reached:] < T_out_stop)[0]

        if len(idx_below_after) == 0:
            stop_idx = len(T_out) - 1
            stop_reason = "max_dis_time_reached"

        else:
            stop_idx = first_reached + int(idx_below_after[0])
            stop_reason = "T_out_below_threshold"

    trimmed = {}

    n_time = len(T_out)

    for key, value in result.items():
        if isinstance(value, np.ndarray) and value.shape[0] == n_time:
            trimmed[key] = value[:stop_idx + 1].copy()
        else:
            trimmed[key] = value

    # If save_every=1, snapshots_Ts has one snapshot for each time value.
    if "snapshots_Ts" in result and len(result["snapshots_Ts"]) > stop_idx:
        T_s_final_half = result["snapshots_Ts"][stop_idx].copy()
    else:
        # Fallback: use final map returned by the fixed-time run
        T_s_final_half = result["T_s_final"].copy()

    trimmed["T_s_final_half"] = T_s_final_half
    trimmed["T_s_final"] = T_s_final_half
    trimmed["stop_idx"] = stop_idx
    trimmed["stop_reason"] = stop_reason

    if "E_rec" in trimmed:
        trimmed["E_rec_kWh"] = trimmed["E_rec"] / J_PER_KWH

    return trimmed


def discharging_2D_air_until_Tout(
    self,
    T_s_init_full,
    m_dot_total,
    T_air_in=293.15,
    T_amb=293.15,
    T_out_stop=573.15,
    max_dis_time=4 * 3600,
    dt=5.0,
    d_p=0.02,
    half_method="average_two_ends",
    verbose=True,
):
    """
    Run the existing final discharging model and stop the useful discharge
    when outlet air temperature falls below T_out_stop.

    Internally:
        - discharging_2D_air_final is run up to max_dis_time
        - the result is trimmed at the threshold crossing
        - save_every=1 is used so that the final half-bed map at the
          threshold time is available for the next charging cycle
    """
    raw = self.discharging_2D_air_final(
        T_s_init_full=T_s_init_full,
        m_dot_total=m_dot_total,
        T_air_in=T_air_in,
        T_amb=T_amb,
        T_target=T_out_stop,
        dis_time=max_dis_time,
        dt=dt,
        d_p=d_p,
        half_method=half_method,
        save_every=1,
        verbose=verbose,
    )

    trimmed = _trim_discharge_result_to_Tout_stop(raw, T_out_stop=T_out_stop)

    if verbose:
        print("========== DISCHARGE THRESHOLD RESULT ==========")
        print(f"stop reason          = {trimmed['stop_reason']}")
        print(f"discharge time       = {trimmed['time'][-1] / 3600:.3f} h")
        print(f"E_rec                = {trimmed['E_rec_kWh'][-1]:.3f} kWh")
        print(f"T_out final          = {trimmed['T_out'][-1] - 273.15:.2f} °C")
        print(f"T_s_mean final half  = {trimmed['T_s_mean'][-1] - 273.15:.2f} °C")
        print("================================================")

    return trimmed


# ============================================================
# FULL CYCLE SIMULATION
# ============================================================

def run_same_Tmean_target_cycles(
    self,
    case_name,
    freq,
    I,
    source_mode,
    T_mean_target_C,
    n_cycles=3,
    T_init_full=293.15,
    T_amb=293.15,
    T_coil=333.15,
    electrical_input="coil",
    charging_dt=10.0,
    max_ch_time=8 * 3600,
    Tmax_safety_C=None,
    m_dot_total=0.20,
    T_air_in=293.15,
    T_out_stop=573.15,
    discharging_dt=5.0,
    max_dis_time=4 * 3600,
    d_p=0.02,
    half_method="average_two_ends",
    verbose=True,
):
    """
    Run repeated charge/discharge cycles with the same final mean packed-bed
    temperature target during each charging phase.

    For each cycle:
        1. charge until T_mean reaches T_mean_target_C
        2. discharge until T_out < T_out_stop
        3. rebuild the full packed-bed map from the discharged half-domain
        4. use that map as the initial condition for the next cycle
    """
    T_current_full = self._as_full_temperature_map(T_init_full)

    cycles = []
    summary = []

    for cyc in range(1, n_cycles + 1):

        if verbose:
            print("\n")
            print("####################################################")
            print(f"CASE: {case_name} | CYCLE {cyc}/{n_cycles}")
            print("####################################################")

        charge = self.charging_2D_until_Tmean_from_map(
            freq=freq,
            I=I,
            T_init_full=T_current_full,
            T_mean_target_C=T_mean_target_C,
            T_amb=T_amb,
            T_coil=T_coil,
            dt=charging_dt,
            max_ch_time=max_ch_time,
            source_mode=source_mode,
            electrical_input=electrical_input,
            Tmax_safety_C=Tmax_safety_C,
            verbose=verbose,
        )

        discharge = self.discharging_2D_air_until_Tout(
            T_s_init_full=charge["T_final"],
            m_dot_total=m_dot_total,
            T_air_in=T_air_in,
            T_amb=T_amb,
            T_out_stop=T_out_stop,
            max_dis_time=max_dis_time,
            dt=discharging_dt,
            d_p=d_p,
            half_method=half_method,
            verbose=verbose,
        )

        E_input_charge_kWh = float(charge["E_input_kWh"][-1])
        E_ind_charge_kWh = float(charge["E_ind_kWh"][-1])
        E_coil_charge_kWh = float(charge["E_coil_kWh"][-1])
        E_conv_charge_kWh = float(charge["E_conv_kWh"][-1])
        E_rec_kWh = float(discharge["E_rec_kWh"][-1])

        if E_input_charge_kWh > 0:
            eta_rt = E_rec_kWh / E_input_charge_kWh
        else:
            eta_rt = np.nan

        row = {
            "case": case_name,
            "cycle": cyc,
            "source_mode": source_mode,
            "freq_Hz": freq,
            "I_A": I,
            "electrical_input": electrical_input,
            "T_mean_target_C": T_mean_target_C,

            "E_input_charge_kWh": E_input_charge_kWh,
            "E_ind_charge_kWh": E_ind_charge_kWh,
            "E_coil_loss_kWh": E_coil_charge_kWh,
            "E_conv_loss_kWh": E_conv_charge_kWh,

            "charge_time_h": float(charge["time"][-1] / 3600),
            "T_mean_end_charge_C": float(charge["T_mean"][-1] - 273.15),
            "T_max_end_charge_C": float(charge["T_max"][-1] - 273.15),
            "T_min_end_charge_C": float(charge["T_min"][-1] - 273.15),

            "E_rec_discharge_kWh": E_rec_kWh,
            "discharge_time_h": float(discharge["time"][-1] / 3600),
            "T_out_final_C": float(discharge["T_out"][-1] - 273.15),
            "T_s_mean_end_discharge_C": float(discharge["T_s_mean"][-1] - 273.15),
            "T_s_max_end_discharge_C": float(discharge["T_s_max"][-1] - 273.15),
            "T_s_min_end_discharge_C": float(discharge["T_s_min"][-1] - 273.15),

            "round_trip_efficiency": float(eta_rt),
            "round_trip_efficiency_percent": float(100.0 * eta_rt),

            "charge_stop_reason": charge["stop_reason"],
            "discharge_stop_reason": discharge["stop_reason"],
        }

        cycles.append(
            {
                "cycle": cyc,
                "charge": charge,
                "discharge": discharge,
                "summary": row,
            }
        )

        summary.append(row)

        # Rebuild full map for the next cycle
        T_current_full = self.rebuild_full_map_from_half(
            discharge["T_s_final_half"]
        )

    return {
        "case_name": case_name,
        "cycles": cycles,
        "summary": summary,
        "T_final_full": T_current_full,
    }


def run_same_Tmean_target_comparison(
    self,
    cases,
    T_mean_target_C,
    n_cycles=3,
    **common_kwargs,
):
    """
    Run the same-Tmean-target cycle analysis for multiple cases.

    Example:
        cases = [
            {
                "case_name": "Hysteresis-only",
                "freq": 80,
                "I": 2203,
                "source_mode": "hys_only",
            },
            {
                "case_name": "Eddy-only",
                "freq": 50000,
                "I": 100,
                "source_mode": "eddy_only",
            },
        ]
    """
    results = {}
    summary_all = []

    for case in cases:
        res = self.run_same_Tmean_target_cycles(
            case_name=case["case_name"],
            freq=case["freq"],
            I=case["I"],
            source_mode=case["source_mode"],
            T_mean_target_C=T_mean_target_C,
            n_cycles=n_cycles,
            **common_kwargs,
        )

        results[case["case_name"]] = res
        summary_all.extend(res["summary"])

    return {
        "results": results,
        "summary": summary_all,
    }


# ============================================================
# ATTACH TO CLASS
# ============================================================

def attach_same_Tmean_cycle_model(IHTES2D_Tdep):
    """
    Attach the same-Tmean-target cycling model to IHTES2D_Tdep.

    This also attaches the existing discharging model because the cycle
    functions use discharging_2D_air_final as the discharge engine.
    """
    attach_discharging_model(IHTES2D_Tdep)

    IHTES2D_Tdep._as_full_temperature_map = _as_full_temperature_map
    IHTES2D_Tdep.rebuild_full_map_from_half = rebuild_full_map_from_half
    IHTES2D_Tdep._volume_integral_full = _volume_integral_full
    IHTES2D_Tdep._compute_charging_powers_same_Tmean = _compute_charging_powers_same_Tmean

    IHTES2D_Tdep.charging_2D_until_Tmean_from_map = charging_2D_until_Tmean_from_map
    IHTES2D_Tdep.discharging_2D_air_until_Tout = discharging_2D_air_until_Tout

    IHTES2D_Tdep.run_same_Tmean_target_cycles = run_same_Tmean_target_cycles
    IHTES2D_Tdep.run_same_Tmean_target_comparison = run_same_Tmean_target_comparison

    print("Same-Tmean-target cycle model attached to IHTES2D_Tdep.")
