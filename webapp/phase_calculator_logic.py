import numpy as np
from scipy.constants import c

# --- Constants ---
C_M_S = c
C_UM_FS = C_M_S * 1e6 / 1e15 # Speed of light in um/fs

# --- FWHM Calculation Function (Robust version) ---
def calculate_fwhm(x_axis, y_signal):
    y_abs = np.abs(np.asarray(y_signal))
    x_axis = np.asarray(x_axis)

    if len(y_abs) < 2 or len(x_axis) != len(y_abs):
        return np.nan, np.nan, np.nan

    max_y = np.max(y_abs)
    if max_y < 1e-12:
        return np.nan, np.nan, np.nan

    half_max = max_y / 2.0

    sort_indices = np.argsort(x_axis)
    x_sorted = x_axis[sort_indices]
    y_sorted = y_abs[sort_indices]

    try:
        peak_idx_sorted = np.argmax(y_sorted)
    except ValueError:
        return np.nan, np.nan, np.nan

    x_left, x_right = np.nan, np.nan

    left_above_half_indices = np.where(y_sorted[:peak_idx_sorted+1] >= half_max)[0]
    if len(left_above_half_indices) > 0:
        first_above_idx = left_above_half_indices[0]
        if first_above_idx == 0:
            x_left = x_sorted[0]
        elif first_above_idx > 0:
            idx_before = first_above_idx - 1
            if abs(y_sorted[first_above_idx] - y_sorted[idx_before]) > 1e-15:
                x_left = np.interp(half_max,
                                   [y_sorted[idx_before], y_sorted[first_above_idx]],
                                   [x_sorted[idx_before], x_sorted[first_above_idx]])
            else:
                x_left = x_sorted[first_above_idx]
        # else: x_left remains np.nan, handled by final check

    right_above_half_indices_relative = np.where(y_sorted[peak_idx_sorted:] >= half_max)[0]
    if len(right_above_half_indices_relative) > 0:
        last_above_idx_relative = right_above_half_indices_relative[-1]
        last_above_idx_absolute = peak_idx_sorted + last_above_idx_relative

        if last_above_idx_absolute == len(y_sorted) - 1:
            x_right = x_sorted[-1]
        elif last_above_idx_absolute < len(y_sorted) - 1:
            idx_after = last_above_idx_absolute + 1
            if abs(y_sorted[last_above_idx_absolute] - y_sorted[idx_after]) > 1e-15:
                x_right = np.interp(half_max,
                                    [y_sorted[idx_after], y_sorted[last_above_idx_absolute]],
                                    [x_sorted[idx_after], x_sorted[last_above_idx_absolute]])
            else:
                x_right = x_sorted[last_above_idx_absolute]
        # else: x_right remains np.nan

    if not np.isnan(x_left) and not np.isnan(x_right) and x_right >= x_left:
        return x_right - x_left, x_left, x_right
    return np.nan, x_left, x_right


# --- Main Calculation Function ---
def calculate_pulse_properties(lambda0_nm, bandwidth_nm, phi0, phi1, phi2, phi3,
                               N_exponent, freq_window_factor,
                               spectrum_shape='gaussian', # New parameter: 'gaussian' or 'sech2'
                               omega_custom_abs=None, E_omega_amp_custom=None,
                               imported_lambda_nm=None, imported_amplitude=None,
                               **kwargs):
    try:
        N = 2**int(N_exponent)
    except ValueError:
        raise ValueError("Invalid N_exponent. Must be an integer.")

    if lambda0_nm <= 0: raise ValueError("lambda0_nm (center wavelength) must be positive.")
    if N <= 1: raise ValueError("N (number of points from N_exponent) must be > 1.")
    if freq_window_factor <= 0: raise ValueError("freq_window_factor must be positive.")
    if bandwidth_nm < 0: raise ValueError("bandwidth_nm cannot be negative.") # Can be zero for delta function

    # omega0_rad_fs is the effective central angular frequency for Taylor expansion AND carrier.
    omega0_rad_fs = 2.0 * np.pi * C_UM_FS / (lambda0_nm / 1000.0)

    # delta_omega_rad_fs is FWHM of intensity spectrum (envelope)
    if lambda0_nm > 1e-9 and bandwidth_nm > 1e-9: # Avoid division by zero if lambda0_nm is tiny
        delta_omega_rad_fs = (omega0_rad_fs / lambda0_nm) * bandwidth_nm
    else: # If bandwidth_nm is zero or lambda0 is zero, treat as zero bandwidth for envelope
        delta_omega_rad_fs = 0.0

    # Determine if the source is an analytical shape or imported from file
    is_analytical_source = (omega_custom_abs is None or E_omega_amp_custom is None or len(omega_custom_abs) < 2)

    # --- Determine omega_span (total width of the *envelope* spectrum simulation window) ---
    content_width_metric = delta_omega_rad_fs # This is FWHM of intensity envelope
    if content_width_metric < 1e-15: # If bandwidth is effectively zero
        default_min_width = omega0_rad_fs * 0.05 # A small fraction of carrier for non-zero carrier
        default_min_width = max(default_min_width, 1e-4) # Ensure positive if omega0_rad_fs is also zero
        content_width_metric = max(content_width_metric, default_min_width)

    # If imported, bandwidth_nm (and thus delta_omega_rad_fs) reflects derived FWHM.
    # The actual extent of imported data might be wider.
    if not is_analytical_source:
        omega_custom_relative_to_omega0 = np.array(omega_custom_abs) - omega0_rad_fs
        if len(omega_custom_relative_to_omega0) >=2:
            min_custom_rel_omega = np.min(omega_custom_relative_to_omega0)
            max_custom_rel_omega = np.max(omega_custom_relative_to_omega0)
            actual_custom_envelope_extent = max_custom_rel_omega - min_custom_rel_omega
            content_width_metric = max(content_width_metric, actual_custom_envelope_extent)
    
    omega_span_for_envelope_resolution = freq_window_factor * content_width_metric
    omega_span_for_envelope_resolution = max(omega_span_for_envelope_resolution, 1e-3) # Ensure a minimal span

    MIN_SAMPLES_PER_CARRIER_PERIOD_IN_TIME = 5.0
    omega_span_for_carrier_sampling_in_time = 0.0
    if omega0_rad_fs > 1e-12: # Only relevant if there's a carrier
        omega_span_for_carrier_sampling_in_time = MIN_SAMPLES_PER_CARRIER_PERIOD_IN_TIME * omega0_rad_fs

    omega_span = max(omega_span_for_envelope_resolution, omega_span_for_carrier_sampling_in_time)
    omega_span = max(omega_span, 1e-3) # Final check for minimal span


    # --- Simulation Grid (Frequency for Envelope Spectrum and Time) ---
    domega_envelope = omega_span / N 
    dt = (2.0 * np.pi) / omega_span   
    if not (np.isfinite(dt) and dt > 1e-18):
         raise ValueError(f"Calculated time step 'dt' ({dt:.2e} fs) is problematic. omega_span={omega_span:.2e}, N={N}")

    _freq_hz_fft_basis = np.fft.fftfreq(N, d=dt)
    omega_relative_fft_basis = _freq_hz_fft_basis * (2.0 * np.pi) 

    t = (np.arange(N) - N // 2) * dt

    omega_relative_for_definition = np.fft.fftshift(omega_relative_fft_basis)
    
    amplitude_S_envelope = np.zeros(N, dtype=float)
    if not is_analytical_source: # Imported spectrum
        query_abs_freqs_for_imported = omega_relative_for_definition + omega0_rad_fs
        amplitude_S_envelope = np.interp(
            query_abs_freqs_for_imported, omega_custom_abs, E_omega_amp_custom, left=0, right=0
        )
    else: # Analytical spectrum (Gaussian or Sech²)
        if delta_omega_rad_fs < 1e-15: # Zero bandwidth (delta function for envelope amplitude)
            center_idx = np.argmin(np.abs(omega_relative_for_definition)) 
            if N > 0: amplitude_S_envelope[center_idx] = 1.0
        else:
            if spectrum_shape == 'gaussian':
                # CORRECTED: Amplitude envelope for Gaussian, where delta_omega_rad_fs is FWHM of Intensity
                # A(omega) = A0 * exp(-2*ln(2) * (omega / FWHM_I)^2)
                amplitude_S_envelope = np.exp(
                    -2.0 * np.log(2.0) * (omega_relative_for_definition / delta_omega_rad_fs)**2
                )
            elif spectrum_shape == 'sech2':
                # For intensity I(x) = sech^2(k*x), FWHM_I is when sech^2(k*FWHM_I/2)=0.5
                # k*FWHM_I/2 = arcsech(1/sqrt(2)) = arccosh(sqrt(2))
                # So k = 2 * arccosh(sqrt(2)) / FWHM_I
                # Amplitude E(x) = sech(k*x)
                # delta_omega_rad_fs is FWHM_I for the intensity envelope
                with np.errstate(over='ignore', invalid='ignore'):
                    arg = (2 * np.arccosh(np.sqrt(2.0)) * omega_relative_for_definition / delta_omega_rad_fs)
                    cosh_val = np.cosh(arg)
                    amplitude_S_envelope = np.where(np.isinf(cosh_val), 0, 1.0 / cosh_val)
                    amplitude_S_envelope[~np.isfinite(amplitude_S_envelope)] = 0 
            else:
                raise ValueError(f"Unknown analytical spectrum_shape: {spectrum_shape}")

    phase_S_envelope = phi0 + \
                       omega_relative_for_definition * phi1 + \
                       0.5 * (omega_relative_for_definition**2) * phi2 + \
                       (1.0/6.0) * (omega_relative_for_definition**3) * phi3

    S_envelope_complex_centered = amplitude_S_envelope * np.exp(-1j * phase_S_envelope) # convention in optics to use -1j instead of +1j

    S_envelope_complex_fft_ordered = np.fft.ifftshift(S_envelope_complex_centered)
    s_envelope_t_intermediate_fft_order = np.fft.ifft(S_envelope_complex_fft_ordered)
    s_envelope_t_centered = np.fft.fftshift(s_envelope_t_intermediate_fft_order) 

    E_t = s_envelope_t_centered * np.exp(1j * omega0_rad_fs * t)
    
    Intensity_t = np.abs(E_t)**2
    max_intensity_val = 0.0
    if len(Intensity_t) > 0 and np.any(Intensity_t):
        max_intensity_val = np.max(Intensity_t)
        
    Intensity_t_normalized = Intensity_t / max_intensity_val if max_intensity_val > 1e-16 else np.zeros_like(Intensity_t)
    fwhm_fs, t_fwhm_left, t_fwhm_right = calculate_fwhm(t, Intensity_t_normalized)

    t_center_plot = 0.0
    default_zoom_span = (t[-1] - t[0]) * 0.5 if len(t) > 1 else 100.0
    default_zoom_span = max(default_zoom_span, 50.0)

    if not np.isnan(fwhm_fs) and fwhm_fs > 1e-9:
        if not np.isnan(t_fwhm_left) and not np.isnan(t_fwhm_right):
            t_center_plot = (t_fwhm_left + t_fwhm_right) / 2.0
        elif max_intensity_val > 1e-16 and len(t) > 0:
            t_center_plot = t[np.argmax(Intensity_t_normalized)]
        zoom_span_plot = max(8 * fwhm_fs, 50.0)
    elif max_intensity_val > 1e-16 and len(t) > 0:
        t_center_plot = t[np.argmax(Intensity_t_normalized)]
        zoom_span_plot = default_zoom_span
    else:
        zoom_span_plot = default_zoom_span

    t_plot_min = t_center_plot - zoom_span_plot / 2.0
    t_plot_max = t_center_plot + zoom_span_plot / 2.0
    if len(t) > 0:
        t_plot_min = max(t_plot_min, t[0])
        t_plot_max = min(t_plot_max, t[-1])
    
    if len(t) < 2 or (len(t) > 0 and ( (t_plot_max - t_plot_min) < 1e-9 * abs(t_plot_max if abs(t_plot_max)>1e-9 else 1.0) or t_plot_min >= t_plot_max ) ):
        t_plot_min = t[0] if len(t) > 0 else -1.0
        t_plot_max = t[-1] if len(t) > 0 else 1.0
        if len(t) > 0 and ( (t_plot_max - t_plot_min) < 1e-9 * abs(t_plot_max if abs(t_plot_max)>1e-9 else 1.0) or t_plot_min >= t_plot_max):
            t_plot_min = (t[0] - 1.0) if len(t) > 0 else -1.0
            t_plot_max = (t[0] + 1.0) if len(t) > 0 else 1.0

    phase_t_np = np.full_like(t, np.nan)
    inst_freq_np = np.full_like(t, np.nan)
    if np.all(np.isfinite(E_t)) and np.max(np.abs(E_t)) > 1e-9 * (np.mean(np.abs(E_t)) + 1e-15) and len(t) > 1:
        try:
            with np.errstate(invalid='ignore'): 
                phase_t_np = np.unwrap(np.angle(E_t))
            if np.all(np.isfinite(phase_t_np)): 
                 inst_freq_np = np.gradient(phase_t_np, dt)
        except Exception: 
            pass 

    phase_plot_min, phase_plot_max = None, None
    inst_freq_plot_min, inst_freq_plot_max = None, None
    try:
        if len(t) > 0 and np.all(np.isfinite([t_plot_min, t_plot_max])) and t_plot_min < t_plot_max:
            valid_indices = np.where((t >= t_plot_min) & (t <= t_plot_max))[0]
            if len(valid_indices) > 1:
                phase_window_data = phase_t_np[valid_indices]
                finite_phase_data = phase_window_data[np.isfinite(phase_window_data)]
                if len(finite_phase_data) > 1:
                    y_min_w, y_max_w = np.min(finite_phase_data), np.max(finite_phase_data)
                    padding = 0.1 * (y_max_w - y_min_w) if (y_max_w - y_min_w) > 1e-9 else max(0.1 * abs(y_min_w),0.1 * abs(y_max_w), 0.5)
                    phase_plot_min, phase_plot_max = y_min_w - padding, y_max_w + padding

                inst_freq_window_data = inst_freq_np[valid_indices]
                finite_inst_freq_data = inst_freq_window_data[np.isfinite(inst_freq_window_data)]
                if len(finite_inst_freq_data) > 1:
                    y_min_w, y_max_w = np.min(finite_inst_freq_data), np.max(finite_inst_freq_data)
                    padding = 0.1 * (y_max_w - y_min_w) if (y_max_w - y_min_w) > 1e-9 else max(0.1 * abs(y_min_w),0.1 * abs(y_max_w), 0.1)
                    inst_freq_plot_min, inst_freq_plot_max = y_min_w - padding, y_max_w + padding
    except Exception: pass

    Autocorr_t_normalized = np.full_like(t, np.nan)
    fwhm_ac, tau_fwhm_left, tau_fwhm_right = np.nan, np.nan, np.nan
    if np.all(np.isfinite(Intensity_t_normalized)) and len(Intensity_t_normalized) > 0 and np.max(Intensity_t_normalized) > 1e-12:
        Intensity_fft = np.fft.fft(Intensity_t_normalized)
        if np.all(np.isfinite(Intensity_fft)):
            Autocorr_fft_order = np.fft.ifft(np.abs(Intensity_fft)**2)
            Autocorr_t = np.fft.fftshift(np.real(Autocorr_fft_order))
            max_autocorr_val = 0.0
            if len(Autocorr_t) > 0 and np.any(Autocorr_t):
                 max_autocorr_val = np.max(Autocorr_t)
            Autocorr_t_normalized = Autocorr_t / max_autocorr_val if max_autocorr_val > 1e-16 else np.zeros_like(Autocorr_t)
            fwhm_ac, tau_fwhm_left, tau_fwhm_right = calculate_fwhm(t, Autocorr_t_normalized)

    ac_plot_min = t[0] if len(t)>0 else -1.0; ac_plot_max = t[-1] if len(t)>0 else 1.0
    ac_zoom_default = (t[-1] - t[0]) * 0.25 if len(t) > 1 else 100.0
    ac_zoom_default = max(ac_zoom_default, 100.0)

    if not np.isnan(fwhm_ac) and fwhm_ac > 1e-9:
        zoom_span_ac = max(8 * fwhm_ac, 100.0)
        ac_plot_min = max(-zoom_span_ac / 2.0, t[0]) if len(t)>0 else -zoom_span_ac / 2.0
        ac_plot_max = min(zoom_span_ac / 2.0, t[-1]) if len(t)>0 else zoom_span_ac / 2.0
    else:
        ac_plot_min = max(-ac_zoom_default / 2.0, t[0]) if len(t)>0 else -ac_zoom_default / 2.0
        ac_plot_max = min(ac_zoom_default / 2.0, t[-1]) if len(t)>0 else ac_zoom_default / 2.0
    
    if len(t) < 2 or (len(t) > 0 and ( (ac_plot_max - ac_plot_min) < 1e-9*abs(ac_plot_max if abs(ac_plot_max)>1e-9 else 1.0) or ac_plot_min >= ac_plot_max ) ) :
        ac_plot_min = t[0] if len(t)>0 else -1.0
        ac_plot_max = t[-1] if len(t)>0 else 1.0
        if len(t)>0 and ( (ac_plot_max - ac_plot_min) < 1e-9*abs(ac_plot_max if abs(ac_plot_max)>1e-9 else 1.0) or ac_plot_min >= ac_plot_max):
            ac_plot_min = (t[0]-1.0) if len(t)>0 else -1.0
            ac_plot_max = (t[0]+1.0) if len(t)>0 else 1.0

    plot_lambda_min_display = max(0.1, lambda0_nm - max(4*bandwidth_nm if bandwidth_nm > 1e-9 else 20.0, 20.0))
    plot_lambda_max_display = lambda0_nm + max(4*bandwidth_nm if bandwidth_nm > 1e-9 else 20.0, 20.0)

    if plot_lambda_min_display >= plot_lambda_max_display :
         plot_lambda_min_display = max(0.1, lambda0_nm * 0.75 if lambda0_nm > 0 else 0.1)
         plot_lambda_max_display = lambda0_nm * 1.25 if lambda0_nm > 0 else 2000.0
    lambda_plot_nm_axis = np.linspace(plot_lambda_min_display, plot_lambda_max_display, 500)
    omega_plot_axis_display_abs = 2.0 * np.pi * C_UM_FS / (lambda_plot_nm_axis / 1000.0)

    spectrum_intensity_plot = np.zeros_like(lambda_plot_nm_axis)
    if imported_lambda_nm is not None and imported_amplitude is not None and len(imported_lambda_nm) >= 2:
        sort_indices_disp = np.argsort(imported_lambda_nm)
        imported_lambda_nm_sorted_disp = np.array(imported_lambda_nm)[sort_indices_disp]
        imported_amplitude_sorted_disp = np.array(imported_amplitude)[sort_indices_disp]
        
        interp_field_amp_display = np.interp(lambda_plot_nm_axis,
                                             imported_lambda_nm_sorted_disp,
                                             imported_amplitude_sorted_disp, left=0, right=0)
        spectrum_intensity_plot = interp_field_amp_display**2
    else: 
        omega_plot_rel_for_analytical_disp = omega_plot_axis_display_abs - omega0_rad_fs
        if delta_omega_rad_fs > 1e-15 :
            if spectrum_shape == 'gaussian':
                # CORRECTED: Intensity envelope for Gaussian display
                # I(omega) = I0 * exp(-4*ln(2) * (omega / FWHM_I)^2)
                spectrum_intensity_plot = np.exp(-4.0 * np.log(2) * (omega_plot_rel_for_analytical_disp / delta_omega_rad_fs)**2)
            elif spectrum_shape == 'sech2':
                with np.errstate(over='ignore', invalid='ignore'):
                    arg_disp = (2 * np.arccosh(np.sqrt(2.0)) * omega_plot_rel_for_analytical_disp / delta_omega_rad_fs)
                    cosh_val_disp = np.cosh(arg_disp)
                    sech_sq_val = np.where(np.isinf(cosh_val_disp), 0, (1.0 / cosh_val_disp)**2)
                    spectrum_intensity_plot = sech_sq_val
                    spectrum_intensity_plot[~np.isfinite(spectrum_intensity_plot)] = 0
        else: 
             center_idx_plot = np.argmin(np.abs(lambda_plot_nm_axis - lambda0_nm))
             if len(spectrum_intensity_plot)>0: spectrum_intensity_plot[center_idx_plot] = 1.0


    max_spec_plot_intensity = 0.0
    if np.any(spectrum_intensity_plot) : max_spec_plot_intensity = np.max(spectrum_intensity_plot)
    if max_spec_plot_intensity > 1e-9: spectrum_intensity_plot /= max_spec_plot_intensity
    else: spectrum_intensity_plot = np.zeros_like(spectrum_intensity_plot)


    omega_plot_rel_for_phase_disp = omega_plot_axis_display_abs - omega0_rad_fs
    phase_vs_lambda_plot = phi0 + \
                           omega_plot_rel_for_phase_disp * phi1 + \
                           0.5 * (omega_plot_rel_for_phase_disp**2) * phi2 + \
                           (1.0/6.0) * (omega_plot_rel_for_phase_disp**3) * phi3

    return {
        "lambda0_nm": lambda0_nm, "bandwidth_nm": bandwidth_nm,
        "lambda_plot_nm_axis": lambda_plot_nm_axis.tolist(), "spectrum_intensity_plot": spectrum_intensity_plot.tolist(),
        "phase_vs_lambda_plot": phase_vs_lambda_plot.tolist(),
        "t": t.tolist(), "E_t_real": np.real(E_t).tolist(), "Intensity_t_normalized": Intensity_t_normalized.tolist(),
        "phase_t": phase_t_np.tolist(), "inst_freq": inst_freq_np.tolist(),
        "fwhm_fs": fwhm_fs if not np.isnan(fwhm_fs) else None,
        "t_fwhm_left": t_fwhm_left if not np.isnan(t_fwhm_left) else None,
        "t_fwhm_right": t_fwhm_right if not np.isnan(t_fwhm_right) else None,
        "t_plot_min": t_plot_min, "t_plot_max": t_plot_max,
        "phase_plot_min": phase_plot_min, "phase_plot_max": phase_plot_max,
        "inst_freq_plot_min": inst_freq_plot_min, "inst_freq_plot_max": inst_freq_plot_max,
        "Autocorr_t_normalized": Autocorr_t_normalized.tolist(),
        "fwhm_ac": fwhm_ac if not np.isnan(fwhm_ac) else None,
        "tau_fwhm_left": tau_fwhm_left if not np.isnan(tau_fwhm_left) else None,
        "tau_fwhm_right": tau_fwhm_right if not np.isnan(tau_fwhm_right) else None,
        "ac_plot_min": ac_plot_min, "ac_plot_max": ac_plot_max,
        "N": N, "omega0_rad_fs": omega0_rad_fs,
        "spectrum_shape_used": spectrum_shape if is_analytical_source else "file" 
    }