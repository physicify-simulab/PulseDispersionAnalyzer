import json
import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import io
import base64
import pandas as pd
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

# Import the calculation logic from phase_calculator_logic.py [[2]]
from phase_calculator_logic import calculate_pulse_properties, calculate_fwhm
from scipy.constants import c

# --- Matplotlib Setup ---
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server-side plotting
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)

# --- Default Parameters ---
DEFAULT_PARAMS = {
    "lambda0_nm": 1030.0,
    "bandwidth_nm": 30.0,
    "phi0": 0.0,
    "phi1": 0.0,
    "phi2": 0.0,
    "phi3": 0.0,
    "phi4": 0.0,
    "N_exponent": 16,
    "freq_window_factor": 1000.0,
    "cropping_fwhm_multiplier": 4.0
}
C_UM_FS = c * 1e6 / 1e15

# --- Plotting Helper with Metadata Extraction ---
def generate_plot_with_metadata(plot_func, *args, **kwargs):
    """
    Generates a plot, extracts its metadata (data limits and pixel bounds),
    and returns the base64 image string and the metadata dictionary.
    """
    fig, axs = plot_func(*args, **kwargs)
    fig.canvas.draw()

    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]

    plot_metadata = {
        "subplots": [],
        "image_size_px": [fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi]
    }

    for ax in axs:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bbox = ax.get_window_extent().bounds
        y0_from_top = (fig.get_figheight() * fig.dpi) - (bbox[1] + bbox[3])
        subplot_info = {
            "xlim": list(xlim),
            "ylim": list(ylim),
            "pixel_bbox": [bbox[0], y0_from_top, bbox[2], bbox[3]]
        }
        plot_metadata["subplots"].append(subplot_info)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    return f"data:image/png;base64,{img_str}", plot_metadata

# --- Plotting Functions (as defined in your original files) ---
def plot_spectral(lambda_nm_plot_axis, intensity_plot_data, phase_plot_data, lambda0_eff_nm, bandwidth_eff_nm, spectrum_source_msg=""):
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    title = "Spectrum and Phase vs Wavelength"
    if spectrum_source_msg: title = f"{spectrum_source_msg}: {title}"
    fig.suptitle(title, fontsize=12)
    axs[0].plot(lambda_nm_plot_axis, intensity_plot_data, color='blue')
    axs[0].set_ylabel("Intensity (arb.)", fontsize=10)
    axs[0].set_title(f"Spectrum (Derived λ₀={lambda0_eff_nm:.2f} nm, Δλ={bandwidth_eff_nm:.2f} nm)", fontsize=11)
    axs[0].grid(True, linestyle=':')
    axs[0].set_ylim(-0.05, 1.1)
    axs[1].plot(lambda_nm_plot_axis, phase_plot_data, color='purple')
    axs[1].set_xlabel("Wavelength (nm)", fontsize=10)
    axs[1].set_ylabel("Phase (rad)", fontsize=10)
    axs[1].set_title("Phase (Taylor expansion around effective ω₀)", fontsize=11)
    axs[1].grid(True, linestyle=':')
    min_lambda, max_lambda = np.min(lambda_nm_plot_axis), np.max(lambda_nm_plot_axis)
    if min_lambda < max_lambda:
        axs[0].set_xlim(min_lambda, max_lambda)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig, axs

def plot_time_domain(t, E_real, I_norm, phase_t, inst_freq, results, x_lim_manual=None):
    fig, axs = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True)
    fwhm_fs = results.get('fwhm_fs')
    title_suffix = f"(λ₀={results.get('lambda0_nm', '?'):.1f} nm, Δλ={results.get('bandwidth_nm', '?'):.1f} nm)"
    fig.suptitle(f"Time Domain Analysis {title_suffix}", fontsize=12)
    t_np = np.array(t)
    t_min, t_max = (x_lim_manual if x_lim_manual else (results.get('t_plot_min'), results.get('t_plot_max')))
    plot_mask = (t_np >= t_min) & (t_np <= t_max) if t_min is not None and t_max is not None else np.ones_like(t_np, dtype=bool)
    def get_padded_ylim(data, pad_factor=0.1):
        masked_data = np.array(data)[plot_mask]
        finite_data = masked_data[np.isfinite(masked_data)]
        if len(finite_data) == 0: return None
        y_min, y_max = np.min(finite_data), np.max(finite_data)
        if abs(y_max - y_min) < 1e-9: y_min, y_max = y_min - 0.5, y_max + 0.5
        padding = (y_max - y_min) * pad_factor
        return y_min - padding, y_max + padding
    axs[0].plot(t_np, E_real, color='dodgerblue', linewidth=1)
    axs[0].set_ylabel("Re(E(t))", fontsize=10)
    axs[0].set_title("Electric Field", fontsize=11)
    axs[0].grid(True, linestyle=':', alpha=0.7)
    e_field_ylim = get_padded_ylim(E_real)
    if e_field_ylim: axs[0].set_ylim(e_field_ylim)
    title_int = "Intensity Profile"
    if fwhm_fs is not None: title_int += f" (FWHM: {fwhm_fs:.2f} fs)"
    axs[1].plot(t_np, I_norm, color='red', linewidth=1.5)
    axs[1].set_ylabel("Norm. Intensity", fontsize=10)
    axs[1].set_title(title_int, fontsize=11)
    axs[1].grid(True, linestyle=':', alpha=0.7)
    axs[1].set_ylim(-0.05, 1.1)
    if x_lim_manual is None and fwhm_fs is not None and results.get('t_fwhm_left') is not None and results.get('t_fwhm_right') is not None:
        t_l, t_r = results['t_fwhm_left'], results['t_fwhm_right']
        axs[1].plot([t_l, t_r], [0.5, 0.5], 'k--', linewidth=1)
        axs[1].plot([t_l, t_r], [0.5, 0.5], 'k|', markersize=8, markeredgewidth=1.5)
        axs[1].annotate('', xy=(t_l, 0.6), xytext=(t_r, 0.6), arrowprops=dict(arrowstyle='<->', color='black', linewidth=1))
        axs[1].text((t_l + t_r) / 2, 0.62, f'{fwhm_fs:.2f} fs', ha='center', va='bottom', fontsize=9)
    axs[2].plot(t_np, phase_t, color='magenta', linewidth=1.5)
    axs[2].set_ylabel("Phase (rad)", fontsize=10)
    axs[2].set_title("Temporal Phase", fontsize=11)
    axs[2].grid(True, linestyle=':', alpha=0.7)
    phase_ylim = get_padded_ylim(phase_t)
    if phase_ylim: axs[2].set_ylim(phase_ylim)
    axs[3].plot(t_np, inst_freq, color='green', linewidth=1.5)
    axs[3].set_xlabel("Time (fs)", fontsize=10)
    axs[3].set_ylabel("ω_inst (rad/fs)", fontsize=10)
    axs[3].set_title("Instantaneous Frequency", fontsize=11)
    axs[3].grid(True, linestyle=':', alpha=0.7)
    if results.get('omega0_rad_fs') is not None:
        axs[3].axhline(results['omega0_rad_fs'], color='grey', linestyle=':', linewidth=1)
    inst_freq_ylim = get_padded_ylim(inst_freq)
    if inst_freq_ylim: axs[3].set_ylim(inst_freq_ylim)
    if t_min is not None and t_max is not None:
        for ax_ in axs: ax_.set_xlim(t_min, t_max)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, axs

def plot_autocorrelation(t, autocorr_norm, results, x_lim_manual=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    fwhm_ac = results.get('fwhm_ac')
    title_ac = "Intensity Autocorrelation"
    if fwhm_ac is not None: title_ac += f" (FWHM: {fwhm_ac:.2f} fs)"
    ax.plot(t, autocorr_norm, color='darkorange', linewidth=1.5)
    ax.set_xlabel("Time Delay τ (fs)", fontsize=10)
    ax.set_ylabel("Norm. Autocorr.", fontsize=10)
    ax.set_title(title_ac, fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_ylim(-0.05, 1.1)
    if x_lim_manual is None and fwhm_ac is not None and results.get('tau_fwhm_left') is not None and results.get('tau_fwhm_right') is not None:
        tau_l, tau_r = results['tau_fwhm_left'], results['tau_fwhm_right']
        ax.plot([tau_l, tau_r], [0.5, 0.5], 'k--', linewidth=1)
        ax.plot([tau_l, tau_r], [0.5, 0.5], 'k|', markersize=8, markeredgewidth=1.5)
        ax.annotate('', xy=(tau_l, 0.6), xytext=(tau_r, 0.6), arrowprops=dict(arrowstyle='<->', color='black', linewidth=1))
        ax.text(0, 0.62, f'{fwhm_ac:.2f} fs', ha='center', va='bottom', fontsize=9)
    ac_min, ac_max = (x_lim_manual if x_lim_manual else (results.get('ac_plot_min'), results.get('ac_plot_max')))
    if ac_min is not None and ac_max is not None: ax.set_xlim(ac_min, ac_max)
    plt.tight_layout()
    return fig, ax

# --- Flask Routes ---
@app.route('/')
def intro_page():
    return render_template('intro.html')

@app.route('/tool')
def calculator_page():
    is_development = os.environ.get('APP_ENV') == 'development'
    return render_template('index.html', params=DEFAULT_PARAMS, is_development=is_development)


@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        form_data = request.form
        n_exponent = form_data.get('N_exponent', DEFAULT_PARAMS['N_exponent'], type=int)
        
        is_development_env = os.environ.get('APP_ENV') == 'development'
        if not is_development_env and n_exponent > 22:
            raise ValueError("Server resource limit: Maximum exponent for Grid Size N is 22.")
        
        params = {
            "phi0": form_data.get('phi0', DEFAULT_PARAMS['phi0'], type=float),
            "phi1": form_data.get('phi1', DEFAULT_PARAMS['phi1'], type=float),
            "phi2": form_data.get('phi2', DEFAULT_PARAMS['phi2'], type=float),
            "phi3": form_data.get('phi3', DEFAULT_PARAMS['phi3'], type=float),
            "phi4": form_data.get('phi4', DEFAULT_PARAMS['phi4'], type=float),
            "N_exponent": n_exponent,
            "freq_window_factor": form_data.get('freq_window_factor', DEFAULT_PARAMS['freq_window_factor'], type=float)
        }

        spectrum_type = form_data.get('spectrum_type', 'gaussian')
        lambda0_eff_nm, bandwidth_eff_nm = None, None
        omega_custom_abs, E_omega_amp_custom = None, None
        imported_filename_display, spectrum_source_msg, analytical_spectrum_shape, cropping_details_msg = None, "", None, None

        if spectrum_type == 'file':
            spectrum_source_msg = "Imported Spectrum"
            if 'spectrum_file' not in request.files or not request.files['spectrum_file'].filename: raise ValueError("No spectrum file selected for import.")
            file = request.files['spectrum_file']
            imported_filename_display = secure_filename(file.filename)
            delimiter = form_data.get('delimiter', ',')
            if delimiter == "\\s+": delimiter = r"\s+"
            skip_rows = form_data.get('skip_rows', 0, type=int)
            x_multiplier = form_data.get('x_multiplier', 1.0, type=float)
            x_exponent = form_data.get('x_exponent', 1.0, type=float)
            cropping_fwhm_multiplier = form_data.get('cropping_fwhm_multiplier', DEFAULT_PARAMS['cropping_fwhm_multiplier'], type=float)
            try:
                df = pd.read_csv(file.stream, sep=delimiter, header=None, skiprows=skip_rows, comment='#', usecols=[0, 1], skipinitialspace=True)
                x_raw, y_raw_intensity = pd.to_numeric(df.iloc[:, 0], 'coerce').values, pd.to_numeric(df.iloc[:, 1], 'coerce').values
            except Exception as e: raise ValueError(f"Error parsing file '{imported_filename_display}': {e}")
            x_nm_processed = x_raw**x_exponent * x_multiplier
            valid_indices = np.isfinite(x_nm_processed) & np.isfinite(y_raw_intensity) & (y_raw_intensity >= 0)
            if np.sum(valid_indices) < 2: raise ValueError("Not enough valid data points in file.")
            sort_indices = np.argsort(x_nm_processed[valid_indices])
            x_nm_sorted_full, y_intensity_sorted_full = x_nm_processed[valid_indices][sort_indices], y_raw_intensity[valid_indices][sort_indices]
            y_amp_full_normalized = np.sqrt(y_intensity_sorted_full)
            if np.max(y_amp_full_normalized) > 1e-12: y_amp_full_normalized /= np.max(y_amp_full_normalized)
            x_nm_for_calculation, y_amp_for_calculation = x_nm_sorted_full, y_amp_full_normalized
            fwhm_for_cropping, _, _ = calculate_fwhm(x_nm_for_calculation, y_amp_for_calculation**2)
            if not np.isnan(fwhm_for_cropping) and fwhm_for_cropping > 1e-9:
                peak_idx = np.argmax(y_amp_for_calculation**2)
                lambda_peak = x_nm_for_calculation[peak_idx]
                window_half = (cropping_fwhm_multiplier / 2.0) * fwhm_for_cropping
                crop_mask = (x_nm_for_calculation >= lambda_peak - window_half) & (x_nm_for_calculation <= lambda_peak + window_half)
                if np.sum(crop_mask) >= 2:
                    x_nm_for_calculation, y_amp_for_calculation = x_nm_for_calculation[crop_mask], y_amp_for_calculation[crop_mask]
                    cropping_details_msg = f"Cropped to {cropping_fwhm_multiplier:.1f}xFWHM"
                else: cropping_details_msg = "Cropping failed (<2 pts), used full"
            else: cropping_details_msg = "Cropping failed (no FWHM), used full"
            intensity_final = y_amp_for_calculation**2
            if np.sum(intensity_final) < 1e-12: raise ValueError("Intensity sum from file is too low.")
            lambda0_eff_nm = np.sum(x_nm_for_calculation * intensity_final) / np.sum(intensity_final)
            bandwidth_eff_nm, _, _ = calculate_fwhm(x_nm_for_calculation, intensity_final)
            if np.isnan(bandwidth_eff_nm): raise ValueError("Could not determine bandwidth from file.")
            omega_custom_abs = 2.0 * np.pi * C_UM_FS / (x_nm_for_calculation / 1000.0)
            sort_omega_indices = np.argsort(omega_custom_abs)
            omega_custom_abs, E_omega_amp_custom = omega_custom_abs[sort_omega_indices], y_amp_for_calculation[sort_omega_indices]
            params['imported_lambda_nm'] = x_nm_for_calculation.tolist()
            params['imported_amplitude'] = y_amp_for_calculation.tolist()
        elif spectrum_type in ['gaussian', 'sech2']:
            lambda0_eff_nm = form_data.get('lambda0_nm', DEFAULT_PARAMS['lambda0_nm'], type=float)
            bandwidth_eff_nm = form_data.get('bandwidth_nm', DEFAULT_PARAMS['bandwidth_nm'], type=float)
            if lambda0_eff_nm <= 0 or bandwidth_eff_nm < 0: raise ValueError("Wavelength must be positive, bandwidth non-negative.")
            spectrum_source_msg = f"{spectrum_type.capitalize()} Spectrum"
            analytical_spectrum_shape = spectrum_type
        else:
            raise ValueError(f"Unknown spectrum type: {spectrum_type}")

        params.update({"lambda0_nm": lambda0_eff_nm, "bandwidth_nm": bandwidth_eff_nm})
        results = calculate_pulse_properties(**params, spectrum_shape=analytical_spectrum_shape, omega_custom_abs=omega_custom_abs, E_omega_amp_custom=E_omega_amp_custom)
        if not results.get('t'): raise ValueError("Calculation failed to produce time-domain data.")

        manual_time_limits = None
        if 'automatic_time_range' not in form_data:
            try:
                t_min = form_data.get('t_plot_min_manual', type=float)
                t_max = form_data.get('t_plot_max_manual', type=float)
                if t_min is not None and t_max is not None and t_min < t_max:
                    manual_time_limits = (t_min, t_max)
            except (TypeError, ValueError): pass
        
        img_spec, spec_meta = generate_plot_with_metadata(
            plot_spectral, lambda_nm_plot_axis=results['lambda_plot_nm_axis'],
            intensity_plot_data=results['spectrum_intensity_plot'], phase_plot_data=results['phase_vs_lambda_plot'],
            lambda0_eff_nm=lambda0_eff_nm, bandwidth_eff_nm=bandwidth_eff_nm, spectrum_source_msg=spectrum_source_msg
        )
        img_time, time_meta = generate_plot_with_metadata(
            plot_time_domain, t=results['t'], E_real=results['E_t_real'], I_norm=results['Intensity_t_normalized'],
            phase_t=results['phase_t'], inst_freq=results['inst_freq'], results=results, x_lim_manual=manual_time_limits
        )
        img_ac, ac_meta = generate_plot_with_metadata(
            plot_autocorrelation, t=results['t'], autocorr_norm=results['Autocorr_t_normalized'],
            results=results, x_lim_manual=manual_time_limits
        )
        
        return jsonify({
            "success": True,
            "plot_spec_img": img_spec, "plot_time_img": img_time, "plot_ac_img": img_ac,
            "plot_spec_meta": spec_meta, "plot_time_meta": time_meta, "plot_ac_meta": ac_meta,
            "fwhm_fs": results.get("fwhm_fs"), "fwhm_ac": results.get("fwhm_ac"),
            "lambda0_nm": lambda0_eff_nm, "bandwidth_nm": bandwidth_eff_nm, "N": results.get("N"),
            "spectrum_source": spectrum_source_msg, "imported_file_name": imported_filename_display,
            "cropping_details": cropping_details_msg
        })

    except ValueError as e:
        app.logger.error(f"ValueError: {e}")
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
