# app.py
import json
import os # Imported OS to access environment variables
from flask import Flask, render_template, request, jsonify
import numpy as np
import io
import base64
import pandas as pd
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix # For production server

# Import the calculation logic
from phase_calculator_logic import calculate_pulse_properties, calculate_fwhm # Assuming this file exists in the same directory
from scipy.constants import c

# --- Matplotlib Setup ---
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server-side plotting
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 # Max file upload size: 5MB

# Apply the ProxyFix middleware.
# This tells Flask to trust the X-Script-Name header (or X-Prefix)
# sent by your Nginx proxy. This is crucial for correct URL generation in production.
# x_prefix=1 means it will look for one level of prefix.
app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1)

# --- Default Parameters ---
DEFAULT_PARAMS = {
    "lambda0_nm": 1030.0,
    "bandwidth_nm": 30.0,
    "phi0": 0.0,
    "phi1": 0.0,
    "phi2": 0.0,
    "phi3": 0.0,
    "phi4": 0.0, # Added fourth-order dispersion
    "N_exponent": 16,
    "freq_window_factor": 1000.0,
    "cropping_fwhm_multiplier": 4.0
}
C_UM_FS = c * 1e6 / 1e15 # Speed of light in micrometers per femtosecond

# --- Helper to generate Matplotlib plot and encode to base64 ---
def create_matplotlib_plot(plot_func, *args, **kwargs):
    """
    Generates a Matplotlib plot using the provided plotting function and returns it as a base64 encoded PNG.
    """
    fig, _ = plot_func(*args, **kwargs) # The plotting function should return fig, axs
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight') # Save plot to a BytesIO buffer
    plt.close(fig) # Close the figure to free memory
    buf.seek(0) # Rewind the buffer to the beginning
    img_str = base64.b64encode(buf.read()).decode('utf-8') # Encode to base64
    return f"data:image/png;base64,{img_str}" # Return as a data URI

# --- Plotting Functions ---
# These functions encapsulate the Matplotlib plotting logic.
# They are called by create_matplotlib_plot.

def plot_spectral(lambda_nm_plot_axis, intensity_plot_data, phase_plot_data, lambda0_eff_nm, bandwidth_eff_nm, spectrum_source_msg=""):
    """Plots spectral intensity and phase vs. wavelength."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    title = f"Spectrum and Phase vs Wavelength"
    if spectrum_source_msg: title = f"{spectrum_source_msg}: " + title
    fig.suptitle(title, fontsize=12)

    # Plot spectral intensity
    axs[0].plot(lambda_nm_plot_axis, intensity_plot_data, color='blue')
    axs[0].set_ylabel("Intensity (arb.)", fontsize=10)
    axs[0].set_title(f"Spectrum (Derived λ₀={lambda0_eff_nm:.2f} nm, Δλ={bandwidth_eff_nm:.2f} nm)", fontsize=11)
    axs[0].grid(True, linestyle=':')
    if len(intensity_plot_data) > 0 and np.max(intensity_plot_data) > 1e-9 :
        axs[0].set_ylim(-0.05 * np.max(intensity_plot_data), 1.1 * np.max(intensity_plot_data))
    else:
        axs[0].set_ylim(-0.05, 1.1) # Default if no significant data

    # Plot spectral phase
    axs[1].plot(lambda_nm_plot_axis, phase_plot_data, color='purple')
    axs[1].set_xlabel("Wavelength (nm)", fontsize=10)
    axs[1].set_ylabel("Phase (rad)", fontsize=10)
    axs[1].set_title(f"Phase (Taylor expansion around effective ω₀)", fontsize=11)
    axs[1].grid(True, linestyle=':')

    axs[0].tick_params(axis='both', which='major', labelsize=9)
    axs[1].tick_params(axis='both', which='major', labelsize=9)

    # Determine plot limits for wavelength axis
    min_lambda_plot_val = np.min(lambda_nm_plot_axis) if len(lambda_nm_plot_axis) > 0 else lambda0_eff_nm - 3*bandwidth_eff_nm
    max_lambda_plot_val = np.max(lambda_nm_plot_axis) if len(lambda_nm_plot_axis) > 0 else lambda0_eff_nm + 3*bandwidth_eff_nm

    # Handle cases where plot range might be invalid
    if min_lambda_plot_val >= max_lambda_plot_val or bandwidth_eff_nm <= 1e-9:
        min_lambda_plot_val = lambda0_eff_nm - max(20, 0.1 * lambda0_eff_nm)
        max_lambda_plot_val = lambda0_eff_nm + max(20, 0.1 * lambda0_eff_nm)
        if min_lambda_plot_val <=0: min_lambda_plot_val = 1.0 # Wavelength must be positive

    if min_lambda_plot_val < max_lambda_plot_val:
         axs[0].set_xlim(min_lambda_plot_val, max_lambda_plot_val)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    return fig, axs

def plot_time_domain(t, E_real, I_norm, phase_t, inst_freq, results):
    """Plots electric field, intensity, temporal phase, and instantaneous frequency vs. time."""
    fig, axs = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True)
    fwhm_fs = results.get('fwhm_fs')
    title_suffix = f"(λ₀={results.get('lambda0_nm', '?'):.1f} nm, Δλ={results.get('bandwidth_nm', '?'):.1f} nm)"
    fig.suptitle(f"Time Domain Analysis {title_suffix}", fontsize=12)

    # Plot Real Electric Field E(t)
    axs[0].plot(t, E_real, color='dodgerblue', linewidth=1)
    axs[0].set_ylabel("Re(E(t))", fontsize=10)
    axs[0].set_title("Electric Field", fontsize=11)
    axs[0].grid(True, linestyle=':', alpha=0.7)
    max_val = np.nanmax(np.abs(E_real)) if E_real is not None and len(E_real)>0 and np.any(np.isfinite(E_real)) else 1.0
    max_val = max(max_val, 1e-9) if np.isfinite(max_val) else 1.0
    axs[0].set_ylim(-1.1 * max_val, 1.1 * max_val)

    # Plot Normalized Intensity I(t)
    title_int = "Intensity Profile"
    if fwhm_fs is not None: title_int += f" (FWHM: {fwhm_fs:.2f} fs)"
    axs[1].plot(t, I_norm, color='red', linewidth=1.5)
    axs[1].set_ylabel("Norm. Intensity", fontsize=10)
    axs[1].set_title(title_int, fontsize=11)
    axs[1].grid(True, linestyle=':', alpha=0.7)
    axs[1].set_ylim(-0.05, 1.1)
    if fwhm_fs is not None and results.get('t_fwhm_left') is not None and results.get('t_fwhm_right') is not None:
        t_l, t_r = results['t_fwhm_left'], results['t_fwhm_right']
        axs[1].plot([t_l, t_r], [0.5, 0.5], 'k--', linewidth=1) # FWHM line
        axs[1].plot([t_l, t_r], [0.5, 0.5], 'k|', markersize=8, markeredgewidth=1.5) # FWHM markers
        axs[1].annotate('', xy=(t_l, 0.6), xytext=(t_r, 0.6),
                        arrowprops=dict(arrowstyle='<->', color='black', linewidth=1)) # FWHM arrow
        axs[1].text((t_l + t_r)/2, 0.62, f'{fwhm_fs:.2f} fs', ha='center', va='bottom', fontsize=9) # FWHM text

    # Plot Temporal Phase
    axs[2].plot(t, phase_t, color='magenta', linewidth=1.5)
    axs[2].set_ylabel("Phase (rad)", fontsize=10)
    axs[2].set_title("Temporal Phase", fontsize=11)
    axs[2].grid(True, linestyle=':', alpha=0.7)
    ph_min, ph_max = results.get('phase_plot_min'), results.get('phase_plot_max')
    if ph_min is not None and ph_max is not None: axs[2].set_ylim(ph_min, ph_max)

    # Plot Instantaneous Frequency
    axs[3].plot(t, inst_freq, color='green', linewidth=1.5)
    axs[3].set_xlabel("Time (fs)", fontsize=10)
    axs[3].set_ylabel("ω_inst (rad/fs)", fontsize=10)
    axs[3].set_title("Instantaneous Frequency", fontsize=11)
    axs[3].grid(True, linestyle=':', alpha=0.7)
    if results.get('omega0_rad_fs') is not None:
        axs[3].axhline(results['omega0_rad_fs'], color='grey', linestyle=':', linewidth=1, label='ω₀') # Center frequency line
    if_min, if_max = results.get('inst_freq_plot_min'), results.get('inst_freq_plot_max')
    if if_min is not None and if_max is not None: axs[3].set_ylim(if_min, if_max)

    # Set common x-axis limits for all time-domain plots
    t_min, t_max = results.get('t_plot_min'), results.get('t_plot_max')
    if t_min is not None and t_max is not None:
        for ax_ in axs: ax_.set_xlim(t_min, t_max)

    for ax_ in axs: ax_.tick_params(axis='both', which='major', labelsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
    return fig, axs

def plot_autocorrelation(t, autocorr_norm, results):
    """Plots the intensity autocorrelation vs. time delay."""
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

    # Plot FWHM for autocorrelation
    if fwhm_ac is not None and results.get('tau_fwhm_left') is not None and results.get('tau_fwhm_right') is not None:
        tau_l, tau_r = results['tau_fwhm_left'], results['tau_fwhm_right']
        ax.plot([tau_l, tau_r], [0.5, 0.5], 'k--', linewidth=1)
        ax.plot([tau_l, tau_r], [0.5, 0.5], 'k|', markersize=8, markeredgewidth=1.5)
        ax.annotate('', xy=(tau_l, 0.6), xytext=(tau_r, 0.6),
                    arrowprops=dict(arrowstyle='<->', color='black', linewidth=1))
        ax.text(0, 0.62, f'{fwhm_ac:.2f} fs', ha='center', va='bottom', fontsize=9) # Centered text for AC FWHM

    ac_min, ac_max = results.get('ac_plot_min'), results.get('ac_plot_max')
    if ac_min is not None and ac_max is not None: ax.set_xlim(ac_min, ac_max)

    ax.tick_params(axis='both', which='major', labelsize=9)
    plt.tight_layout()
    return fig, ax


# --- Flask Routes ---
# These routes define the application's endpoints.
# They are relative to the prefix defined by X-Script-Name (e.g., /PulseDispersionAnalyzer/)
# when running behind Nginx, or relative to the DispatcherMiddleware prefix when run locally.

@app.route('/')
def intro_page():
    """Serves the introduction page (intro.html)."""
    is_development_env = os.environ.get('APP_ENV') == 'development'
    return render_template('intro.html', is_development_env=is_development_env)

@app.route('/tool')
def calculator_page():
    """Serves the main calculator page (index.html) with default parameters."""
    is_development_env = os.environ.get('APP_ENV') == 'development'
    return render_template('index.html', params=DEFAULT_PARAMS, is_development_env=is_development_env)


@app.route('/calculate', methods=['POST'])
def calculate():
    """
    Handles the calculation request from the form.
    Parses input, calls the core calculation logic, generates plots, and returns results as JSON.
    """
    try:
        form_data = request.form

        # Validate N_exponent (grid size for FFT)
        n_exponent = form_data.get('N_exponent', DEFAULT_PARAMS['N_exponent'], type=int)
        
        # Check if running in a local development environment
        # The APP_ENV variable is set in run_local.py
        is_development_env = os.environ.get('APP_ENV') == 'development'
        
        if not is_development_env: # If NOT development (i.e., production or other environment)
            if n_exponent > 22: 
                # Enforce server-side limit to prevent resource exhaustion
                raise ValueError("Server resource limit: Maximum exponent for Grid Size N is 22.")
        else: # If it IS development (running locally via run_local.py)
            # No hard limit for n_exponent locally, as per user request.
            # Optionally, add a warning for very high values that might still cause local issues.
            if n_exponent > 24: # You can adjust this local warning threshold
                app.logger.warning(f"Local Warning: N_exponent ({n_exponent}) is very high and may consume significant memory/time.")
        
        # Collect phase parameters
        params = {
            "phi0": form_data.get('phi0', DEFAULT_PARAMS['phi0'], type=float),
            "phi1": form_data.get('phi1', DEFAULT_PARAMS['phi1'], type=float),
            "phi2": form_data.get('phi2', DEFAULT_PARAMS['phi2'], type=float),
            "phi3": form_data.get('phi3', DEFAULT_PARAMS['phi3'], type=float),
            "phi4": form_data.get('phi4', DEFAULT_PARAMS['phi4'], type=float), # Added fourth-order
            "N_exponent": n_exponent, # Use the (potentially unlimited locally) n_exponent
            "freq_window_factor": form_data.get('freq_window_factor', DEFAULT_PARAMS['freq_window_factor'], type=float)
        }

        spectrum_type = form_data.get('spectrum_type', 'gaussian')
        lambda0_eff_nm = None # Effective center wavelength
        bandwidth_eff_nm = None # Effective bandwidth
        omega_custom_abs = None # Custom frequency axis (for file import)
        E_omega_amp_custom = None # Custom spectral amplitude (for file import)
        imported_filename_display = None # Filename for display
        spectrum_source_msg = "" # Message indicating spectrum source (e.g., "Gaussian", "Imported")
        analytical_spectrum_shape = None # For 'gaussian' or 'sech2'
        cropping_details_msg = None # Message about file cropping

        if spectrum_type == 'file':
            spectrum_source_msg = "Imported Spectrum"
            if 'spectrum_file' not in request.files or not request.files['spectrum_file'].filename:
                raise ValueError("No spectrum file selected for import.")

            file = request.files['spectrum_file']
            imported_filename_display = secure_filename(file.filename) # Sanitize filename

            # Parameters for parsing the imported file
            delimiter = form_data.get('delimiter', ',')
            if delimiter == "\\s+": delimiter = r"\s+" # Handle regex for whitespace
            skip_rows = form_data.get('skip_rows', 0, type=int)
            x_multiplier = form_data.get('x_multiplier', 1.0, type=float)
            x_exponent = form_data.get('x_exponent', 1.0, type=float)
            cropping_fwhm_multiplier = form_data.get('cropping_fwhm_multiplier', DEFAULT_PARAMS['cropping_fwhm_multiplier'], type=float)
            if cropping_fwhm_multiplier < 0.1:
                raise ValueError("Cropping FWHM multiplier must be at least 0.1.")

            try:
                file.stream.seek(0) # Ensure reading from the start of the file stream
                # Read first two columns, skip header rows, handle comments
                df = pd.read_csv(file.stream, sep=delimiter, header=None, skiprows=skip_rows, comment='#', skipinitialspace=True, usecols=[0, 1])
                if df.shape[1] < 2:
                    raise ValueError("Imported file must have at least two columns (Wavelength, Intensity).")
                x_raw = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                y_raw_intensity = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
            except Exception as e:
                raise ValueError(f"Error parsing file '{imported_filename_display}': {str(e)}")

            # Process and validate imported data
            x_nm_processed = x_raw**x_exponent * x_multiplier # Apply transformations to X data
            # Keep only valid, finite, non-negative intensity data points
            valid_indices = np.isfinite(x_nm_processed) & np.isfinite(y_raw_intensity) & (y_raw_intensity >= 0)
            x_nm_valid_unsorted = x_nm_processed[valid_indices]
            y_intensity_valid_unsorted = y_raw_intensity[valid_indices]

            if len(x_nm_valid_unsorted) < 2: # Need at least two points for calculations
                raise ValueError("Not enough valid data points (wavelength, non-negative intensity) after processing and NaN removal from file.")

            # Sort data by wavelength
            sort_indices = np.argsort(x_nm_valid_unsorted)
            x_nm_sorted_full = x_nm_valid_unsorted[sort_indices]
            y_intensity_sorted_full = y_intensity_valid_unsorted[sort_indices]

            # Convert intensity to amplitude and normalize
            y_amp_full_normalized = np.sqrt(y_intensity_sorted_full)
            peak_amp_val = np.max(y_amp_full_normalized) if len(y_amp_full_normalized) > 0 else 0.0
            if peak_amp_val > 1e-12: # Avoid division by zero or tiny numbers
                y_amp_full_normalized /= peak_amp_val
            else:
                y_amp_full_normalized = np.zeros_like(y_amp_full_normalized) # Set to zero if no significant amplitude

            # --- Cropping Logic (Optional) ---
            x_nm_for_calculation = x_nm_sorted_full
            y_amp_for_calculation = y_amp_full_normalized
            intensity_full_spectrum_norm_for_fwhm = y_amp_full_normalized**2

            # Attempt cropping only if there's enough data and energy
            if np.sum(intensity_full_spectrum_norm_for_fwhm) >= 1e-12 and len(x_nm_sorted_full) >=2 :
                fwhm_for_cropping, _, _ = calculate_fwhm(x_nm_sorted_full, intensity_full_spectrum_norm_for_fwhm)

                if not np.isnan(fwhm_for_cropping) and fwhm_for_cropping > 1e-9: # If FWHM is valid
                    peak_intensity_idx = np.argmax(intensity_full_spectrum_norm_for_fwhm)
                    lambda_at_peak = x_nm_sorted_full[peak_intensity_idx]
                    window_half_width = (cropping_fwhm_multiplier / 2.0) * fwhm_for_cropping
                    crop_lambda_min = max(0, lambda_at_peak - window_half_width) # Ensure non-negative
                    crop_lambda_max = lambda_at_peak + window_half_width
                    cropping_details_msg = f"Attempted crop to {cropping_fwhm_multiplier:.1f}xFWHM"

                    # Check if crop window is valid and wide enough
                    if crop_lambda_min < crop_lambda_max and (crop_lambda_max - crop_lambda_min) > 1e-9 * max(1.0, lambda_at_peak):
                        crop_mask = (x_nm_sorted_full >= crop_lambda_min) & (x_nm_sorted_full <= crop_lambda_max)

                        if np.sum(crop_mask) >= 2: # Need at least 2 points in cropped data
                            x_nm_cropped_candidate = x_nm_sorted_full[crop_mask]
                            y_amp_cropped_candidate = y_amp_full_normalized[crop_mask]
                            intensity_cropped_candidate = y_amp_cropped_candidate**2
                            # Check if cropped data has enough energy and variance
                            if np.sum(intensity_cropped_candidate) > 1e-12 and len(np.unique(x_nm_cropped_candidate)) > 1:
                                x_nm_for_calculation = x_nm_cropped_candidate
                                y_amp_for_calculation = y_amp_cropped_candidate
                                spectrum_source_msg += f" (Cropped to {cropping_fwhm_multiplier:.1f}xFWHM)"
                                app.logger.info(f"Imported spectrum cropped: {len(x_nm_for_calculation)} points, peak {lambda_at_peak:.2f}nm, range [{crop_lambda_min:.2f}-{crop_lambda_max:.2f}]nm using {cropping_fwhm_multiplier:.1f}xFWHM multiplier.")
                                cropping_details_msg = f"Cropped to {cropping_fwhm_multiplier:.1f}xFWHM ({len(x_nm_for_calculation)} pts)"
                            else:
                                app.logger.warning("Cropped spectrum candidate has insufficient energy/variance. Using full spectrum.")
                                cropping_details_msg += ", but used full (low energy/variance in crop)"
                        else:
                            app.logger.warning(f"Cropping (peak {lambda_at_peak:.2f}nm, {cropping_fwhm_multiplier:.1f}xFWHM) gave <2 points. Using full spectrum.")
                            cropping_details_msg += ", but used full (<2 pts in crop)"
                    else:
                        app.logger.warning(f"Crop window [{crop_lambda_min:.2f}-{crop_lambda_max:.2f}] invalid. Using full spectrum.")
                        cropping_details_msg += ", but used full (invalid window)"
                else:
                    app.logger.warning("FWHM for cropping not determined or too small. Using full spectrum.")
                    cropping_details_msg = "Used full spectrum (FWHM for cropping not found/small)"
            else:
                app.logger.warning("Full spectrum has insufficient energy or points for cropping. Using full spectrum.")
                cropping_details_msg = "Used full spectrum (low energy/points in full)"
            # --- End Cropping Logic ---

            intensity_final_for_calc = y_amp_for_calculation**2
            if np.sum(intensity_final_for_calc) < 1e-12: # Final check for energy
                 raise ValueError("Sum of intensities from (potentially cropped) file data is too low. Check Y data or units.")

            # Calculate effective lambda0 and bandwidth from the (potentially cropped) data
            lambda0_eff_nm = np.sum(x_nm_for_calculation * intensity_final_for_calc) / np.sum(intensity_final_for_calc)
            if not (0 < lambda0_eff_nm < 1e7): # Sanity check for lambda0
                raise ValueError(f"Calculated center wavelength ({lambda0_eff_nm:.2f} nm) from data is out of reasonable range.")

            fwhm_val, _, _ = calculate_fwhm(x_nm_for_calculation, intensity_final_for_calc)
            if np.isnan(fwhm_val) or fwhm_val <= 0: # If FWHM calculation fails
                if len(x_nm_for_calculation) > 1: # Try std-dev based bandwidth
                    std_dev_nm = np.sqrt(np.sum(intensity_final_for_calc * (x_nm_for_calculation - lambda0_eff_nm)**2) / np.sum(intensity_final_for_calc))
                    bandwidth_eff_nm = 2.355 * std_dev_nm # Approx FWHM for Gaussian
                    if bandwidth_eff_nm <= 0:
                         raise ValueError("Could not determine a positive bandwidth from the imported spectrum.")
                    app.logger.warning(f"FWHM calculation failed for spectrum. Using std-dev based bandwidth: {bandwidth_eff_nm:.2f} nm")
                else:
                    raise ValueError("Could not determine bandwidth from spectrum (FWHM failed, not enough points for std-dev).")
            else:
                bandwidth_eff_nm = fwhm_val

            if not (0 < bandwidth_eff_nm < lambda0_eff_nm * 10): # Sanity check for bandwidth
                app.logger.warning(f"Calculated bandwidth ({bandwidth_eff_nm:.2f} nm) from data seems unusual relative to center wavelength ({lambda0_eff_nm:.2f} nm). Proceeding.")

            # Convert wavelength (nm) to angular frequency (rad/fs) for calculations
            omega_custom_abs = 2.0 * np.pi * C_UM_FS / (x_nm_for_calculation / 1000.0) # x_nm to um
            # Sort by angular frequency (required by calculation logic)
            sort_omega_indices = np.argsort(omega_custom_abs)
            omega_custom_abs = omega_custom_abs[sort_omega_indices]
            E_omega_amp_custom = y_amp_for_calculation[sort_omega_indices]

            # Store imported data in params for potential use or logging
            params["imported_lambda_nm"] = x_nm_for_calculation.tolist()
            params["imported_amplitude"] = y_amp_for_calculation.tolist()

        elif spectrum_type == 'gaussian' or spectrum_type == 'sech2':
            # Use parameters from the form for analytical spectra
            lambda0_eff_nm = form_data.get('lambda0_nm', DEFAULT_PARAMS['lambda0_nm'], type=float)
            bandwidth_eff_nm = form_data.get('bandwidth_nm', DEFAULT_PARAMS['bandwidth_nm'], type=float)
            if lambda0_eff_nm <= 0 or bandwidth_eff_nm < 0: # Bandwidth can be 0 for monochromatic
                raise ValueError("Center wavelength must be positive. Bandwidth cannot be negative.")
            if spectrum_type == 'gaussian':
                spectrum_source_msg = "Gaussian Spectrum"
                analytical_spectrum_shape = "gaussian"
            else: # sech2
                spectrum_source_msg = "Sech² Spectrum"
                analytical_spectrum_shape = "sech2"
        else:
            raise ValueError(f"Unknown spectrum type: {spectrum_type}")

        if lambda0_eff_nm is None or bandwidth_eff_nm is None: # Should be set by now
             raise ValueError("Effective lambda0 or bandwidth could not be determined.")

        # Update params with the effective lambda0 and bandwidth to be used in calculations
        params["lambda0_nm"] = lambda0_eff_nm
        params["bandwidth_nm"] = bandwidth_eff_nm

        # --- Core Calculation ---
        results = calculate_pulse_properties(
            **params, # Pass all collected parameters
            spectrum_shape=analytical_spectrum_shape,
            omega_custom_abs=omega_custom_abs,
            E_omega_amp_custom=E_omega_amp_custom
        )

        t_data = results.get('t')
        if t_data is None or len(t_data) == 0: # Check if calculation produced time-domain data
            raise ValueError("Calculation failed to produce time-domain data.")

        # --- Generate Plots ---
        lambda_plot_nm_axis = results.get('lambda_plot_nm_axis', [])
        spectrum_intensity_plot = results.get('spectrum_intensity_plot', [])
        phase_vs_lambda_plot = results.get('phase_vs_lambda_plot', [])

        img_spec = create_matplotlib_plot(plot_spectral,
                                          lambda_plot_nm_axis,
                                          spectrum_intensity_plot,
                                          phase_vs_lambda_plot,
                                          lambda0_eff_nm, bandwidth_eff_nm,
                                          spectrum_source_msg)

        img_time = create_matplotlib_plot(plot_time_domain,
                                          results.get('t'),
                                          results.get('E_t_real'),
                                          results.get('Intensity_t_normalized'),
                                          results.get('phase_t'),
                                          results.get('inst_freq'),
                                          results) # Pass full results dict for plot annotations

        img_ac = create_matplotlib_plot(plot_autocorrelation,
                                        results.get('t'), # Autocorrelation uses the same time axis for delay
                                        results.get('Autocorr_t_normalized'),
                                        results) # Pass full results dict

        # --- Prepare JSON Response ---
        return_data = {
            "success": True,
            "plot_spec_img": img_spec,
            "plot_time_img": img_time,
            "plot_ac_img": img_ac,
            "fwhm_fs": results.get("fwhm_fs"),
            "fwhm_ac": results.get("fwhm_ac"),
            "lambda0_nm": lambda0_eff_nm, # Effective lambda0 used
            "bandwidth_nm": bandwidth_eff_nm, # Effective bandwidth used
            "N": results.get("N"), # Actual N used in calculation
            "spectrum_source": spectrum_source_msg,
            "imported_file_name": imported_filename_display,
            "spectrum_shape_used": results.get("spectrum_shape_used") # e.g., "gaussian", "sech2", "custom"
        }
        if cropping_details_msg: # Add cropping info if applicable
            return_data["cropping_details"] = cropping_details_msg

        return jsonify(return_data)

    except ValueError as e: # Handle known errors (e.g., bad input)
        app.logger.error(f"ValueError: {e}")
        return jsonify({"success": False, "error": str(e)}), 400 # Bad Request
    except MemoryError: # Handle out-of-memory errors (e.g., N too large)
        app.logger.error("MemoryError during calculation or plotting.")
        return jsonify({"success": False, "error": "Server ran out of memory. Try reducing N (exponent) or file size."}), 500 # Internal Server Error
    except Exception as e: # Catch-all for unexpected errors
        app.logger.error(f"Unexpected error: {e}", exc_info=True) # Log with traceback
        return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500 # Internal Server Error


# --- Main Execution Guard ---
if __name__ == '__main__':
    # This block is executed ONLY when the script is run directly (e.g., `python app.py`)
    # It's useful for very simple local testing without the run_local.py dispatcher.
    # However, for the /PulseDispersionAnalyzer prefix to work locally, you should use `run_local.py`.
    # Gunicorn or other WSGI servers will import the 'app' object directly and will not run this block.
    app.run(debug=True, host='0.0.0.0', port=5000)
