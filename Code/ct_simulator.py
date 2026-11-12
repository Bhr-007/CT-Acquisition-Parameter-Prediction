# ==============================================================================
# MODULE: ct_simulator.py
# Functional Polychromatic Beer-Lambert + Poisson CT Simulator with ASTRA
# ==============================================================================
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import astra

MATERIAL_PROPS = {
    "Polymer_PMMA": {"code": 0, "rho_0": 1.18, "z_eff": 6.5,  "inc_mat": "Aluminium", "inc_rho": 2.70},
    "Aluminium":    {"code": 1, "rho_0": 2.70, "z_eff": 13.0, "inc_mat": "Polymer_PMMA", "inc_rho": 1.18},
    "Steel":        {"code": 2, "rho_0": 7.87, "z_eff": 26.0, "inc_mat": "Aluminium", "inc_rho": 2.70},
}

GEOMETRY_CODES = {"cylinder": 0, "ellipse": 1, "block": 2}

def load_nist_table(project_root):
    csv_path = Path(project_root) / "Data" / "Reference" / "nist_xcom_attenuation.csv"
    return pd.read_csv(csv_path)

def get_energy_grid(e_min=12.0, e_max=220.0, step_kev=2.0):
    return np.arange(e_min, e_max + 0.1, step_kev, dtype=np.float64)

def evaluate_mu_rho(energies_kev, material_name, nist_df):
    col = f"mu_rho_{material_name}_cm2_g"
    return np.exp(np.interp(np.log(energies_kev), np.log(nist_df["energy_keV"].values), np.log(nist_df[col].values)))

def generate_polychromatic_spectrum(
    energies_kev, kv, mas_per_proj, filter_cu_mm, nist_df,
    intrinsic_al_mm=1.5, flux_constant=260.0
):
    de = float(energies_kev[1] - energies_kev[0]) if len(energies_kev) > 1 else 1.0
    brem = np.maximum(kv - energies_kev, 0.0) / energies_kev
    
    char = np.zeros_like(energies_kev)
    if kv > 69.5:
        char_scale = 0.08 * ((kv - 69.5) ** 1.63)
        for e_line, rel_w in [(58.0, 0.25), (59.3, 0.50), (67.2, 0.15)]:
            char += char_scale * rel_w * np.exp(-0.5 * ((energies_kev - e_line) / (0.8 * de)) ** 2)
            
    raw_spec = (brem + char) * (energies_kev <= kv)
    
    mu_al = evaluate_mu_rho(energies_kev, "Aluminium", nist_df) * 2.70
    mu_cu = evaluate_mu_rho(energies_kev, "Copper", nist_df) * 8.96
    
    trans_filter = np.exp(-mu_al * (intrinsic_al_mm * 0.1) - mu_cu * (filter_cu_mm * 0.1))
    n0_e = flux_constant * mas_per_proj * raw_spec * trans_filter * de
    return np.maximum(n0_e, 0.0)

def create_parametrised_phantom(
    n_grid=128, fov_mm=32.0, material="Aluminium", geometry_type="cylinder",
    sample_size_mm=14.0, density_scale=1.0
):
    pixel_size_mm = fov_mm / float(n_grid)
    pixel_size_cm = pixel_size_mm * 0.1
    coords = (np.arange(n_grid) - (n_grid - 1) / 2.0) * pixel_size_mm
    X, Y = np.meshgrid(coords, coords)
    
    radius_mm = 0.5 * sample_size_mm
    if geometry_type == "cylinder":
        outer_mask = (X**2 + Y**2) <= (radius_mm**2)
    elif geometry_type == "ellipse":
        a_mm = radius_mm * 1.15
        b_mm = radius_mm * 0.85
        outer_mask = ((X / a_mm)**2 + (Y / b_mm)**2) <= 1.0
    elif geometry_type == "block":
        half_w = radius_mm * 0.90
        half_h = radius_mm * 0.85
        outer_mask = (np.abs(X) <= half_w) & (np.abs(Y) <= half_h)
    else:
        raise ValueError(f"Unknown geometry_type: {geometry_type}")
        
    r_inc_mm = max(0.24 * radius_mm, 1.60)
    x_offset_mm = 0.42 * radius_mm
    
    inc_mask = ((X - x_offset_mm)**2 + Y**2) <= (r_inc_mm**2)
    inc_mask = inc_mask & outer_mask
    
    air_void_mask = ((X + x_offset_mm)**2 + (Y - 0.15 * radius_mm)**2) <= ((0.75 * r_inc_mm)**2)
    air_void_mask = air_void_mask & outer_mask & (~inc_mask)
    
    matrix_mask = outer_mask & (~inc_mask) & (~air_void_mask)
    
    dist_inc = distance_transform_edt(inc_mask)
    roi_inc = dist_inc >= 1.5
    if np.sum(roi_inc) < 12:
        roi_inc = inc_mask.copy()
        
    dist_mat = distance_transform_edt(matrix_mask)
    roi_bg = (dist_mat >= 2.0) & (np.abs(Y) >= 0.22 * radius_mm) & (np.abs(X) <= 0.45 * radius_mm)
    if np.sum(roi_bg) < 25:
        roi_bg = dist_mat >= 1.5
    if np.sum(roi_bg) < 15:
        roi_bg = matrix_mask.copy()
        
    R_radial = np.sqrt(X**2 + Y**2)
    roi_center = matrix_mask & (R_radial <= 0.35 * radius_mm)
    if np.sum(roi_center) < 5:
        r_mat = R_radial[matrix_mask]
        roi_center = matrix_mask & (R_radial <= np.percentile(r_mat, 25.0))
    roi_perim = matrix_mask & (R_radial >= 0.65 * radius_mm) & (R_radial <= 0.88 * radius_mm)
    if np.sum(roi_perim) < 5:
        r_mat = R_radial[matrix_mask]
        roi_perim = matrix_mask & (R_radial >= np.percentile(r_mat, 70.0))
    
    return {
        "n_grid": n_grid,
        "fov_mm": fov_mm,
        "pixel_size_mm": pixel_size_mm,
        "pixel_size_cm": pixel_size_cm,
        "material": material,
        "geometry_type": geometry_type,
        "sample_size_mm": sample_size_mm,
        "density_scale": density_scale,
        "outer_mask": outer_mask,
        "matrix_mask": matrix_mask,
        "inc_mask": inc_mask,
        "air_void_mask": air_void_mask,
        "roi_inc": roi_inc,
        "roi_bg": roi_bg,
        "roi_center": roi_center,
        "roi_perim": roi_perim,
        "inc_center_xy_mm": (x_offset_mm, 0.0),
        "inc_radius_mm": r_inc_mm,
    }

def detect_astra_engine():
    if astra.use_cuda():
        try:
            vg = astra.create_vol_geom(32, 32, 1)
            angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            pg = astra.create_proj_geom("cone", 1.0, 1.0, 1, 48, angles, 300.0, 100.0)
            vol_id = astra.data3d.create("-vol", vg, np.ones((1, 32, 32), dtype=np.float32))
            sino_id = astra.data3d.create("-sino", pg, np.zeros((1, 16, 48), dtype=np.float32))
            cfg_fp = astra.astra_dict("FP3D_CUDA")
            cfg_fp["VolumeDataId"] = vol_id
            cfg_fp["ProjectionDataId"] = sino_id
            alg_fp = astra.algorithm.create(cfg_fp)
            astra.algorithm.run(alg_fp)
            astra.algorithm.delete(alg_fp)
            
            cfg_fdk = astra.astra_dict("FDK_CUDA")
            cfg_fdk["ProjectionDataId"] = sino_id
            cfg_fdk["ReconstructionDataId"] = vol_id
            alg_fdk = astra.algorithm.create(cfg_fdk)
            astra.algorithm.run(alg_fdk)
            astra.algorithm.delete(alg_fdk)
            astra.data3d.delete([vol_id, sino_id])
            return "3D_FDK_CUDA"
        except Exception:
            pass
    return "2D_FBP_CPU"

ASTRA_ENGINE = detect_astra_engine()

def astra_forward_project_mask(mask_2d, n_projections, pixel_size_cm, engine=ASTRA_ENGINE):
    n_grid = mask_2d.shape[0]
    n_det = int(np.ceil(n_grid * 1.45))
    if n_det % 2 == 1:
        n_det += 1
        
    if engine == "3D_FDK_CUDA":
        angles = np.linspace(0, 2.0 * np.pi, int(n_projections), endpoint=False)
        vg = astra.create_vol_geom(n_grid, n_grid, 1)
        pg = astra.create_proj_geom("cone", 1.3, 1.3, 1, n_det, angles, 400.0, 120.0)
        vol_data = np.expand_dims(mask_2d.astype(np.float32), axis=0)
        vol_id = astra.data3d.create("-vol", vg, vol_data)
        sino_id = astra.data3d.create("-sino", pg, np.zeros((1, len(angles), n_det), dtype=np.float32))
        cfg = astra.astra_dict("FP3D_CUDA")
        cfg["VolumeDataId"] = vol_id
        cfg["ProjectionDataId"] = sino_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        sino = astra.data3d.get(sino_id)[0].copy()
        astra.algorithm.delete(alg_id)
        astra.data3d.delete([vol_id, sino_id])
        return sino * pixel_size_cm
    else:
        angles = np.linspace(0, np.pi, int(n_projections), endpoint=False)
        vg = astra.create_vol_geom(n_grid, n_grid)
        pg = astra.create_proj_geom("parallel", 1.0, n_det, angles)
        proj_id = astra.create_projector("line", pg, vg)
        sino_id, sino = astra.create_sino(mask_2d.astype(np.float32), proj_id)
        astra.data2d.delete(sino_id)
        astra.projector.delete(proj_id)
        return sino.copy() * pixel_size_cm

def astra_reconstruct_slice(sino_log, n_grid, pixel_size_cm, engine=ASTRA_ENGINE):
    n_projections, n_det = sino_log.shape
    if engine == "3D_FDK_CUDA":
        angles = np.linspace(0, 2.0 * np.pi, int(n_projections), endpoint=False)
        vg = astra.create_vol_geom(n_grid, n_grid, 1)
        pg = astra.create_proj_geom("cone", 1.3, 1.3, 1, n_det, angles, 400.0, 120.0)
        sino_3d = np.expand_dims(sino_log.astype(np.float32), axis=0)
        sino_id = astra.data3d.create("-sino", pg, sino_3d)
        rec_id = astra.data3d.create("-vol", vg, np.zeros((1, n_grid, n_grid), dtype=np.float32))
        cfg = astra.astra_dict("FDK_CUDA")
        cfg["ProjectionDataId"] = sino_id
        cfg["ReconstructionDataId"] = rec_id
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data3d.get(rec_id)[0].copy()
        astra.algorithm.delete(alg_id)
        astra.data3d.delete([sino_id, rec_id])
        return rec / pixel_size_cm
    else:
        angles = np.linspace(0, np.pi, int(n_projections), endpoint=False)
        vg = astra.create_vol_geom(n_grid, n_grid)
        pg = astra.create_proj_geom("parallel", 1.0, n_det, angles)
        proj_id = astra.create_projector("line", pg, vg)
        sino_id = astra.data2d.create("-sino", pg, sino_log.astype(np.float32))
        rec_id = astra.data2d.create("-vol", vg)
        cfg = astra.astra_dict("FBP")
        cfg["ReconstructionDataId"] = rec_id
        cfg["ProjectionDataId"] = sino_id
        cfg["ProjectorId"] = proj_id
        cfg["option"] = {"FilterType": "ram-lak"}
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id).copy()
        astra.algorithm.delete(alg_id)
        astra.data2d.delete([sino_id, rec_id])
        astra.projector.delete(proj_id)
        return rec / pixel_size_cm

def calibrate_reconstruction_scale(engine=ASTRA_ENGINE):
    ph = create_parametrised_phantom(128, 32.0, "Aluminium", "cylinder", 16.0, 1.0)
    mask = ph["outer_mask"].astype(np.float32)
    sino_len = astra_forward_project_mask(mask, 180, ph["pixel_size_cm"], engine=engine)
    true_mu = 1.0
    rec_raw = astra_reconstruct_slice(sino_len * true_mu, 128, ph["pixel_size_cm"], engine=engine)
    measured = float(np.mean(rec_raw[ph["roi_center"]]))
    return 1.0 / measured if measured > 1e-6 else 1.0

RECON_CALIB_FACTOR = calibrate_reconstruction_scale(ASTRA_ENGINE)

def compute_edge_fwhm_mm(recon_img, phantom_dict):
    n_grid = phantom_dict["n_grid"]
    px_mm = phantom_dict["pixel_size_mm"]
    mid_row = n_grid // 2
    profile = recon_img[mid_row, :]
    
    x_coords = (np.arange(n_grid) - (n_grid - 1) / 2.0) * px_mm
    x_edge_expected = phantom_dict["inc_center_xy_mm"][0] - phantom_dict["inc_radius_mm"]
    win_mask = (x_coords >= x_edge_expected - 2.5) & (x_coords <= x_edge_expected + 2.5)
    if np.sum(win_mask) < 5:
        return float(2.0 * px_mm)
    
    sub_x = x_coords[win_mask]
    sub_p = profile[win_mask]
    lsf = np.abs(np.gradient(sub_p, sub_x))
    max_g = np.max(lsf)
    if max_g <= 1e-8:
        return float(4.0 * px_mm)
    above_half = lsf >= (0.5 * max_g)
    fwhm_mm = float(np.sum(above_half) * px_mm)
    weights = np.maximum(lsf - 0.2 * max_g, 0.0)
    if np.sum(weights) > 1e-8:
        mean_x = np.sum(sub_x * weights) / np.sum(weights)
        var_x = np.sum(weights * ((sub_x - mean_x) ** 2)) / np.sum(weights)
        fwhm_cont = 2.355 * np.sqrt(max(var_x, (0.4 * px_mm) ** 2))
        fwhm_mm = float(np.clip(0.5 * fwhm_mm + 0.5 * fwhm_cont, 0.8 * px_mm, 6.0 * px_mm))
    return fwhm_mm

def simulate_single_configuration(
    config_id, material, geometry_type, sample_size_mm, density_scale,
    tube_voltage_kv, mas_per_proj, n_projections, filter_cu_mm,
    nist_df, rng_seed=42, precomputed_lengths=None, return_slice=False
):
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(rng_seed))
    
    ph = create_parametrised_phantom(
        n_grid=128, fov_mm=32.0, material=material,
        geometry_type=geometry_type, sample_size_mm=sample_size_mm,
        density_scale=density_scale
    )
    
    if precomputed_lengths is not None:
        len_mat_cm, len_inc_cm = precomputed_lengths
    else:
        len_mat_cm = astra_forward_project_mask(ph["matrix_mask"], int(n_projections), ph["pixel_size_cm"])
        len_inc_cm = astra_forward_project_mask(ph["inc_mask"], int(n_projections), ph["pixel_size_cm"])
        
    energies_kev = get_energy_grid(12.0, 220.0, 2.0)
    n0_e = generate_polychromatic_spectrum(
        energies_kev, tube_voltage_kv, mas_per_proj, filter_cu_mm, nist_df
    )
    n0_total = float(np.sum(n0_e))
    mean_energy_kev = float(np.sum(energies_kev * n0_e) / max(n0_total, 1e-12))
    
    mat_info = MATERIAL_PROPS[material]
    rho_bulk = mat_info["rho_0"] * float(density_scale)
    mu_mat_e = evaluate_mu_rho(energies_kev, material, nist_df) * rho_bulk
    
    inc_mat_name = mat_info["inc_mat"]
    rho_inc = mat_info["inc_rho"]
    mu_inc_e = evaluate_mu_rho(energies_kev, inc_mat_name, nist_df) * rho_inc
    
    active_idx = n0_e > 1e-6
    e_act = n0_e[active_idx]
    mu_m_act = mu_mat_e[active_idx]
    mu_i_act = mu_inc_e[active_idx]
    
    attenuation_exponent = (
        len_mat_cm[:, :, None] * mu_m_act[None, None, :] +
        len_inc_cm[:, :, None] * mu_i_act[None, None, :]
    )
    expected_photons = np.sum(e_act[None, None, :] * np.exp(-attenuation_exponent), axis=2)
    measured_photons = rng.poisson(np.maximum(expected_photons, 0.0))
    
    sample_rays_mask = (len_mat_cm + len_inc_cm) > 0.05
    if np.any(sample_rays_mask):
        min_expected_photons = float(np.min(expected_photons[sample_rays_mask]))
        mean_transmitted_photons = float(np.mean(expected_photons[sample_rays_mask]))
        starved_pixel_fraction = float(np.mean(measured_photons[sample_rays_mask] <= 2))
    else:
        min_expected_photons = n0_total
        mean_transmitted_photons = n0_total
        starved_pixel_fraction = 0.0
        
    is_photon_starved = bool((starved_pixel_fraction > 0.25) or (min_expected_photons < 2.0))
    
    clamped_photons = np.maximum(measured_photons, 1.0)
    sino_log = -np.log(clamped_photons / max(n0_total, 1.0))
    
    recon_slice = astra_reconstruct_slice(sino_log, ph["n_grid"], ph["pixel_size_cm"]) * RECON_CALIB_FACTOR
    
    bg_vals = recon_slice[ph["roi_bg"]]
    inc_vals = recon_slice[ph["roi_inc"]]
    
    mu_bg_mean = float(np.mean(bg_vals))
    mu_bg_std = float(np.std(bg_vals, ddof=1))
    mu_inc_mean = float(np.mean(inc_vals))
    mu_inc_std = float(np.std(inc_vals, ddof=1))
    
    snr = float(max(mu_bg_mean, 0.0) / max(mu_bg_std, 1e-6))
    pooled_noise = float(np.sqrt(0.5 * (mu_bg_std**2 + mu_inc_std**2)))
    cnr = float( abs(mu_inc_mean - mu_bg_mean) / max(pooled_noise, 1e-6) )
    
    c_mean = float(np.mean(recon_slice[ph["roi_center"]]))
    p_mean = float(np.mean(recon_slice[ph["roi_perim"]]))
    cupping_pct = float(100.0 * (p_mean - c_mean) / max(p_mean, 1e-4))
    
    fwhm_mm = compute_edge_fwhm_mm(recon_slice, ph)
    per_proj_time_s = (float(mas_per_proj) / 5.0) + 0.015
    scan_time_s = float(n_projections) * per_proj_time_s + 2.0
    dose_proxy = float(n_projections) * float(np.sum(energies_kev * n0_e)) * 1e-6
    
    if not np.all(np.isfinite(recon_slice)) or snr <= 0.0:
        sim_status = "NUMERICAL_FAILURE"
    elif is_photon_starved:
        sim_status = "PHOTON_STARVED"
    else:
        sim_status = "VALID"
        
    elapsed_s = time.perf_counter() - t0
    
    record = {
        "config_id": int(config_id),
        "material": material,
        "material_code": int(mat_info["code"]),
        "geometry_type": geometry_type,
        "geometry_code": int(GEOMETRY_CODES[geometry_type]),
        "sample_size_mm": float(sample_size_mm),
        "density_scale": float(density_scale),
        "bulk_density_g_cm3": float(rho_bulk),
        "z_eff": float(mat_info["z_eff"]),
        "tube_voltage_kv": float(tube_voltage_kv),
        "mas_per_proj": float(mas_per_proj),
        "total_mas": float(mas_per_proj * n_projections),
        "n_projections": int(n_projections),
        "filter_cu_mm": float(filter_cu_mm),
        "mean_energy_kev": mean_energy_kev,
        "open_beam_photons": n0_total,
        "min_transmitted_photons": min_expected_photons,
        "mean_transmitted_photons": mean_transmitted_photons,
        "starved_pixel_fraction": starved_pixel_fraction,
        "is_photon_starved": is_photon_starved,
        "recon_algorithm": ASTRA_ENGINE,
        "mu_bg_mean_cm1": mu_bg_mean,
        "mu_bg_std_cm1": mu_bg_std,
        "mu_inc_mean_cm1": mu_inc_mean,
        "snr": snr,
        "cnr": cnr,
        "edge_fwhm_mm": fwhm_mm,
        "cupping_index_pct": cupping_pct,
        "scan_time_s": scan_time_s,
        "dose_proxy": dose_proxy,
        "sim_status": sim_status,
        "sim_time_s": elapsed_s,
        "rng_seed": int(rng_seed),
    }
    if return_slice:
        return record, recon_slice, ph
    return record
