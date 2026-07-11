"""
Script : simulate with Nref = 1e6 per C-arm (worker per carm),
convert measured kerma -> N_meas, compute scale = N_meas / Nref,
apply scale to simulated doses and compute H*(10) equivalent to Hp(10) at this energy.

Usage: run this script normally. It will spawn two worker processes.
"""

import sys
import json
import subprocess
import pathlib
import numpy as np

import opengate as gate
from scipy.spatial.transform import Rotation

from opengate.utility import LazyModuleLoader

# ------------------------
# Units (OpenGATE 10.0.3)
# ------------------------
units = gate.g4_units
cm = units.cm
mm = units.mm
m  = units.m
keV = units.keV
deg = units.deg

# -------------------------------------------------
# USER PARAMETERS (modifiable)
# -------------------------------------------------

# Measured data for C-arm 1: VERTICAL
kerma_C1_mGy = 1            # Measured Kerma (mGy)
kV_C1 = 75                    # Tube voltage (kV)
field_cm_at_panel_C1 = 40.0    # Field size on panel (cm)
kerma_distance_C1_cm = 61.5   # Kerma measurement distance
filter_al_mm_C1 = 3.5
filter_cu_mm_C1 = 0.9

# Measured data for C-arm 2: HORIZONTAL
kerma_C2_mGy = 1
kV_C2 = 75
field_cm_at_panel_C2 = 40.0
kerma_distance_C2_cm = 75.0
filter_al_mm_C2 = 3.5
filter_cu_mm_C2 = 0.9   

With_CathPax = False  # Set to True to include CathPax in the simulation 

# Number of simulated photons (reference for each worker) and low number if the simulation is visualized
Visualisation = True  # Set to True to visualize the simulation (reduces Nref)
if Visualisation :
    Nref = 100
else:
    Nref = int(1e6)

# Output directory
current_path = pathlib.Path(__file__).parent.resolve()
output_dir = current_path / "output_scaled"
output_dir.mkdir(parents=True, exist_ok=True)

THIS_SCRIPT = str(pathlib.Path(__file__).resolve())

# -------------------------------------------------
# Colors for visualization in OpenGATE
# (RGBA between 0 and 1)
# -------------------------------------------------
colors = {
	"cyan":   [0, 1, 1, 1],
	"grey":   [.7, .7, .7, 1],
	"yellow": [1, 1, 0, 1],
	"green":  [0, 1, 0, 1],
    "black":  [0.0, 0.0, 0.0, 1],
    "orange": [1.0, 0.5, 0.0, 1]
}

# -------------------------------------------------
# Function: conversion Kerma → number of photons
# -------------------------------------------------
def kerma_to_photons(kerma_mGy, kV, field_size_cm_at_panel, kerma_distance_cm, source_panel_distance_cm,
                    filter_al_mm=3.5, filter_cu_mm=0.9):
    """
    Converts the measured kerma (mGy) to total number of emitted photons.
    The kerma is measured at kerma_distance_cm, the field is defined on the panel at source_panel_distance_cm.
    """

    # (1) Convert kerma to Gy
    K = kerma_mGy * 1e-3

    # (2) True spectrum mean energy (keV) for the given kV and filtration
    sp = LazyModuleLoader("spekpy")
    spectrum = sp.Spek(kV, th=10, physics="kqp")
    spectrum.filter("Al", filter_al_mm).filter("Cu", filter_cu_mm)

    # (3) Kerma per photon (Gy/photon) using SpekPy
    kerma_per_photon = spectrum.get_kerma() / spectrum.get_flu()
    if kerma_per_photon <= 0 or not np.isfinite(kerma_per_photon):
        raise ValueError("SpekPy returned an invalid kerma-per-photon value.")


    # (4) Fluence at measurement distance
    fluence_mesure = K / kerma_per_photon

    # (5) Fluence at panel level (inverse square law)
    fluence_panel = fluence_mesure * (kerma_distance_cm / source_panel_distance_cm) ** 2

    # (6) Area of field on panel
    area_panel_cm2 = field_size_cm_at_panel ** 2

    # (7) Total number of photons = fluence at panel × panel area
    N = fluence_panel * area_panel_cm2
    return N

# -------------------------------------------------
# Utility function to launch a worker subprocess
# -------------------------------------------------
def run_worker(which, params, out_npz):
    # Construction of the python command to launch the worker
    cmd = [sys.executable, THIS_SCRIPT, "worker", which, json.dumps(params), str(out_npz)]
    print("Launching worker:", " ".join(cmd))

    # Execution of the worker
    res = subprocess.run(cmd, check=False)

    # Check if execution failed
    if res.returncode != 0:
        raise RuntimeError(f"Worker {which} failed with returncode {res.returncode}")

# -------------------------------------------------
# MAIN PART: MASTER PROCESS
# -------------------------------------------------
if __name__ == "__main__" and len(sys.argv) == 1:

    # Workers only need kV and Nref
    params1 = {"kV": int(kV_C1), "Nref": int(Nref)}
    params2 = {"kV": int(kV_C2), "Nref": int(Nref)}

    # Output NPZ files
    out1 = output_dir / "sim_carm1.npz"
    out2 = output_dir / "sim_carm2.npz"

    # Launch worker 1
    run_worker("1", params1, out1)

    # Launch worker 2
    run_worker("2", params2, out2)

    # Loading simulated results
    r1 = np.load(out1)
    dose1 = r1["dose_sphere"]         # dose (Gy)
    dose1_unc = r1["dose_unc_sphere"] # relative uncertainty

    r2 = np.load(out2)
    dose2 = r2["dose_sphere"]
    dose2_unc = r2["dose_unc_sphere"]

    # Calculation of real distances between emission center and panel center
    # Carm Vertical (Carm1)
    translation1 = np.array([0, -120*cm, 0*cm])
    rotation1 = Rotation.from_euler("ZYX", [0, 180, 180], degrees=True).as_matrix()
    tube_offset1 = np.array([0.0, -40.0*cm, -40.0*cm])
    panel_offset1 = np.array([0.0, -40.0*cm, 67.0*cm])
    tube_pos1 = translation1 + rotation1 @ tube_offset1
    panel_pos1 = translation1 + rotation1 @ panel_offset1
    distance_C1_cm = np.linalg.norm(panel_pos1 - tube_pos1) / cm

    # Carm Horizontal (Carm2)
    translation2 = np.array([0, -80*cm, 75*cm])
    rotation2 = Rotation.from_euler("ZYX", [180, 0, 0], degrees=True).as_matrix()
    tube_offset2 = np.array([50.0*cm, 0.0, -60.0*cm])
    panel_offset2 = np.array([-59.0*cm, 0.0, -58.0*cm])
    tube_pos2 = translation2 + rotation2 @ tube_offset2
    panel_pos2 = translation2 + rotation2 @ panel_offset2
    distance_C2_cm = np.linalg.norm(panel_pos2 - tube_pos2) / cm

    # Conversion Measured Kerma → equivalent number of photons
    N1_meas = kerma_to_photons(kerma_C1_mGy, kV_C1, field_cm_at_panel_C1, kerma_distance_C1_cm, distance_C1_cm, filter_al_mm_C1, filter_cu_mm_C1)
    N2_meas = kerma_to_photons(kerma_C2_mGy, kV_C2, field_cm_at_panel_C2, kerma_distance_C2_cm, distance_C2_cm, filter_al_mm_C2, filter_cu_mm_C2)

    print(f"N_meas C1 = {N1_meas:.3e} photons ; Nref = {Nref} ; distance = {distance_C1_cm:.1f} cm")
    print(f"N_meas C2 = {N2_meas:.3e} photons ; Nref = {Nref} ; distance = {distance_C2_cm:.1f} cm")

    # Scaling factors compared to simulation (fixed at Nref photons)
    scale1 = N1_meas / float(Nref)
    scale2 = N2_meas / float(Nref)

    print(f"scale1 = {scale1:.3e}, scale2 = {scale2:.3e}")

    # Scaling of simulated doses
    dose1_scaled = dose1 * scale1
    dose2_scaled = dose2 * scale2

    # Sum of contributions
    dose_total = dose1_scaled + dose2_scaled

    # Calculation of Hp(10) = mean dose in ICRU sphere (w_R = 1 → Sv = Gy)
    mean_dose_Gy = float(np.mean(dose_total))
    Hp10_mSv = mean_dose_Gy * 1000.0

    # Uncertainty propagation
    sigma1_abs = dose1_scaled * dose1_unc
    sigma2_abs = dose2_scaled * dose2_unc

    sigma_total_voxel = np.sqrt(sigma1_abs**2 + sigma2_abs**2)
    Nvox = sigma_total_voxel.size

    # uncertainty on Hp(10)
    Hp10_abs_unc_Gy = np.sqrt(np.sum(sigma_total_voxel**2)) / Nvox
    Hp10_unc_mSv = Hp10_abs_unc_Gy * 1000.0

    # relative uncertainty
    Hp10_rel_percent = (Hp10_abs_unc_Gy / mean_dose_Gy) * 100.0 if mean_dose_Gy > 0 else np.nan

    # Display of final results
    print("=== Final results ===")
    print(f"Mean dose in sphere (Gy): {mean_dose_Gy:.6e}")
    print(f"Hp(10) (mSv): {Hp10_mSv:.6f}")
    print(f"Relative uncertainty: {Hp10_rel_percent:.2f} %")
    print(f"Absolute uncertainty (mSv): {Hp10_unc_mSv:.6f}")

    # Saving
    np.savez(output_dir / "final_results.npz",
             dose_total=dose_total,
             hp10_mSv=Hp10_mSv,
             hp10_unc_mSv=Hp10_unc_mSv,
             scale1=scale1, scale2=scale2)

    print("Outputs saved to:", output_dir)

# -------------------------------------------------
# WORKER: launched in subprocess to avoid the problem
#          "only one GATE instance per process".
# -------------------------------------------------
if len(sys.argv) >= 2 and sys.argv[1] == "worker":

    # Retrieval of worker arguments
    which = sys.argv[2]
    params = json.loads(sys.argv[3])
    out_npz = pathlib.Path(sys.argv[4])

    sp = LazyModuleLoader("spekpy")

    def rotation_matrix_from_euler_xyz_deg(angles_deg):
        """
        Return 3x3 numpy matrix for Euler angles in degrees (XYZ convention).
        """
        rot = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()
        return rot

    def add_source_to_tube(sim, tube, name, kvp=100, filter_al_mm=3.5, filter_cu_mm=0.9):
        """
        Add a GenericSource attached to the given tube volume.
        """

        # Create source
        src = sim.add_source("GenericSource", f"{name}_src")
        src.particle = "gamma"
        # Attach source to sourcebox
        src.attached_to = tube.name

        # Direction spectrum       
        src.direction_relative_to_attached_volume = True
        src.direction.type = "histogram"
        # TODO: Need real values for the anode heel effect
        # data = np.load("anodeheeleffect.npz")
        # src.direction.histogram_phi_weights = data["weight"]
        # src.direction.histogram_phi_angles = data["angle"]
        src.direction.histogram_phi_weights = [0.05, 0.8, 0.05]
        src.direction.histogram_phi_angles = [0 * deg, 80 * deg, 100 * deg, 180 * deg]
        src.direction.histogram_theta_weights = [0.05, 0.8, 0.05]
        src.direction.histogram_theta_angles = [0 * deg, 80 * deg, 100 * deg, 180 * deg]
        # src.direction.acceptance_angle.volumes = [sourcebox.name]
        # src.direction.acceptance_angle.normal_flag = False
        # src.direction.acceptance_angle.normal_vector = [0, 0, 1]  # local z direction
        # src.direction.acceptance_angle.normal_tolerance = 5 * deg
        # src.direction.acceptance_angle.skip_policy = "SkipEvents"

        
        # # Additional X-ray filtration
        # try:
        #     src.energy.filter = {
        #         "G4_Al": filter_al_mm * mm,
        #         "G4_Cu": filter_cu_mm * mm,
        #     }
        # except Exception:
        #     pass

        # Energy
        # Energy spectrum with filtration   
        spectrum = sp.Spek(kvp, th=10, physics="kqp")
        spectrum.filter("Al", filter_al_mm).filter("Cu", filter_cu_mm)
        energy_bins, weights = spectrum.get_spectrum()
        src.energy.type = "histogram"
        src.energy.histogram_energy = energy_bins * keV
        src.energy.histogram_weight = weights

        # Number of primaries to simulate
        src.n = Nref  # low if visu is True else high

        return src

    def build_Carm_vertical(sim, name, translation, rotation, kvp=100, filter_al_mm=3.5, filter_cu_mm=0.9):
        """
        Build an Carm-like C-arm with tube, detector panel, and source.
        """

        arc = sim.add_volume("Tesselated", name=f"{name}_arc")
        arc.material = "G4_AIR"
        arc.mother = "world"
        arc.file_name = "Carm_VERTICAL.stl" # Create with Blender3D by H.Necib
        arc.origin_at_cog = True
        arc.translation = translation
        arc.rotation = rotation
        
        # --------- Tube volume ----------
        tube = sim.add_volume("TubsVolume", f"{name}_tube")
        tube.material = "G4_AIR"
        tube.rmin = 0.0 * mm
        tube.rmax = 5*cm
        tube.dz = 5.5*cm
        tube.translation = translation + rotation @ np.array([0.0, -40.0*cm, -40.0*cm])
        tube.rotation = rotation @ rotation_matrix_from_euler_xyz_deg([-90.0, 0.0, 0.0])

        # --------- Detector panel ----------
        panel = sim.add_volume("BoxVolume", f"{name}_panel")
        panel.size = [40*cm, 40*cm, 5*cm]
        panel.translation = translation + rotation @ np.array([0.0, -40.0*cm, 67.0*cm])
        # panel.rotation = rotation_matrix_from_euler_xyz_deg([0.0, 0.0, 0.0])
        panel.material = "G4_CESIUM_IODIDE"
        panel.color = [0.0, 0.0, 1.0, 0.5]

        # --------- Source attached to the tube ----------   
        src = add_source_to_tube(sim, tube, name, kvp, filter_al_mm, filter_cu_mm)

        return {"arc": arc, "tube": tube, "src": src}

    def build_Carm_horizontal(sim, name, translation, rotation, kvp=100, filter_al_mm=3.5, filter_cu_mm=0.9):
        """
        Build an Carm-like C-arm with tube, detector panel, and source.
        """

        arc = sim.add_volume("Tesselated", name=f"{name}_arc")
        arc.material = "G4_AIR"
        arc.mother = "world"
        arc.file_name = "Carm_HORIZONTAL.stl" # Create with Blender3D by H.Necib
        arc.origin_at_cog = True
        arc.translation = translation
        arc.rotation = rotation
        
        # --------- Tube volume ----------
        tube = sim.add_volume("TubsVolume", f"{name}_tube")
        tube.material = "G4_AIR"
        tube.rmin = 0.0 * mm
        tube.rmax = 5*cm
        tube.dz = 5.5*cm
        tube.translation = translation + rotation @ np.array([50.0*cm, 0.0, -60.0*cm])
        tube.rotation = rotation @ rotation_matrix_from_euler_xyz_deg([0.0, 0.0, -90.0])

        # --------- Detector panel ----------
        panel = sim.add_volume("BoxVolume", f"{name}_panel")
        panel.size = [5*cm, 40*cm, 40*cm]
        panel.translation = translation + rotation @ np.array([-59.0*cm, 0.0, -58.0*cm])
        # panel.rotation = rotation_matrix_from_euler_xyz_deg([0.0, 0.0, 0.0])
        panel.material = "G4_CESIUM_IODIDE"
        panel.color = [1.0, 0.5, 0.0, 0.5]

        # --------- Source attached to the tube ----------
        src = add_source_to_tube(sim, tube, name, kvp, filter_al_mm, filter_cu_mm)

        return {"arc": arc, "tube": tube, "src": src}

    # Construction of the simulation GATE
    sim = gate.Simulation()
    sim.g4_verbose = False
    sim.visu = Visualisation
    sim.number_of_threads = 1
    sim.random_seed = "auto"
    sim.check_volumes_overlap = False

    # Folder for the worker
    worker_outdir = out_npz.parent / f"worker_{which}"
    worker_outdir.mkdir(parents=True, exist_ok=True)
    sim.output_dir = worker_outdir

    # Physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    # sim.physics_manager.set_production_cut("world", "gamma", 5*m)

    # World
    world = sim.world
    world.size = [5*m, 5*m, 5*m]
    world.material = "G4_AIR"

    # Creation of the gantry according to the worker
    if which == "1":
        kV = int(params.get("kV", 80))
        translation = np.array([0, -120*cm, 0*cm])
        rotation = Rotation.from_euler("ZYX", [0, 180, 180], degrees=True).as_matrix()
        carm = build_Carm_vertical(sim, "Carm_vertical", translation, rotation, kV_C1, filter_al_mm_C1, filter_cu_mm_C1)
        carm['src'].n = int(params.get("Nref", Nref))

    else:
        kV = int(params.get("kV", 70))
        translation = np.array([0, -80*cm, 75*cm])
        rotation = Rotation.from_euler("ZYX", [180, 0, 0], degrees=True).as_matrix()
        carm = build_Carm_horizontal(sim, "Carm_horizontal", translation, rotation, kV_C2, filter_al_mm_C2, filter_cu_mm_C2)
        carm['src'].n = int(params.get("Nref", Nref))

    # [X, Y, Z] = PATIENT [LEFT-RIGHT ; HEAD-FEET ; ANTERO-POST]
    #  carbon table
    table = sim.add_volume("Box", "graphite_table")
    table.size = [60 * cm, 2 * m, 1 * cm]
    table.material = "G4_GRAPHITE"
    table.translation = [0 * m, 0 * cm, 0 * cm]
    table.color =  colors["grey"]

    # patient Head_Body
    Head_Body = sim.add_volume("Tesselated", name="Head_Body")
    Head_Body.material = "G4_PLEXIGLASS"
    Head_Body.mother = "world"
    Head_Body.file_name = "Head_Body.stl"
    Head_Body.origin_at_cog = True
    Head_Body.translation = [0 * m, -75* cm, 12.5 * cm] # from the reference where table is at [0, 0, 0]
    Head_Body.rotation = rotation_matrix_from_euler_xyz_deg([-90.0, 0.0, 180.0])
    Head_Body.color = colors["orange"]

    # patient Head_Skeleton
    Head_Skeleton = sim.add_volume("Tesselated", name="Head_Skeleton")
    Head_Skeleton.material = "G4_BONE_COMPACT_ICRU"
    Head_Skeleton.mother = "Head_Body"
    Head_Skeleton.file_name = "Head_Skeleton.stl"
    Head_Skeleton.origin_at_cog = True
    Head_Skeleton.translation = [0 * m, 0* cm, 0 * cm] # from the reference of the Head_Body
    Head_Skeleton.rotation = rotation_matrix_from_euler_xyz_deg([0.0, 0.0, 0.0])
    Head_Skeleton.color = colors["grey"]

    # Patient Thorax
    thorax = sim.add_volume("BoxVolume", "thorax")
    thorax.size = [35*cm, 35*cm, 20*cm]
    thorax.material = "G4_PLEXIGLASS"
    thorax.translation = [0, -48*cm, 12.5*cm]
    thorax.color = colors["orange"]

    # Patient Abdomen 
    abdo = sim.add_volume("BoxVolume", "abdo")
    abdo.size = [35*cm, 35*cm, 20*cm]
    abdo.material = "G4_PLEXIGLASS"
    abdo.translation = [0, -13*cm, 12.5*cm]
    abdo.color = colors["orange"]

    # ICRU sphere for dose calculation
    sphere = sim.add_volume("SphereVolume", "icru_sphere")
    sphere.rmax = 15 * cm
    sphere.translation = [45.0*cm, 0.0*cm, 30*cm] # from the reference where table is at [0, 0, 0]
    sphere.material = "G4_WATER"
    sphere.color = colors["cyan"]

    if With_CathPax:
        # CathPax between ICRU sphere and the C-arm and not inside the table
        cathPax1 = sim.add_volume("BoxVolume", "cathPax1") # Large door
        cathPax1.size = [80*cm, 2*mm, 2.0*m] # 2 mm Pb
        cathPax1.translation = [80.0*cm, -30.0*cm, 0.0*cm] 
        cathPax1.material = "G4_Pb"
        cathPax1.color = colors["grey"]

        cathPax2 = sim.add_volume("BoxVolume", "cathPax2") # Small door above the patient
        cathPax2.size = [80*cm, 2*mm, 60*cm] # 2 mm Pb
        cathPax2.translation = [0.0*cm, -30.0*cm, 70.0*cm] 
        cathPax2.material = "G4_Pb"
        cathPax2.color = colors["grey"]

        cathPax3 = sim.add_volume("BoxVolume", "cathPax3") # Tab on small door 2 above the patient
        cathPax3.size = [80*cm, 1*mm, 15*cm] # 1 mm Pb
        cathPax3.translation = [0.0*cm, -30.0*cm, 32.5*cm] 
        cathPax3.material = "G4_Pb"
        cathPax3.color = colors["grey"]
        
        cathPax4 = sim.add_volume("BoxVolume", "cathPax4") # Small door below the table
        cathPax4.size = [2*mm, 80*cm, 1*m] # 1 mm Pb !! See cathPax diagram
        cathPax4.translation = [40.0*cm, 11.0*cm, -50.0*cm] 
        cathPax4.material = "G4_Pb"
        cathPax4.color = colors["grey"]

        cathPax5 = sim.add_volume("BoxVolume", "cathPax5") # Apron above door 4
        cathPax5.size = [2*mm, 80*cm, 30*cm] # 1 mm Pb !! See cathPax diagram
        cathPax5.translation = [30.0*cm, 11.0*cm, 11*cm] 
        cathPax5.rotation = Rotation.from_euler("ZYX", [0, -45, 0], degrees=True).as_matrix()
        cathPax5.material = "G4_Pb"
        cathPax5.color = colors["grey"]


        cathPax6 = sim.add_volume("BoxVolume", "cathPax6") # Corner tab on small door 2 above the patient
        cathPax6.size = [25*cm, 1*mm, 25*cm] # 1 mm Pb !! See cathPax diagram
        cathPax6.translation = [27.5*cm, -30.0*cm, 12.5*cm] 
        cathPax6.material = "G4_Pb"
        cathPax6.color = colors["grey"]

    # DoseActor attached to the ICRU sphere
    dose = sim.add_actor("DoseActor", "SphereDose")
    dose.attached_to = "icru_sphere"
    dose.edep.active = True
    dose.dose.active = True
    dose.dose_uncertainty.active = True
    dose.spacing = [10*mm, 10*mm, 10*mm]
    dose.hit_type = "random"
    dose.edep.output_filename = "icru_edep.mhd"
    dose.dose.output_filename = "icru_dose.mhd"
    dose.dose_uncertainty.output_filename = "icru_dose_unc.mhd"

    # Statistics actor
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.track_types_flag = True
    stats.output_filename = "stats.txt"

    # Launch the simulation
    sim.run()

    # Read DoseActor results
    da = sim.get_actor("SphereDose")
    dose_image = da.dose.image
    dose_unc_image = da.dose_uncertainty.image

    dose_arr = np.asarray(dose_image)
    dose_unc_arr = np.asarray(dose_unc_image)

    # Saving results to NPZ file
    np.savez(out_npz,
             dose_sphere=dose_arr,
             dose_unc_sphere=dose_unc_arr,
             Nref=int(params.get("Nref", Nref)),
             kV=int(params.get("kV", kV)))

    print(f"Worker {which} done. Saved {out_npz}")
    sys.exit(0)
