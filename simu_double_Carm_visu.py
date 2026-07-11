"""
OpenGATE 10.0.3 — biplane Carm scene (2 C-arms) with patient on, for visualization only.
This script creates a scene with a 3D model of a biplane C-arm, a room,
table, patient, sphère ICRU, DoseActor + export GDML + PyVista preview.

"""

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import opengate as gate
import opengate_core as g4

from opengate.utility import LazyModuleLoader

sp = LazyModuleLoader("spekpy")

# Optional visualization
try:
    PV_AVAILABLE = True
except Exception:
    PV_AVAILABLE = False

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
    "orange": [1.0, 0.5, 0.0, 1],
    "blue":   [0.0, 0.0, 1.0, 1],
    "red":    [1.0, 0.0, 0.0, 1],
    "white":  [1.0, 1.0, 1.0, 1],
    "purple": [0.5, 0.0, 0.5, 1],
    "pink":   [1.0, 0.75, 0.8, 1],
    "brown":  [0.6, 0.4, 0.2, 1],
    "light_grey": [0.8, 0.8, 0.8, 1],
    "dark_grey": [0.3, 0.3, 0.3, 1],
    "light_blue": [0.68, 0.85, 0.9, 1],
    "dark_blue": [0.0, 0.0, 0.55, 1],
    "light_green": [0.56, 0.93, 0.56, 1],
    "dark_green": [0.0, 0.39, 0.0, 1]
}

# ------------------------
# Units (OpenGATE 10 style)
# ------------------------
units = gate.g4_units
cm = units.cm
mm = units.mm
m  = units.m
keV = units.keV
deg = units.deg

# ------------------------
# User parameters (edit here)
# ------------------------
OUT_DIR = "output"
Path(OUT_DIR).mkdir(exist_ok=True)

VRML_FILE = os.path.join(OUT_DIR, "scene_Carm_biplane.wrl")

# geometry positions (meters via units)
# patient placed on table at z = 12.5 cm
PATIENT_CENTER = [0.0, 0.0, 12.5*cm]

# sphere ICRU: center at x=1.0 m lateral, z=1.7 m height
SPHERE_CENTER = [1.0*m, 0.0*m, 1.7*m]
SPHERE_RADIUS = 15*cm

# Monte Carlo runs (default commented)
N_EVENTS_TEST = int(2e2)   # small quick test
N_EVENTS_FULL = int(5e6)   # realistic run (uncomment when ready)

# ------------------------ 
# Helper: rotation function
# ------------------------
def rotation_matrix_from_euler_xyz_deg(angles_deg):
    """
    Return 3x3 numpy matrix for Euler angles in degrees (XYZ convention).
    """
    rot = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()
    return rot

# ------------------------
# Create simulation & geometry
# ------------------------
def create_scene(sim):
    # world
    sim.world.size = [10*m, 10*m, 4*m]
    sim.world.material = "G4_AIR"
    sim.world.color = [0.1, 0.1, 0.1, 0.1]


    # Ground plane
    ground = sim.add_volume("Tesselated", name="ground")
    ground.material = "G4_CONCRETE"
    ground.mother = "world"
    ground.file_name = "room_HD49.stl" # Create with Blender3D by H.Necib
    ground.origin_at_cog = True
    ground.translation = [0 * m, 1.5 * m, +20 * cm]
    ground.color =  colors["grey"]

    # Protection console wall
    console_wall = sim.add_volume("Tesselated", name="paravent")
    console_wall.material = "G4_Pb"
    console_wall.mother = "world"
    console_wall.file_name = "paravent.stl" # Create with Blender3D by H.Necib
    console_wall.origin_at_cog = True
    console_wall.translation = [-2.2 * m, 0 * m, +20 * cm]
    console_wall.color =  colors["blue"]

    # -----------------------
    # Utility to build one Carm-like arm
    # -----------------------

    # [X, Y, Z] = PATIENT [LEFT-RIGHT ; HEAD-FEET ; ANTERO-POST]
    # Carbon table
    table = sim.add_volume("Box", "graphite_table")
    table.size = [60 * cm, 2 * m, 1 * cm]
    table.material = "G4_GRAPHITE"
    table.translation = [0 * m, 0 * cm, 0 * cm]
    table.color =  colors["grey"]

    # Patient Head_Body
    Head_Body = sim.add_volume("Tesselated", name="Head_Body")
    Head_Body.material = "G4_PLEXIGLASS"
    Head_Body.mother = "world"
    Head_Body.file_name = "Head_Body.stl"
    Head_Body.origin_at_cog = True
    Head_Body.translation = [0 * m, -75* cm, 12.5 * cm] # from the reference where table is at [0, 0, 0]
    Head_Body.rotation = rotation_matrix_from_euler_xyz_deg([-90.0, 0.0, 180.0])
    Head_Body.color = colors["orange"]

    # Patient Head_Skeleton
    Head_Skeleton = sim.add_volume("Tesselated", name="Head_Skeleton")
    Head_Skeleton.material = "G4_BONE_COMPACT_ICRU"
    Head_Skeleton.mother = "Head_Body"
    Head_Skeleton.file_name = "Head_Skeleton.stl"
    Head_Skeleton.origin_at_cog = True
    Head_Skeleton.translation = [0 * m, 0* cm, 0 * cm] # from the reference of the Head_Body
    Head_Skeleton.rotation = rotation_matrix_from_euler_xyz_deg([0.0, 0.0, 0.0])
    Head_Skeleton.color = colors["grey"]

    # Patient
    thorax = sim.add_volume("BoxVolume", "thorax")
    thorax.size = [35*cm, 35*cm, 20*cm]
    thorax.material = "G4_PLEXIGLASS"
    thorax.translation = [0, -48*cm, 12.5*cm]
    thorax.color = colors["orange"]

    # Patient
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


    # CathPax between ICRU sphere and the C-arm and not inside the table
    cathPax1 = sim.add_volume("BoxVolume", "cathPax1") # Large door
    cathPax1.size = [80*cm, 2*mm, 2.0*m] # 2 mm Pb
    cathPax1.translation = [80.0*cm, -30.0*cm, 0.0*cm] 
    cathPax1.material = "G4_Pb"
    cathPax1.color = colors["pink"]

    cathPax2 = sim.add_volume("BoxVolume", "cathPax2") # Small door above the patient
    cathPax2.size = [80*cm, 2*mm, 60*cm] # 2 mm Pb
    cathPax2.translation = [0.0*cm, -30.0*cm, 70.0*cm] 
    cathPax2.material = "G4_Pb"
    cathPax2.color = colors["pink"]

    cathPax3 = sim.add_volume("BoxVolume", "cathPax3") # Tab on small door 2 above the patient
    cathPax3.size = [80*cm, 1*mm, 15*cm] # 1 mm Pb
    cathPax3.translation = [0.0*cm, -30.0*cm, 32.5*cm] 
    cathPax3.material = "G4_Pb"
    cathPax3.color = colors["pink"]
    
    cathPax4 = sim.add_volume("BoxVolume", "cathPax4") # Small door below the table
    cathPax4.size = [2*mm, 80*cm, 1*m] # 1 mm Pb !! See cathPax diagram
    cathPax4.translation = [40.0*cm, 11.0*cm, -50.0*cm] 
    cathPax4.material = "G4_Pb"
    cathPax4.color = colors["pink"]

    cathPax5 = sim.add_volume("BoxVolume", "cathPax5") # Apron above door 4
    cathPax5.size = [2*mm, 80*cm, 30*cm] # 1 mm Pb !! See cathPax diagram
    cathPax5.translation = [30.0*cm, 11.0*cm, 11*cm] 
    cathPax5.rotation = rotation_matrix_from_euler_xyz_deg([0.0, -45.0, 0.0]) 
    # Rotation.from_euler("ZYX", [0, -45, 0], degrees=True).as_matrix()
    cathPax5.material = "G4_Pb"
    cathPax5.color = colors["pink"]


    cathPax6 = sim.add_volume("BoxVolume", "cathPax6") # Corner tab on small door 2 above the patient
    cathPax6.size = [25*cm, 1*mm, 25*cm] # 1 mm Pb !! See cathPax diagram
    cathPax6.translation = [27.5*cm, -30.0*cm, 12.5*cm] 
    cathPax6.material = "G4_Pb"
    cathPax6.color = colors["pink"]


    def add_source_to_tube(sim, tube, name, kvp=100, filter_al_mm=3.5, filter_cu_mm=0.9):
        """
        Add a GenericSource attached to the given tube volume.
        """
        src = sim.add_source("GenericSource", f"{name}_src")
        src.particle = "gamma"
        # Attach source to tube
        src.attached_to = tube.name

        # Energy 
        src.energy.type = "histogram"

        # # Additional X-ray filtration
        # try:
        #     src.energy.filter = {
        #         "G4_Al": filter_al_mm * mm,
        #         "G4_Cu": filter_cu_mm * mm,
        #     }
        # except Exception:
        #     pass

        # # Direction spectrum       
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
        # src.direction.acceptance_angle.volumes = [tube.name]
        # src.direction.acceptance_angle.normal_flag = True
        # src.direction.acceptance_angle.normal_vector = [0, 0, 1]
        # src.direction.acceptance_angle.normal_tolerance = 5 * deg
        # src.direction.acceptance_angle.skip_policy = "SkipEvents"

        # Energy spectrum with filtration   
        spectrum = sp.Spek(kvp, th=10, physics="kqp")
        spectrum.filter("Al", filter_al_mm).filter("Cu", filter_cu_mm)

        energy_bins, weights = spectrum.get_spectrum()
        src.energy.histogram_energy = energy_bins * keV
        src.energy.histogram_weight = weights

        # Number of primaries (GATE 10.x syntax)
        src.n = N_EVENTS_TEST

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
        tube.color = colors["blue"]

        # --------- Detector panel ----------
        panel = sim.add_volume("BoxVolume", f"{name}_panel")
        panel.size = [40*cm, 40*cm, 3*cm]
        panel.translation = translation + rotation @ np.array([0.0, -40.0*cm, 67.0*cm])
        # panel.rotation = rotation_matrix_from_euler_xyz_deg([0.0, 0.0, 0.0])
        panel.material = "G4_CESIUM_IODIDE"
        panel.color = colors["blue"]

        # --------- Source attached to the tube ----------        
        filter_al_mm = 3.5
        filter_cu_mm = 0.9
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
        tube.color = colors["yellow"]

        # --------- Detector panel ----------
        panel = sim.add_volume("BoxVolume", f"{name}_panel")
        panel.size = [3*cm, 40*cm, 40*cm]
        panel.translation = translation + rotation @ np.array([-59.0*cm, 0.0, -58.0*cm])
        # panel.rotation = rotation_matrix_from_euler_xyz_deg([0.0, 0.0, 0.0])
        panel.material = "G4_CESIUM_IODIDE"
        panel.color = colors["yellow"]

        # --------- Source attached to the tube ----------
        src = add_source_to_tube(sim, tube, name, kvp, filter_al_mm, filter_cu_mm)

        return {"arc": arc, "tube": tube, "src": src}

    # Build vertical Carm (ceiling) at iso center near [0,0,80cm]
    kV_v = int(80)
    translation_v = np.array([0, -120*cm, 0*cm])
    rotation_v = Rotation.from_euler("ZYX", [0, 180, 180], degrees=True).as_matrix()
    filter_al_mm_v = 3.5
    filter_cu_mm_v = 0.9
    Carm_v = build_Carm_vertical(sim, "Carm_vertical", translation_v, rotation_v, kvp=kV_v, filter_al_mm=filter_al_mm_v, filter_cu_mm=filter_cu_mm_v)
    kV_h = int(70)
    translation_h = np.array([0, -80*cm, 75*cm])
    rotation_h = Rotation.from_euler("ZYX", [180, 0, 0], degrees=True).as_matrix()
    filter_al_mm_h = 3.5
    filter_cu_mm_h = 0.9
    Carm_h = build_Carm_horizontal(sim, "Carm_horizontal", translation_h, rotation_h, kvp=kV_h, filter_al_mm=filter_al_mm_h, filter_cu_mm=filter_cu_mm_h)

    # Physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    # sim.physics_manager.set_production_cut("world", "gamma", 0.1*mm)
    # sim.physics_manager.set_production_cut("world", "e-", 0.1*mm)
    # sim.physics_manager.set_production_cut("world", "e+", 0.1*mm)

    # return references
    return {
        "table": table,
        "Head_Body": Head_Body,
        "Head_Skeleton": Head_Skeleton,
        "thorax": thorax,
        "abdo": abdo,
        "sphere": sphere,
        "Carm_v": Carm_v,
        "Carm_h": Carm_h
    }

# ------------------------
# Add DoseActor for ICRU sphere (if available in your build)
# ------------------------
def add_sphere_dose_actor(sim, sphere_name="icru_sphere"):
    try:
        actor = sim.add_actor("DoseActor", "SphereDose")
        # attach actor to the volume by its physical name
        actor.SetPhysicalVolumeName(sphere_name)
        # actor.output = out_file
        actor.write_to_disk = True
        actor.spacing = [5*mm, 5*mm, 5*mm]  # small voxels
        # print("DoseActor for sphere created, output:", out_file)
        return actor
    except Exception as e:
        print("Warning: Could not add a DoseActor; error:", e)
        return None

# ------------------------
# Main
# ------------------------
def main(do_pv_preview=True):
    sim = gate.Simulation()
    sim.output_dir = OUT_DIR

    # Setup GDML export via visualization
    sim.visu = True
    sim.visu_type = "vrml"
    sim.visu_filename = VRML_FILE
    print("VRML will be written to:", VRML_FILE)

    # Build scene: table, patient, ICRU sphere, Carm arcs, cathPax
    refs = create_scene(sim)

    # Add DoseActor for the ICRU sphere
    add_sphere_dose_actor(sim, sphere_name="icru_sphere")

    sim.check_volumes_overlap = False

    sim.run()

    # Launch Gate viewer externally if desired
    # try:
    #     subprocess.Popen(["Gate", "--qt", GDML_FILE])
    # except Exception:
    #     print("No simulation or open manually:", GDML_FILE)


if __name__ == "__main__":
    main(do_pv_preview=PV_AVAILABLE)
