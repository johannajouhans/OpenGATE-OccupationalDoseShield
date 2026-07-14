# OpenGATE-OccupationalDoseShield

Monte Carlo simulation (OpenGATE / Geant4) of occupational dosimetry in a **biplane angiography room**, with **CathPax** lead shielding.

Repository: [github.com/johannajouhans/OpenGATE-OccupationalDoseShield](https://github.com/johannajouhans/OpenGATE-OccupationalDoseShield)

License **GPL-3.0** · **Python 3** · Engine **OpenGATE 10.0.3 (Geant4)**

## Overview

This repository contains an [OpenGATE](https://opengate.readthedocs.io/) (Geant4-based Monte Carlo toolkit) simulation of a **biplane C-arm angiography room**. It estimates the radiation dose received by a worker (modeled as an ICRU sphere) standing at a fixed position in the room, with or without the **CathPax** lead-shielding system in place, for two X-ray tubes operated simultaneously (one ceiling-mounted "vertical" C-arm and one floor-mounted "horizontal" C-arm).

The simulated dose is rescaled from measured clinical kerma values (mGy, at a given kV/filtration), so the output is expressed as an absolute, clinically-relevant dose (Hp(10) equivalent, in mSv) rather than an arbitrary Monte Carlo unit.

## Repository contents

| File | Description |
|---|---|
| **simu_double_Carm.py** | Main dosimetric script. Builds the full scene (room, table, patient, both C-arms, ICRU sphere, optional CathPax shielding), runs the simulation for each C-arm in a separate subprocess, rescales the dose to the measured kerma, and reports the Hp(10) dose with its uncertainty. |
| **simu_double_Carm_visu.py** | Visualization / scene-export script. Builds the same geometry (with additional room and shielding-wall meshes) and exports it as a VRML scene for 3D inspection, with a small number of events (fast preview only). |
| **Carm_HORIZONTAL.stl / Carm_VERTICAL.stl** | 3D meshes of the two C-arm gantries (created in Blender3D). |
| **Head_Body.stl / Head_Skeleton.stl** | Simplified patient phantom (soft tissue + skeleton). |
| **room_HD49.stl** | Room / floor mesh used in the visualization scene. |
| **paravent.stl** | Lead protective wall mesh used in the visualization scene. |
| **cathPax\*.stl** | Reference meshes for the CathPax shielding panels (the shielding itself is built parametrically as lead boxes directly in the scripts). |
| **LICENSE** | GPL-3.0. |

## How it works — simu_double_Carm.py

- **User parameters** — measured kerma (mGy), tube voltage (kV), field size at the panel, source-to-detector distance, and Al/Cu filtration, set at the top of the script for each of the two C-arms.
- **Kerma → photon count** — the function `kerma_to_photons()` builds the tube's real spectrum with [SpekPy](https://bitbucket.org/spekpy/spekpy_release), derives the kerma-per-photon from `spectrum.get_kerma()`/`get_flu()`, and back-calculates the number of photons corresponding to the measured kerma at the panel.
- **Two-worker architecture** — because a GATE simulation can only run once per Python process, the master process launches **one subprocess per C-arm**, each building its own scene and running Nref reference photons (1,000,000 by default, or 100 in visualization mode).
- **Dose scoring** — a DoseActor attached to a water-filled ICRU sphere (representing the worker) records deposited dose and its Monte Carlo uncertainty for each C-arm.
- **Rescaling & combination** — each worker's simulated dose is scaled by `scale = N_measured / Nref`, then both C-arms' contributions are summed for simultaneous biplane operation.
- **Result** — the mean dose in the sphere is converted to **Hp(10) (mSv)**, assuming a photon radiation weighting factor of 1, with absolute and relative statistical uncertainty. Results are saved as `.npz` files in `output_scaled/`.

> **NOTE**
> Set `With_CathPax = True` to add the CathPax lead panels (1–2 mm Pb, six parametric plates `cathPax1`–`cathPax6`) between the ICRU sphere and the C-arm/patient, to quantify the dose reduction they provide.

### The kerma → photons conversion, in code

```python
def kerma_to_photons(kerma_mGy, kV, field_size_cm_at_panel,
                      kerma_distance_cm, source_panel_distance_cm,
                      filter_al_mm=3.5, filter_cu_mm=0.9):
    K = kerma_mGy * 1e-3  # (1) Gy
    spectrum = sp.Spek(kV, th=10, physics="kqp")
    spectrum.filter("Al", filter_al_mm).filter("Cu", filter_cu_mm)
    kerma_per_photon = spectrum.get_kerma() / spectrum.get_flu()
    fluence_mesure = K / kerma_per_photon
    fluence_panel = fluence_mesure * (kerma_distance_cm / source_panel_distance_cm) ** 2
    N = fluence_panel * (field_size_cm_at_panel ** 2)
    return N
```

## Requirements

- Python 3
- opengate (OpenGATE 10.0.3, Geant4 Monte Carlo bindings)
- spekpy (X-ray tube spectrum generation — loaded lazily, only needed when a simulation actually runs)
- numpy
- scipy (`scipy.spatial.transform.Rotation`)

## Usage

```bash
python simu_double_Carm.py
```

Edit the "USER PARAMETERS" block at the top of the file to match your measured kerma, kV, filtration and geometry before running. Set `Visualisation = True` for a fast, low-statistics run with the GATE 3D viewer enabled, or `False` for a full-statistics production run (1×10⁶ photons/worker).

For a pure 3D scene preview (no dosimetric calculation), run:

```bash
python simu_double_Carm_visu.py
```

which exports a VRML file to `output/scene_Carm_biplane.wrl`.

## Notes & known limitations

- The angular emission histogram of each X-ray tube is currently a simplified placeholder (a rough forward cone); the code marks the actual anode heel effect as a TODO to be implemented from real measured data.
- Filter thickness (Al/Cu) is passed into the spectrum generation itself; the alternate energy-filter branch in the source is present in the code but currently commented out.

## License

GPL-3.0 — see [LICENSE](LICENSE).

## Citation / acknowledgements

3D meshes (C-arms, room, patient phantom) created in Blender3D by H. Necib. If you reuse or extend this code, please cite/credit this repository and its author.

## At a glance

| | | |
|---|---|---|
| **2** | **6** | **GPL-3.0** |
| complementary Python scripts | parametric CathPax panels | repository license |
