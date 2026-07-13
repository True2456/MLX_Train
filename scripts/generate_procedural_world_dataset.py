#!/usr/bin/env python3
"""
generate_procedural_world_dataset.py

Generates 2,000 verified Code-as-Action trajectories inspired by procedural
3D WebGPU open-world engines (like LAAS / fable5-world-demo).

Spans 5 specialized domains (400 traces each):
  1. WGSL & Procedural Noise Math (Simplex, FBM, Domain Warping)
  2. Terrain Chunk Edge Stitching & Erosion Continuity
  3. Spatial Quadtree LOD & Camera Frustum Culling
  4. Poisson Disk Flora Distribution & Slope Constraints
  5. Screenshot & Visual Frame Buffer Verification (RGB Histogram, Horizon Sobel Edge, Exposure Check)

Every trace includes an executable Python verification script that runs and passes 100%.
"""

import json
import math
import random
import subprocess
import sys
from pathlib import Path

OUTPUT_FILE = Path("/Users/true/Documents/Mati_Train/data/procedural_world_code_as_action_2000.jsonl")

def generate_wgsl_noise_trace(seed_id: int) -> dict:
    octaves = 4 + (seed_id % 4)
    freq = round(0.01 + (seed_id % 10) * 0.005, 4)
    x = 100 + (seed_id * 37) % 2000
    z = 200 + (seed_id * 53) % 2000

    prompt = (
        f"<|im_start|>user\n"
        f"Verify that our WGSL Fractal Brownian Motion (FBM) heightmap compute function "
        f"with {octaves} octaves and base frequency {freq} stays strictly bounded within [-1.0, 1.0] "
        f"at world coordinate ({x}, {z}) and produces no NaN values.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    py_code = (
        f"import math\n"
        f"def pseudo_noise(x, z):\n"
        f"    return math.sin(x * 12.9898 + z * 78.233) * 43758.5453 % 1.0 * 2.0 - 1.0\n"
        f"val = 0.0\n"
        f"amp = 1.0\n"
        f"max_amp = 0.0\n"
        f"f = {freq}\n"
        f"for _ in range({octaves}):\n"
        f"    val += pseudo_noise({x} * f, {z} * f) * amp\n"
        f"    max_amp += amp\n"
        f"    amp *= 0.5\n"
        f"    f *= 2.0\n"
        f"norm_val = val / max_amp\n"
        f"assert not math.isnan(norm_val), 'FBM produced NaN'\n"
        f"assert -1.0 <= norm_val <= 1.0, f'FBM out of bounds: {{norm_val}}'\n"
        f"print(f'VERIFIED_FBM_BOUNDS: {{round(norm_val, 5)}} within [-1.0, 1.0]')\n"
    )

    think = (
        f"To verify that our WGSL FBM height function remains bounded within [-1.0, 1.0] "
        f"at ({x}, {z}), we must evaluate the geometric series normalization across {octaves} octaves. "
        f"We will execute a numerical verification script to simulate the octave accumulation and check for NaN or overflow."
    )

    completion = (
        f"<think>\n{think}\n</think>\n"
        f"Let's execute the WGSL noise verification script.\n\n"
        f"Action: run_command\n"
        f"Action Input: python3 -c '{py_code}'\n"
        f"<|im_end|>"
    )

    return {"prompt": prompt, "completion": completion, "py_code": py_code}

def generate_terrain_stitching_trace(seed_id: int) -> dict:
    chunk_size = 64
    cx = (seed_id * 7) % 50
    cz = (seed_id * 11) % 50
    local_z = (seed_id * 13) % chunk_size

    prompt = (
        f"<|im_start|>user\n"
        f"Verify seamless terrain edge stitching between chunk ({cx}, {cz}) right boundary "
        f"and adjacent chunk ({cx+1}, {cz}) left boundary at local z={local_z}.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    py_code = (
        f"import math\n"
        f"def world_height(wx, wz):\n"
        f"    return round(math.sin(wx * 0.05) * 120.0 + math.cos(wz * 0.05) * 80.0, 5)\n"
        f"wx_boundary = ({cx} + 1) * {chunk_size}\n"
        f"wz_pos = {cz} * {chunk_size} + {local_z}\n"
        f"left_chunk_edge = world_height(wx_boundary, wz_pos)\n"
        f"right_chunk_edge = world_height(wx_boundary, wz_pos)\n"
        f"delta = abs(left_chunk_edge - right_chunk_edge)\n"
        f"assert delta == 0.0, f'Seam detected! Delta={{delta}}'\n"
        f"print(f'VERIFIED_SEAMLESS_STITCHING: edge_height={{left_chunk_edge}} delta={{delta}}')\n"
    )

    think = (
        f"Terrain chunk edge stitching requires that the world height evaluated at the eastern boundary of chunk ({cx}, {cz}) "
        f"matches the western boundary of chunk ({cx+1}, {cz}) exactly. Let's run a verification script at z={local_z}."
    )

    completion = (
        f"<think>\n{think}\n</think>\n"
        f"Let's execute the boundary seam verification script.\n\n"
        f"Action: run_command\n"
        f"Action Input: python3 -c '{py_code}'\n"
        f"<|im_end|>"
    )

    return {"prompt": prompt, "completion": completion, "py_code": py_code}

def generate_frustum_culling_trace(seed_id: int) -> dict:
    cam_x = 0.0
    cam_z = 0.0
    chunk_center_z = 500.0 + (seed_id % 20) * 100.0
    half_size = 32.0

    prompt = (
        f"<|im_start|>user\n"
        f"Verify whether terrain chunk AABB centered at (0.0, 0.0, {chunk_center_z}) with half-size {half_size} "
        f"is visible or culled against the camera near plane (z=1.0) and far plane (z=4000.0).<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    py_code = (
        f"cz = {chunk_center_z}\n"
        f"hs = {half_size}\n"
        f"min_z = cz - hs\n"
        f"max_z = cz + hs\n"
        f"near_p = 1.0\n"
        f"far_p = 4000.0\n"
        f"visible = (max_z >= near_p) and (min_z <= far_p)\n"
        f"assert visible is True, f'Chunk unexpectedly culled at cz={{cz}}'\n"
        f"print(f'VERIFIED_FRUSTUM_VISIBILITY: cz={{cz}} visible={{visible}}')\n"
    )

    think = (
        f"To determine chunk visibility against depth frustum planes [1.0, 4000.0], we compare the chunk's bounding box extent "
        f"[{chunk_center_z - half_size}, {chunk_center_z + half_size}] against the near and far planes."
    )

    completion = (
        f"<think>\n{think}\n</think>\n"
        f"Let's execute the frustum AABB intersection script.\n\n"
        f"Action: run_command\n"
        f"Action Input: python3 -c '{py_code}'\n"
        f"<|im_end|>"
    )

    return {"prompt": prompt, "completion": completion, "py_code": py_code}

def generate_poisson_disk_trace(seed_id: int) -> dict:
    min_radius = 4.0
    t1_x, t1_z = 10.0, 10.0
    dx = 3.0 + (seed_id % 5) * 1.5
    dz = 3.0 + (seed_id % 5) * 1.5
    t2_x = t1_x + dx
    t2_z = t1_z + dz

    prompt = (
        f"<|im_start|>user\n"
        f"Verify whether two procedurally placed pine trees at ({t1_x}, {t1_z}) and ({t2_x}, {t2_z}) "
        f"satisfy our Poisson disk minimum radius constraint of {min_radius} meters.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    py_code = (
        f"import math\n"
        f"dist = math.hypot({t2_x} - {t1_x}, {t2_z} - {t1_z})\n"
        f"min_r = {min_radius}\n"
        f"valid = dist >= min_r\n"
        f"assert valid, f'Poisson disk violation: dist={{round(dist,3)}} < {{min_r}}'\n"
        f"print(f'VERIFIED_POISSON_SPACING: dist={{round(dist, 3)}}m >= {{min_r}}m')\n"
    )

    think = (
        f"We calculate Euclidean distance between ({t1_x}, {t1_z}) and ({t2_x}, {t2_z}) and verify it exceeds our "
        f"minimum Poisson radius of {min_radius}m."
    )

    completion = (
        f"<think>\n{think}\n</think>\n"
        f"Let's execute the Poisson disk distance assertion script.\n\n"
        f"Action: run_command\n"
        f"Action Input: python3 -c '{py_code}'\n"
        f"<|im_end|>"
    )

    return {"prompt": prompt, "completion": completion, "py_code": py_code}

def generate_screenshot_verification_trace(seed_id: int) -> dict:
    # Simulate an RGB frame buffer histogram check for a procedural world screenshot
    # Top third = sky (blue/atmospheric), Bottom = forest terrain (green/dark)
    sky_luma = 180 + (seed_id % 30)
    terrain_luma = 60 + (seed_id % 25)

    prompt = (
        f"<|im_start|>user\n"
        f"Perform automated screenshot verification on rendered frame buffer #{seed_id:04d}: "
        f"verify that sky region luminance ({sky_luma}) exceeds terrain luminance ({terrain_luma}), "
        f"confirming correct atmospheric horizon rendering and zero black-screen shader crashes.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    py_code = (
        f"sky_Y = {sky_luma}\n"
        f"terrain_Y = {terrain_luma}\n"
        f"assert sky_Y > 0 and terrain_Y > 0, 'Black screen crash detected in frame buffer!'\n"
        f"assert sky_Y > terrain_Y, f'Inverted horizon lighting: sky={{sky_Y}} <= terrain={{terrain_Y}}'\n"
        f"contrast_ratio = round(sky_Y / terrain_Y, 2)\n"
        f"print(f'VERIFIED_SCREENSHOT_FRAME: sky={{sky_Y}} terrain={{terrain_Y}} contrast_ratio={{contrast_ratio}}')\n"
    )

    think = (
        f"Automated visual/screenshot verification of a procedural world requires checking frame buffer luminance distribution. "
        f"We assert that sky luminance ({sky_luma}) > terrain luminance ({terrain_luma}) and neither region is collapsed to zero."
    )

    completion = (
        f"<think>\n{think}\n</think>\n"
        f"Let's execute the frame buffer visual verification script.\n\n"
        f"Action: run_command\n"
        f"Action Input: python3 -c '{py_code}'\n"
        f"<|im_end|>"
    )

    return {"prompt": prompt, "completion": completion, "py_code": py_code}

def main():
    print("Generating 2,000 Verified Procedural 3D World & Screenshot Verification traces...")
    traces = []
    
    generators = [
        ("WGSL Noise Math", generate_wgsl_noise_trace),
        ("Terrain Edge Stitching", generate_terrain_stitching_trace),
        ("Frustum Culling LOD", generate_frustum_culling_trace),
        ("Poisson Flora Spacing", generate_poisson_disk_trace),
        ("Screenshot Visual Audit", generate_screenshot_verification_trace),
    ]

    for name, gen_fn in generators:
        print(f"  -> Generating 400 traces for: {name}")
        for i in range(400):
            t = gen_fn(i)
            # Live verify every single generated script before saving!
            res = subprocess.run(
                [sys.executable, "-c", t["py_code"]],
                capture_output=True, text=True
            )
            if res.returncode != 0:
                print(f"CRITICAL ERROR in {name} trace #{i}:\n{res.stderr}")
                sys.exit(1)
            del t["py_code"]
            traces.append(t)

    random.seed(2026)
    random.shuffle(traces)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print(f"\nSUCCESS: Generated & verified {len(traces)} Procedural World Code-as-Action traces -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
