# Python snippets for calculating ArtFID (StyleID) in each scene
# Then, refer to https://github.com/jiwoogit/StyleID for specific evaluation codes. :p
"""
@InProceedings{Chung_2024_CVPR,
    author    = {Chung, Jiwoo and Hyun, Sangeek and Heo, Jae-Pil},
    title     = {Style Injection in Diffusion: A Training-free Approach for Adapting Large-scale Diffusion Models for Style Transfer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {8795-8805}
}
"""
import subprocess
import csv
import re

models = ["gstyle_new", "stylos_old"]
scenes = ["garden", "M60", "train", "truck"]

# CSV header
csv_file = "gstyle_new_metrics.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "scene",
        "ArtFID", "FID", "LPIPS", "LPIPS_gray",
        "CFSD", "color_loss"
    ])

# Regex for parsing outputs
re_artfid = re.compile(r"ArtFID:\s*([0-9.]+)\s*FID:\s*([0-9.]+)\s*LPIPS:\s*([0-9.]+)\s*LPIPS_gray:\s*([0-9.]+)")
re_cfsd   = re.compile(r"CFSD:\s*([0-9.]+)")
re_color  = re.compile(r"color matching loss:\s*([0-9.]+)")

def run(cmd):
    """Run a command and return stdout string."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = proc.communicate()
    return out

for model in models:
    for scene in scenes:
        print(f"=== Running model={model} scene={scene} ===")

        # -------- run eval_artfid.py ----------
        out1 = run([
            "python", "eval_artfid.py",
            "--sty", "/projects/bfcb/hliu15/3d_style_sota_artfid/input_styles",
            "--cnt", f"/projects/bfcb/hliu15/3d_style_sota_artfid/input_contents/{scene}",
            "--tar", f"/projects/bfcb/hliu15/3d_style_sota_artfid/{model}/{scene}",
        ])

        ArtFID = FID = LPIPS = LPIPS_gray = CFSD = "nan"

        m1 = re_artfid.search(out1)
        if m1:
            ArtFID, FID, LPIPS, LPIPS_gray = m1.groups()

        m2 = re_cfsd.search(out1)
        if m2:
            CFSD = m2.group(1)

        # -------- run eval_histogan.py ----------
        out2 = run([
            "python", "eval_histogan.py",
            "--sty", "/projects/bfcb/hliu15/3d_style_sota_artfid/input_styles",
            "--tar", f"/projects/bfcb/hliu15/3d_style_sota_artfid/{model}/{scene}",
        ])

        color_loss = "nan"
        m3 = re_color.search(out2)
        if m3:
            color_loss = m3.group(1)

        # -------- write line to CSV --------
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                model, scene,
                ArtFID, FID, LPIPS, LPIPS_gray,
                CFSD, color_loss
            ])

        print("Done.\n")

print("All evaluations completed. CSV saved to:", csv_file)
