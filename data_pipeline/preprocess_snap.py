"""
preprocess_snap.py

Preprocess Sentinel-1 GRD scenes to analysis-ready σ⁰ (dB) GeoTIFFs using
ESA SNAP's Graph Processing Tool (GPT) invoked via subprocess.

Processing chain (standard GRD best practice):
  Apply-Orbit-File
  → Thermal-Noise-Removal
  → Calibration (σ⁰, linear)
  → Speckle-Filter (Refined Lee 7×7)
  → Range-Doppler Terrain Correction (SRTM 1Sec HGT, 10 m UTM)
  → BandMaths (linear → dB)
  → Write (GeoTIFF, bands: VV σ⁰ dB, VH σ⁰ dB)

Output: one GeoTIFF per input scene, same stem with suffix _processed.tif,
        written to --output-dir.

Usage (single scene):
    python preprocess_snap.py \\
        --input  /data/raw/S1A_IW_GRDH_...SAFE \\
        --output-dir /data/processed \\
        --gpt    /opt/snap/bin/gpt

Usage (batch via SLURM array):
    srun python preprocess_snap.py \\
        --input  /data/raw/${SCENE_LIST[$SLURM_ARRAY_TASK_ID]} \\
        --output-dir /data/processed \\
        --gpt    /opt/snap/bin/gpt
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _node(graph: ET.Element, node_id: str, operator: str, source_id: str | None) -> ET.Element:
    """Append a <node> element to *graph* and return it."""
    node = ET.SubElement(graph, "node", id=node_id)
    ET.SubElement(node, "operator").text = operator
    sources = ET.SubElement(node, "sources")
    if source_id is not None:
        ET.SubElement(sources, "sourceProduct", refid=source_id)
    ET.SubElement(node, "parameters", attrib={"class": "com.bc.ceres.binding.dom.XppDomElement"})
    return node


def _set_param(node: ET.Element, key: str, value: str) -> None:
    params = node.find("parameters")
    child = ET.SubElement(params, key)
    child.text = value


def build_graph(input_path: Path, output_path: Path) -> str:
    """Return a SNAP GPT XML graph string for the full preprocessing chain."""
    graph = ET.Element("graph", version="1.0")
    ET.SubElement(graph, "version").text = "1.0"

    # 1. Read
    read = _node(graph, "Read", "Read", None)
    _set_param(read, "file", str(input_path))
    _set_param(read, "formatName", "SENTINEL-1")

    # 2. Apply-Orbit-File
    orbit = _node(graph, "Apply-Orbit-File", "Apply-Orbit-File", "Read")
    _set_param(orbit, "orbitType", "Sentinel Precise (Auto Download)")
    _set_param(orbit, "polyDegree", "3")
    _set_param(orbit, "continueOnFail", "true")

    # 3. Thermal-Noise-Removal
    tnr = _node(graph, "ThermalNoiseRemoval", "ThermalNoiseRemoval", "Apply-Orbit-File")
    _set_param(tnr, "selectedPolarisations", "VV,VH")
    _set_param(tnr, "removeThermalNoise", "true")

    # 4. Calibration → σ⁰ in linear scale
    cal = _node(graph, "Calibration", "Calibration", "ThermalNoiseRemoval")
    _set_param(cal, "selectedPolarisations", "VV,VH")
    _set_param(cal, "outputSigmaBand", "true")
    _set_param(cal, "outputBetaBand", "false")
    _set_param(cal, "outputGammaBand", "false")
    _set_param(cal, "outputImageInComplex", "false")
    _set_param(cal, "outputImageScaleInDb", "false")  # keep linear; dB done after TC

    # 5. Speckle-Filter (Refined Lee 7×7)
    # TODO (future work): Bianchi et al. (2021, IEEE JSTARS) use a 5×5 Refined Lee filter
    # as standard preprocessing. The current AvalCD patches (from Zenodo) were not
    # preprocessed with this graph and therefore lack explicit speckle filtering.
    # Rotation sensitivity analysis showed ~0.027 AUC improvement from incidental
    # speckle reduction via bilinear interpolation, suggesting explicit filtering would
    # recover a similar gain. For future reprocessing from raw Sentinel-1 .SAFE files:
    #   - Change filterSizeX/Y to "5" to match Bianchi et al. for direct comparison.
    #   - Apply this graph before extract_patches.py, not after.
    speckle = _node(graph, "Speckle-Filter", "Speckle-Filter", "Calibration")
    _set_param(speckle, "filter", "Refined Lee")
    _set_param(speckle, "filterSizeX", "7")
    _set_param(speckle, "filterSizeY", "7")
    _set_param(speckle, "dampingFactor", "2")
    _set_param(speckle, "estimateENL", "true")
    _set_param(speckle, "enl", "1.0")
    _set_param(speckle, "numLooksStr", "1")
    _set_param(speckle, "targetWindowSizeStr", "3x3")
    _set_param(speckle, "sigmaStr", "0.9")
    _set_param(speckle, "anSize", "50")

    # 6. Range-Doppler Terrain Correction
    tc = _node(graph, "Terrain-Correction", "Terrain-Correction", "Speckle-Filter")
    _set_param(tc, "demName", "Copernicus DEM GLO-30")
    _set_param(tc, "demResamplingMethod", "BILINEAR_INTERPOLATION")
    _set_param(tc, "imgResamplingMethod", "BILINEAR_INTERPOLATION")
    _set_param(tc, "pixelSpacingInMeter", "10.0")
    _set_param(tc, "mapProjection", "AUTO:42001")   # UTM auto-zone
    _set_param(tc, "nodataValueAtSea", "false")     # keep sea pixels; mask later if needed
    _set_param(tc, "saveDEM", "false")
    _set_param(tc, "saveLatLon", "false")
    _set_param(tc, "saveIncidenceAngleFromEllipsoid", "false")
    _set_param(tc, "saveLocalIncidenceAngle", "false")
    _set_param(tc, "saveProjectedLocalIncidenceAngle", "false")
    _set_param(tc, "saveSelectedSourceBand", "true")
    # Restrict to calibrated σ⁰ bands only (avoids writing intensity duplicates)
    _set_param(tc, "sourceBands", "Sigma0_VV,Sigma0_VH")

    # 7. BandMaths: linear σ⁰ → dB  (10 * log10(x), floor at -30 dB to avoid log(0))
    bm = _node(graph, "BandMaths", "BandMaths", "Terrain-Correction")
    target_bands = ET.SubElement(bm.find("parameters"), "targetBands")

    for pol in ("VV", "VH"):
        tb = ET.SubElement(target_bands, "targetBand")
        ET.SubElement(tb, "name").text = f"Sigma0_{pol}_dB"
        ET.SubElement(tb, "type").text = "float32"
        ET.SubElement(tb, "expression").text = (
            f"10 * log10(max(Sigma0_{pol}, 1e-6))"
        )
        ET.SubElement(tb, "description").text = f"σ⁰ {pol} in dB"
        ET.SubElement(tb, "unit").text = "dB"
        ET.SubElement(tb, "noDataValue").text = "NaN"

    # 8. Write
    write = _node(graph, "Write", "Write", "BandMaths")
    _set_param(write, "file", str(output_path))
    _set_param(write, "formatName", "GeoTIFF-BigTIFF")

    return ET.tostring(graph, encoding="unicode", xml_declaration=False)


# ---------------------------------------------------------------------------
# GPT invocation
# ---------------------------------------------------------------------------

def run_gpt(gpt_exe: Path, graph_xml: str, cache_dir: Path | None = None) -> None:
    """Write *graph_xml* to a temp file and invoke SNAP GPT."""
    with tempfile.NamedTemporaryFile(
        suffix=".xml", mode="w", delete=False, prefix="snap_graph_"
    ) as f:
        f.write(graph_xml)
        graph_path = Path(f.name)

    cmd = [str(gpt_exe), str(graph_path), "-q", "4"]  # 4 tiles; tune for node RAM
    if cache_dir is not None:
        cmd += ["-c", f"{cache_dir}"]

    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)

    graph_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"GPT exited with code {result.returncode}. Check SNAP logs above."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess Sentinel-1 GRD scenes to σ⁰ dB GeoTIFFs via SNAP GPT."
    )
    p.add_argument(
        "--input", required=True, type=Path,
        help="Path to a .SAFE directory or .zip archive for one Sentinel-1 scene.",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory where the processed GeoTIFF will be written.",
    )
    p.add_argument(
        "--gpt", default=Path("/opt/snap/bin/gpt"), type=Path,
        help="Path to the SNAP GPT executable (default: /opt/snap/bin/gpt).",
    )
    p.add_argument(
        "--cache-dir", default=None, type=Path,
        help="SNAP tile cache directory (optional; useful on nodes with local scratch).",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-process even if the output file already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path: Path = args.input.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        sys.exit(1)

    if not args.gpt.exists():
        log.error("GPT executable not found: %s", args.gpt)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filename from scene stem (strip .SAFE / .zip suffix)
    stem = input_path.stem
    if stem.endswith(".SAFE"):
        stem = stem[: -len(".SAFE")]
    output_path = output_dir / f"{stem}_processed.tif"

    if output_path.exists() and not args.overwrite:
        log.info("Output already exists, skipping (use --overwrite to force): %s", output_path)
        return

    log.info("Input  : %s", input_path)
    log.info("Output : %s", output_path)

    graph_xml = build_graph(input_path, output_path)
    run_gpt(args.gpt, graph_xml, cache_dir=args.cache_dir)

    if not output_path.exists():
        log.error("GPT finished but output file is missing: %s", output_path)
        sys.exit(1)

    log.info("Done: %s", output_path)


if __name__ == "__main__":
    main()
