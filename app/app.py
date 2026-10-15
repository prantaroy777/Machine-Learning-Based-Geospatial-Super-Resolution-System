"""Local Streamlit interface for continuous optical GeoTIFFs."""
import hashlib
import sys
import uuid
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
from rasterio.enums import Resampling
import streamlit as st

from src.geospatial_utils import inspect_raster
from src.predict import predict
from src.reporting import output_report, save_json
from src.validation import compare_rasters


def resolve_path(value):
    path = Path(value.strip().strip('"')).expanduser()
    return path if path.is_absolute() else ROOT / path


def save_upload(upload, directory):
    """Use a content-based filename; never trust a browser-supplied file path."""
    buffer = upload.getbuffer()
    digest = hashlib.sha256(buffer).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.tif"
    if not target.exists():
        target.write_bytes(buffer)
    return target


def preview(path, stretch=None, bands=None):
    with rasterio.open(path) as ds:
        factor = min(1, 768 / max(ds.width, ds.height))
        bands = bands or ([1, 2, 3] if ds.count >= 3 else [1])
        a = ds.read(bands, out_shape=(len(bands), max(1, int(ds.height * factor)), max(1, int(ds.width * factor))),
                    resampling=Resampling.nearest, masked=True).astype("float32").filled(np.nan)
    if not np.isfinite(a).any():
        return np.zeros((a.shape[1], a.shape[2], 3), dtype="float32"), (0., 1.)
    lo, hi = stretch if stretch is not None else np.nanpercentile(a, [2, 98])
    a = np.nan_to_num(np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1))
    return (a[0] if len(bands) == 1 else a.transpose(1, 2, 0)), (float(lo), float(hi))


def spacing(info, scale=1):
    x, y = info["pixel_size"]
    return f"{x / scale:g} × {y / scale:g} {info['units']}"


def main():
    st.set_page_config(page_title="Geo Enhance", page_icon="🌍", layout="wide")
    st.title("Geo Enhance")
    st.write("A finer grid for your satellite imagery. Upload a GeoTIFF, choose your output, and download the result.")
    with st.sidebar:
        st.subheader("Quick start")
        st.markdown("1. Upload a **.tif / .tiff** file.\n2. Choose **4×** for 40 m → 10 m.\n3. Click **Create enhanced GeoTIFF**.")
        st.caption("Your files are processed on this computer. Uploads and results are saved under outputs/gui.")
        st.info("Smoother images do not necessarily contain new measured detail. Model predictions need validation against reference imagery.")
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    session_dir = ROOT / "outputs/gui" / st.session_state.session_id
    st.subheader("1. Choose your image")
    source_mode = st.radio("Image source", ["Upload GeoTIFF", "Local file path", "Try the demo"], horizontal=True, key="source_mode")
    source = None
    if source_mode == "Upload GeoTIFF":
        uploaded = st.file_uploader("Drag a GeoTIFF here or browse your files", type=["tif", "tiff"], key="upload")
        st.caption("Up to 200 MB. For larger rasters or files with external .msk/.aux.xml metadata, use Local file path. JPG/PNG photos do not contain the required map information.")
        if uploaded is not None:
            try:
                source = save_upload(uploaded, session_dir / "uploads")
            except OSError as exc:
                st.error(f"Could not save the upload: {exc}")
    elif source_mode == "Local file path":
        value = st.text_input("GeoTIFF file path", placeholder=r"D:\imagery\input_40m.tif", key="source_path")
        if value.strip():
            source = resolve_path(value)
    else:
        source = ROOT / "data/demo/synthetic_40m.tif"
        st.warning("This demo is a synthetic pattern, not a real satellite observation.")
        if not source.is_file():
            # A fresh GitHub checkout excludes local demo data. Create a small,
            # session-owned fixture rather than requiring a terminal command.
            from examples.make_demo import write
            source = session_dir / "demo/synthetic_40m.tif"
            if not source.is_file():
                y, x = np.mgrid[:64, :64]
                data = np.stack([.25 + .15 * np.sin(x / 4), .3 + .1 * np.cos(y / 5),
                                 .2 + .15 * np.sin((x + y) / 6)]).astype("float32")
                source.parent.mkdir(parents=True, exist_ok=True)
                write(source, data, 500000, 40)

    info = None
    if source is not None:
        try:
            info = inspect_raster(source)
        except (OSError, ValueError) as exc:
            st.error(f"Could not read this GeoTIFF. Check the file path and that it contains a coordinate system. Details: {exc}")
    st.subheader("2. Choose the enhancement")
    preview_bands = None
    if info and info['bands'] >= 3:
        with st.expander("Preview band colors"):
            st.caption("Choose red, green and blue display bands. This affects previews only; all bands are processed and exported.")
            color_columns = st.columns(3)
            preview_bands = [column.selectbox(label, list(range(1, info['bands'] + 1)), index=index,
                                              key=f"preview_{label}")
                             for index, (column, label) in enumerate(zip(color_columns, ["Red", "Green", "Blue"]))]
            st.caption("Band descriptions: " + ", ".join(f"{i+1}: {name or 'unnamed'}" for i, name in enumerate(info['band_descriptions'])))
    settings, details = st.columns([1, 1])
    with settings:
        scale = st.selectbox("Increase resolution by", [2, 4, 8], index=1,
                             format_func=lambda n: {2: "2× — 40 m → 20 m", 4: "4× — 40 m → 10 m", 8: "8× — 40 m → 5 m (experimental)"}[n], key="scale")
        st.caption("Examples assume a 40 m input. Your actual output spacing appears alongside.")
        method_label = st.radio("Enhancement method", ["Smooth resize — ready to use", "Trained model — checkpoint required"], key="method")
        method = "model" if method_label.startswith("Trained") else "bicubic"
        checkpoint = None
        if method == "model":
            value = st.text_input("Trained checkpoint path (.pt)", key="checkpoint")
            if value.strip():
                checkpoint = resolve_path(value)
            st.caption("Use weights trained for your sensor, bands and scale. The bundled synthetic demo weights are not suitable for satellite enhancement.")
        else:
            st.caption("Bicubic interpolation produces a smoother grid. No training or GPU is required.")
        with st.expander("Advanced settings"):
            tile = st.select_slider("Tile size", [64, 128, 256, 512], value=128, key="tile")
            st.caption("Use a smaller tile if memory runs low.")
            device = st.selectbox("Compute device", ["auto", "cpu", "cuda"], key="device")
            ensemble = st.checkbox("Average eight orientations (model only, slower)", value=False, disabled=method != "model", key="ensemble")
            overviews = st.checkbox("Build GIS preview overviews", value=True, key="overviews")
    with details:
        if info:
            st.metric("Input pixel spacing", spacing(info))
            st.metric("Output pixel spacing", spacing(info, scale))
            st.write(f"**{info['width']:,} × {info['height']:,}** → **{info['width'] * scale:,} × {info['height'] * scale:,}** pixels · {info['bands']} bands")
            estimated_mb = info['width'] * info['height'] * scale ** 2 * info['bands'] * 4 / 1024 ** 2
            st.caption(f"Uncompressed output data: approximately {estimated_mb:,.1f} MB. Actual file size depends on compression.")
            if info['units'] == "degrees":
                st.warning("This raster uses degrees. Reproject to a suitable meter-based CRS before interpreting its pixels as meters.")
            with st.expander("Map information"):
                st.json(info)
        else:
            st.info("Choose a GeoTIFF to see its pixel size and planned output resolution.")
    if scale == 8:
        st.warning("8× creates 64 output pixels per input pixel. Treat the added detail as uncertain.")
    if checkpoint is not None and not checkpoint.is_file():
        st.error("Checkpoint not found. Enter the path to a trained .pt file.")
    ready = info is not None and (method != "model" or (checkpoint is not None and checkpoint.is_file()))
    if st.button("Create enhanced GeoTIFF", type="primary", disabled=not ready, key="generate"):
        try:
            destination = session_dir / f"enhanced_{scale}x_{method}_{uuid.uuid4().hex[:8]}.tif"
            with st.status("Creating your GeoTIFF…", expanded=True) as status:
                st.write(f"Processing {info['bands']} bands at {scale}× scale. Large rasters can take several minutes.")
                progress_bar = st.progress(0, text="Preparing raster tiles")
                def update_progress(done, total):
                    progress_bar.progress(done / total, text=f"Processed {done:,} of {total:,} tiles")
                result = predict(source, destination, scale=scale, checkpoint=checkpoint, method=method,
                                 tile_size=tile, device=device, ensemble=ensemble,
                                 progress=update_progress, build_overviews=overviews)
                report = output_report(result)
                save_json(destination.with_suffix(".json"), report)
                before, stretch = preview(source, bands=preview_bands)
                after, _ = preview(result, stretch, preview_bands)
                with rasterio.open(result) as ds:
                    tags = ds.tags()
                st.session_state.result = dict(path=str(result), before=before, after=after, info=info,
                                               scale=scale, method=method, tags=tags, report=report)
                status.update(label="Your GeoTIFF is ready", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"Could not create the output: {exc}")
            st.caption("For GPU memory errors, select a smaller tile or CPU in Advanced settings.")
    if "result" in st.session_state:
        result = st.session_state.result
        if not Path(result['path']).is_file():
            st.warning("The last output file was moved or deleted. Generate a new result to continue.")
            return
        if 'report' not in result:
            result['report'] = output_report(result['path'])
        st.subheader("3. Review and save your result")
        st.caption("Last completed result. Changing settings above does not change this file; generate again to apply them.")
        st.success(f"Created a {spacing(result['info'], result['scale'])} grid using {'model prediction' if result['method'] == 'model' else 'bicubic interpolation'}.")
        if result['method'] == "model" and result['tags'].get("SR_BEATS_BICUBIC") != "True":
            st.warning("This model has not beaten the interpolation baseline in validation. Compare carefully before using it.")
        left, right = st.columns(2)
        left.image(result['before'], caption="Original image", width="stretch")
        right.image(result['after'], caption="Output image", width="stretch")
        st.caption("Both previews use the same contrast stretch and selected display bands. Full detail and every input band are preserved in the GeoTIFF.")
        path = Path(result['path'])
        st.write("Saved on this computer:")
        st.code(str(path), language=None)
        if path.exists() and path.stat().st_size <= 128 * 1024 ** 2:
            st.download_button("Download enhanced GeoTIFF", data=path.read_bytes(), file_name=path.name,
                               mime="image/tiff", on_click="ignore", key="download")
        else:
            st.info("This file is too large for the in-browser download. Open or copy it from the saved location above.")
        st.download_button("Download processing report", data=json.dumps(result['report'], indent=2),
                           file_name=path.stem + '.json', mime="application/json", on_click="ignore", key="report_download")
        with st.expander("Check accuracy against a reference image"):
            st.write("Use an independently observed reference with the same grid, CRS, bands and acquisition conditions. Misaligned rasters are rejected.")
            reference = st.text_input("Reference GeoTIFF path", key="reference")
            data_range = st.number_input("Signal range for PSNR", min_value=0.000001, value=1., format="%.6f", key="data_range")
            st.caption("Use 1 for reflectance in [0,1], or the appropriate range in your stored/physical units. Do not estimate it separately for each image.")
            apply_scale = st.checkbox("Apply GeoTIFF scale/offset metadata before comparing", key="apply_scale")
            if st.button("Compare with reference", disabled=not reference.strip(), key="compare"):
                try:
                    with st.spinner("Comparing valid pixels…"):
                        comparison = compare_rasters(path, resolve_path(reference), data_range, apply_scale_offset=apply_scale)
                    result['comparison'] = comparison
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
            if 'comparison' in result:
                comparison = result['comparison']
                st.caption(f"Last reference comparison: {comparison['reference']}")
                st.json(comparison)
                st.download_button("Download accuracy report", data=json.dumps(comparison, indent=2),
                                   file_name=path.stem + '_accuracy.json', mime="application/json", on_click="ignore")


if __name__ == "__main__":
    main()
