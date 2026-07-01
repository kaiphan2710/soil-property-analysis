import os
import sys
import json
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import io
from osgeo import gdal
import re

# Parse wavelengths from HDR file
def parse_hdr_wavelengths(hdr_path):
    if not os.path.exists(hdr_path):
        return []
    try:
        with open(hdr_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        match = re.search(r'wavelength\s*=\s*\{([^}]+)\}', content, re.IGNORECASE)
        if not match:
            return []
        
        raw_vals = match.group(1).replace('\n', '').replace('\r', '').split(',')
        wavelengths = []
        for val in raw_vals:
            val_clean = val.strip()
            if val_clean:
                try:
                    wavelengths.append(float(val_clean))
                except ValueError:
                    pass
        return wavelengths
    except Exception as e:
        print(f"Error parsing wavelengths: {e}")
        return []

# Stretch helper for RGB visualizer
def stretch_band(data):
    p2, p98 = np.percentile(data, (2, 98))
    if p98 == p2:
        p98 += 1.0
    stretched = np.clip((data - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    return stretched

class HyperspectralServerHandler(BaseHTTPRequestHandler):
    # Global state to keep dataset and cache loaded.
    # Note: In a production setting we'd use objects, but since this is a single-client lightweight helper,
    # global class variables are perfectly fine.
    dataset = None
    npz_datacube = None
    hdr_path = ""
    data_path = ""
    bands_cache = {}
    api_key = ""

    def _set_headers(self, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(status=200)

    def do_POST(self):
        # Validate api key
        req_key = self.headers.get("X-API-Key", "")
        if HyperspectralServerHandler.api_key and req_key != HyperspectralServerHandler.api_key:
            self._set_headers(status=401)
            self.wfile.write(json.dumps({"error": "Unauthorized"}).encode('utf-8'))
            return

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Read JSON body
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            req_data = json.loads(post_data.decode('utf-8')) if post_data else {}
        except Exception:
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": "Invalid JSON format"}).encode('utf-8'))
            return

        if path == "/connect":
            self.handle_connect()
        elif path == "/list_dir":
            self.handle_list_dir(req_data)
        elif path == "/open":
            self.handle_open(req_data)
        elif path == "/rgb":
            self.handle_rgb(req_data)
        elif path == "/pixel":
            self.handle_pixel(req_data)
        elif path == "/save_roi":
            self.handle_save_roi(req_data)
        elif path == "/get_rois":
            self.handle_get_rois(req_data)
        elif path == "/save_rois":
            self.handle_save_rois(req_data)
        else:
            self._set_headers(status=404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode('utf-8'))

    def handle_connect(self):
        self._set_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))

    def handle_list_dir(self, data):
        path = data.get("path", "")
        if not path:
            path = os.getcwd()
        
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
        
        if not os.path.exists(path) or not os.path.isdir(path):
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": f"Path '{path}' is not a valid directory or does not exist."}).encode('utf-8'))
            return

        try:
            items = []
            for entry in os.scandir(path):
                is_dir = entry.is_dir()
                is_hdr = entry.name.lower().endswith('.hdr')
                is_npz = entry.name.lower().endswith('.npz')
                if is_dir or is_hdr or is_npz:
                    try:
                        stat = entry.stat()
                        mtime = stat.st_mtime
                        ctime = stat.st_ctime
                    except Exception:
                        mtime = 0
                        ctime = 0
                    items.append({
                        "name": entry.name,
                        "path": entry.path,
                        "is_dir": is_dir,
                        "mtime": mtime,
                        "ctime": ctime
                    })
            
            items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
            
            parent_dir = os.path.dirname(path)
            if parent_dir == path:
                parent_dir = ""

            resp = {
                "current_dir": path,
                "parent_dir": parent_dir,
                "items": items
            }
            self._set_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
        except Exception as e:
            self._set_headers(status=500)
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

    def handle_open(self, data):
        file_path = data.get("hdr_path")
        if not file_path:
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": "hdr_path is required"}).encode('utf-8'))
            return

        if file_path.lower().endswith('.npz'):
            try:
                npz_data = np.load(file_path, allow_pickle=True)
                if 'datacube' not in npz_data:
                    raise Exception("NPZ file does not contain 'datacube'")
                
                datacube = npz_data['datacube']
                shape = datacube.shape
                if len(shape) != 3:
                    raise Exception(f"Datacube must be 3D, got shape {shape}")
                
                num_bands = shape[0]
                height = shape[1]
                width = shape[2]
                
                HyperspectralServerHandler.dataset = None
                HyperspectralServerHandler.npz_datacube = datacube
                HyperspectralServerHandler.hdr_path = file_path
                HyperspectralServerHandler.bands_cache.clear()
                
                wavelengths = []
                if 'wavelengths' in npz_data:
                    wavelengths = npz_data['wavelengths'].tolist()

                annotations = []
                if 'annotations_json' in npz_data:
                    try:
                        raw_annotations = npz_data['annotations_json']
                        if hasattr(raw_annotations, 'item'):
                            raw_annotations = raw_annotations.item()
                        annotations = json.loads(str(raw_annotations))
                    except Exception as e:
                        print(f"Failed to read embedded NPZ annotations: {e}")
                        annotations = []
                else:
                    sidecar_json_path = os.path.splitext(file_path)[0] + ".json"
                    if os.path.exists(sidecar_json_path):
                        try:
                            with open(sidecar_json_path, 'r', encoding='utf-8') as f:
                                annotations = json.load(f)
                        except Exception as e:
                            print(f"Failed to read NPZ sidecar annotations: {e}")
                            annotations = []

                source_roi = {}
                if 'source_roi_json' in npz_data:
                    try:
                        raw_source_roi = npz_data['source_roi_json']
                        if hasattr(raw_source_roi, 'item'):
                            raw_source_roi = raw_source_roi.item()
                        source_roi = json.loads(str(raw_source_roi))
                    except Exception as e:
                        print(f"Failed to read NPZ source ROI metadata: {e}")
                        source_roi = {}
                    
                if wavelengths:
                    warr = np.array(wavelengths)
                    r_idx = int(np.argmin(np.abs(warr - 640.0)))
                    g_idx = int(np.argmin(np.abs(warr - 550.0)))
                    b_idx = int(np.argmin(np.abs(warr - 460.0)))
                else:
                    r_idx = min(183, num_bands - 1)
                    g_idx = min(116, num_bands - 1)
                    b_idx = min(48, num_bands - 1)
                    
                resp = {
                    "width": width,
                    "height": height,
                    "bands": num_bands,
                    "wavelengths": wavelengths,
                    "default_rgb": [r_idx, g_idx, b_idx],
                    "is_npz": True,
                    "annotations": annotations,
                    "source_roi": source_roi
                }
                self._set_headers()
                self.wfile.write(json.dumps(resp).encode('utf-8'))
                print(f"Successfully opened NPZ dataset: {file_path} ({width}x{height}, {num_bands} bands)")
                return
            except Exception as e:
                self._set_headers(status=500)
                self.wfile.write(json.dumps({"error": f"Failed to open NPZ dataset: {str(e)}"}).encode('utf-8'))
                return

        # Resolve data path on server filesystem
        base_path, _ = os.path.splitext(file_path)
        possible_exts = ['.raw', '.dat', '.img', '']
        resolved_data_path = ""
        
        for ext in possible_exts:
            test_path = base_path + ext
            if os.path.exists(test_path):
                resolved_data_path = test_path
                break

        if not resolved_data_path:
            # Fallback if no matching data file is found in standard extensions
            self._set_headers(status=404)
            self.wfile.write(json.dumps({"error": f"Corresponding raw data file for {file_path} not found on server."}).encode('utf-8'))
            return

        try:
            # Load dataset
            dataset = gdal.Open(resolved_data_path, gdal.GA_ReadOnly)
            if dataset is None:
                raise Exception("GDAL could not open the dataset.")

            HyperspectralServerHandler.dataset = dataset
            HyperspectralServerHandler.npz_datacube = None
            HyperspectralServerHandler.hdr_path = file_path
            HyperspectralServerHandler.data_path = resolved_data_path
            HyperspectralServerHandler.bands_cache.clear()

            num_bands = dataset.RasterCount
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            wavelengths = parse_hdr_wavelengths(file_path)

            # Recommend RGB bands
            if wavelengths:
                warr = np.array(wavelengths)
                r_idx = int(np.argmin(np.abs(warr - 640.0)))
                g_idx = int(np.argmin(np.abs(warr - 550.0)))
                b_idx = int(np.argmin(np.abs(warr - 460.0)))
            else:
                r_idx = min(183, num_bands - 1)
                g_idx = min(116, num_bands - 1)
                b_idx = min(48, num_bands - 1)

            resp = {
                "width": width,
                "height": height,
                "bands": num_bands,
                "wavelengths": wavelengths,
                "default_rgb": [r_idx, g_idx, b_idx]
            }
            self._set_headers()
            self.wfile.write(json.dumps(resp).encode('utf-8'))
            print(f"Successfully opened dataset: {file_path} ({width}x{height}, {num_bands} bands)")

        except Exception as e:
            self._set_headers(status=500)
            self.wfile.write(json.dumps({"error": f"Failed to open dataset: {str(e)}"}).encode('utf-8'))

    def handle_rgb(self, data):
        if HyperspectralServerHandler.dataset is None and HyperspectralServerHandler.npz_datacube is None:
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": "No dataset opened yet"}).encode('utf-8'))
            return

        r_idx = data.get("r_band", 0)
        g_idx = data.get("g_band", 0)
        b_idx = data.get("b_band", 0)

        try:
            # Helper to retrieve band data from cache or file
            def get_band(band_idx):
                if band_idx in HyperspectralServerHandler.bands_cache:
                    return HyperspectralServerHandler.bands_cache[band_idx]
                if HyperspectralServerHandler.dataset is not None:
                    band = HyperspectralServerHandler.dataset.GetRasterBand(band_idx + 1)
                    band_data = band.ReadAsArray()
                else:
                    band_data = HyperspectralServerHandler.npz_datacube[band_idx, :, :]
                HyperspectralServerHandler.bands_cache[band_idx] = band_data
                return band_data

            r_band = get_band(r_idx)
            g_band = get_band(g_idx)
            b_band = get_band(b_idx)

            h, w = r_band.shape
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[..., 0] = stretch_band(r_band)
            rgb[..., 1] = stretch_band(g_band)
            rgb[..., 2] = stretch_band(b_band)

            # Convert to PIL Image and save as JPEG to send to the client
            from PIL import Image
            img = Image.fromarray(rgb)
            img_io = io.BytesIO()
            img.save(img_io, 'JPEG', quality=85)
            img_bytes = img_io.getvalue()

            self._set_headers(content_type="image/jpeg")
            self.wfile.write(img_bytes)

        except Exception as e:
            self._set_headers(status=500)
            self.wfile.write(json.dumps({"error": f"Failed to generate RGB image: {str(e)}"}).encode('utf-8'))

    def handle_pixel(self, data):
        if HyperspectralServerHandler.dataset is None and HyperspectralServerHandler.npz_datacube is None:
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": "No dataset opened yet"}).encode('utf-8'))
            return

        x = data.get("x")
        y = data.get("y")

        if x is None or y is None:
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": "x and y coordinates are required"}).encode('utf-8'))
            return

        if HyperspectralServerHandler.dataset is not None:
            num_bands = HyperspectralServerHandler.dataset.RasterCount
        else:
            num_bands = len(HyperspectralServerHandler.npz_datacube)
            
        spectrum = []

        try:
            for b in range(num_bands):
                if b in HyperspectralServerHandler.bands_cache:
                    spectrum.append(float(HyperspectralServerHandler.bands_cache[b][y, x]))
                elif HyperspectralServerHandler.dataset is not None:
                    band = HyperspectralServerHandler.dataset.GetRasterBand(b + 1)
                    val = band.ReadAsArray(x, y, 1, 1)[0, 0]
                    spectrum.append(float(val))
                else:
                    val = HyperspectralServerHandler.npz_datacube[b, y, x]
                    spectrum.append(float(val))

            self._set_headers()
            self.wfile.write(json.dumps({"spectrum": spectrum}).encode('utf-8'))

        except Exception as e:
            self._set_headers(status=500)
            self.wfile.write(json.dumps({"error": f"Failed to read pixel: {str(e)}"}).encode('utf-8'))

    def handle_save_roi(self, data):
        if HyperspectralServerHandler.dataset is None:
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": "No dataset opened yet"}).encode('utf-8'))
            return

        x = data.get("x")
        y = data.get("y")
        w = data.get("w")
        h = data.get("h")
        name = data.get("name")
        shape_type = data.get("shape_type", "Rectangle")
        angle = data.get("angle", 0.0)
        roi_type = data.get("type", "roi")
        all_rois = data.get("all_rois")

        if any(v is None for v in [x, y, w, h, name]):
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": "x, y, w, h, and name are required"}).encode('utf-8'))
            return

        try:
            dest_dir = os.path.dirname(HyperspectralServerHandler.hdr_path)
            base_hdr_name = os.path.splitext(os.path.basename(HyperspectralServerHandler.hdr_path))[0]
            dest_path = os.path.join(dest_dir, f"{base_hdr_name}_{name}.npz")
            local_json_path = os.path.join(dest_dir, f"{base_hdr_name}_{name}.json")

            # Load all annotations if not sent in request
            if all_rois is None:
                json_path = os.path.join(dest_dir, f"{base_hdr_name}.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            all_rois = json.load(f)
                    except Exception:
                        all_rois = []
                else:
                    all_rois = []

            theta = np.deg2rad(angle)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            if shape_type in ("Ellipse", "Circle"):
                cx_rot = x + w / 2.0
                cy_rot = y + h / 2.0
                local_to_global = np.array([
                    [cos_t, -sin_t, cx_rot - (w / 2.0) * cos_t + (h / 2.0) * sin_t],
                    [sin_t,  cos_t, cy_rot - (w / 2.0) * sin_t - (h / 2.0) * cos_t],
                    [0.0,    0.0,   1.0]
                ], dtype=np.float64)
            else:
                local_to_global = np.array([
                    [cos_t, -sin_t, float(x)],
                    [sin_t,  cos_t, float(y)],
                    [0.0,    0.0,   1.0]
                ], dtype=np.float64)

            global_to_local = np.linalg.inv(local_to_global)

            def transform_point(matrix, px, py):
                pt = matrix @ np.array([float(px), float(py), 1.0], dtype=np.float64)
                return float(pt[0]), float(pt[1])

            def annotation_center(item):
                tx = float(item["x"])
                ty = float(item["y"])
                tw = float(item["w"])
                th = float(item["h"])
                t_shape = item.get("shape_type", "Rectangle")
                t_angle = float(item.get("angle", 0.0))
                if t_shape in ("Ellipse", "Circle"):
                    return tx + tw / 2.0, ty + th / 2.0
                t_theta = np.deg2rad(t_angle)
                return (
                    tx + (tw / 2.0) * np.cos(t_theta) - (th / 2.0) * np.sin(t_theta),
                    ty + (tw / 2.0) * np.sin(t_theta) + (th / 2.0) * np.cos(t_theta)
                )

            def is_inside_export_roi(lx, ly):
                if not (0 <= lx < w and 0 <= ly < h):
                    return False
                if shape_type in ("Ellipse", "Circle"):
                    rx = w / 2.0
                    ry = h / 2.0
                    if rx <= 0 or ry <= 0:
                        return False
                    return (((lx - w / 2.0) ** 2) / (rx ** 2) + ((ly - h / 2.0) ** 2) / (ry ** 2)) <= 1.0
                return True

            source_roi = {
                "name": name,
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "shape_type": shape_type,
                "angle": float(angle),
                "type": roi_type
            }

            # Transform target annotations from original HDR coordinates into this NPZ crop coordinates.
            local_targets = []
            if roi_type == "roi" and all_rois:
                for item in all_rois:
                    if item.get("type", "roi") == "target":
                        tx = float(item["x"])
                        ty = float(item["y"])
                        tw = float(item["w"])
                        th = float(item["h"])
                        t_shape = item.get("shape_type", "Rectangle")
                        t_angle = float(item.get("angle", 0.0))

                        tcx, tcy = annotation_center(item)
                        lcx, lcy = transform_point(global_to_local, tcx, tcy)

                        if is_inside_export_roi(lcx, lcy):
                            local_angle = t_angle - angle
                            local_angle = (local_angle + 180) % 360 - 180

                            if t_shape in ("Ellipse", "Circle"):
                                local_x = lcx - tw / 2.0
                                local_y = lcy - th / 2.0
                            else:
                                local_x, local_y = transform_point(global_to_local, tx, ty)

                            local_item = {
                                "name": item["name"],
                                "x": float(local_x),
                                "y": float(local_y),
                                "w": float(tw),
                                "h": float(th),
                                "shape_type": t_shape,
                                "angle": float(local_angle),
                                "type": "target",
                                "source_x": float(tx),
                                "source_y": float(ty),
                                "source_angle": float(t_angle)
                            }
                            local_targets.append(local_item)

                try:
                    with open(local_json_path, 'w', encoding='utf-8') as f:
                        json.dump(local_targets, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    print(f"Failed to write local label file: {e}")

            elif roi_type == "target":
                if shape_type in ("Ellipse", "Circle"):
                    local_x = 0.0
                    local_y = 0.0
                    local_angle = 0.0
                else:
                    local_x = 0.0
                    local_y = 0.0
                    local_angle = 0.0
                local_targets.append({
                    "name": name,
                    "x": float(local_x),
                    "y": float(local_y),
                    "w": float(w),
                    "h": float(h),
                    "shape_type": shape_type,
                    "angle": float(local_angle),
                    "type": "target",
                    "source_x": float(x),
                    "source_y": float(y),
                    "source_angle": float(angle)
                })
                try:
                    with open(local_json_path, 'w', encoding='utf-8') as f:
                        json.dump(local_targets, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    print(f"Failed to write local label file: {e}")

            num_bands = HyperspectralServerHandler.dataset.RasterCount
            first_band = HyperspectralServerHandler.dataset.GetRasterBand(1)
            dtype = gdal.GetDataTypeName(first_band.DataType)

            np_dtype = np.uint16
            if "Byte" in dtype:
                np_dtype = np.uint8
            elif "Int16" in dtype:
                np_dtype = np.int16
            elif "UInt32" in dtype:
                np_dtype = np.uint32
            elif "Float32" in dtype:
                np_dtype = np.float32
            elif "Float64" in dtype:
                np_dtype = np.float64

            img_w = HyperspectralServerHandler.dataset.RasterXSize
            img_h = HyperspectralServerHandler.dataset.RasterYSize

            Y_grid, X_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

            if shape_type in ("Ellipse", "Circle"):
                cx_rot = x + w / 2.0
                cy_rot = y + h / 2.0
                X_orig = cx_rot + (X_grid - w / 2.0) * cos_t - (Y_grid - h / 2.0) * sin_t
                Y_orig = cy_rot + (X_grid - w / 2.0) * sin_t + (Y_grid - h / 2.0) * cos_t
            else:
                X_orig = x + X_grid * cos_t - Y_grid * sin_t
                Y_orig = y + X_grid * sin_t + Y_grid * cos_t

            min_x = int(np.floor(np.min(X_orig)))
            max_x = int(np.ceil(np.max(X_orig)))
            min_y = int(np.floor(np.min(Y_orig)))
            max_y = int(np.ceil(np.max(Y_orig)))

            min_x = max(0, min(img_w - 1, min_x))
            max_x = max(0, min(img_w - 1, max_x))
            min_y = max(0, min(img_h - 1, min_y))
            max_y = max(0, min(img_h - 1, max_y))

            crop_w = max_x - min_x + 1
            crop_h = max_y - min_y + 1

            roi_cube = np.zeros((num_bands, h, w), dtype=np_dtype)

            if crop_w > 0 and crop_h > 0:
                X_crop = X_orig - min_x
                Y_crop = Y_orig - min_y
                X_crop_nearest = np.clip(np.round(X_crop).astype(int), 0, crop_w - 1)
                Y_crop_nearest = np.clip(np.round(Y_crop).astype(int), 0, crop_h - 1)

                for b in range(num_bands):
                    band = HyperspectralServerHandler.dataset.GetRasterBand(b + 1)
                    band_data = band.ReadAsArray(min_x, min_y, crop_w, crop_h)
                    roi_cube[b, :, :] = band_data[Y_crop_nearest, X_crop_nearest]

            if shape_type in ("Ellipse", "Circle"):
                cy = (h - 1) / 2.0
                cx = (w - 1) / 2.0
                ry = h / 2.0
                rx = w / 2.0
                Y_mask, X_mask = np.ogrid[:h, :w]
                mask = ((Y_mask - cy) ** 2) / (ry ** 2) + ((X_mask - cx) ** 2) / (rx ** 2) > 1.0
                roi_cube[:, mask] = 0

            wavelengths = parse_hdr_wavelengths(HyperspectralServerHandler.hdr_path)
            np.savez_compressed(
                dest_path,
                datacube=roi_cube,
                roi_coords=np.array([x, y, w, h]),
                hdr_path=HyperspectralServerHandler.hdr_path,
                wavelengths=np.array(wavelengths),
                angle=angle,
                type=roi_type,
                name=name,
                shape_type=shape_type,
                source_roi_json=json.dumps(source_roi, ensure_ascii=False),
                annotations_json=json.dumps(local_targets, ensure_ascii=False),
                local_to_global_matrix=local_to_global,
                global_to_local_matrix=global_to_local
            )

            self._set_headers()
            self.wfile.write(json.dumps({
                "message": f"ROI saved to {dest_path}",
                "path": dest_path,
                "annotation_path": local_json_path,
                "annotation_count": len(local_targets),
                "shape": list(roi_cube.shape)
            }).encode('utf-8'))

        except Exception as e:
            self._set_headers(status=500)
            self.wfile.write(json.dumps({"error": f"Failed to save ROI: {str(e)}"}).encode('utf-8'))

    def handle_get_rois(self, data):
        hdr_path = data.get("hdr_path")
        if not hdr_path:
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": "hdr_path is required"}).encode('utf-8'))
            return
            
        json_path = os.path.splitext(hdr_path)[0] + ".json"
        if not os.path.exists(json_path):
            self._set_headers()
            self.wfile.write(json.dumps([]).encode('utf-8'))
            return
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                rois = json.load(f)
            self._set_headers()
            self.wfile.write(json.dumps(rois).encode('utf-8'))
        except Exception as e:
            self._set_headers(status=500)
            self.wfile.write(json.dumps({"error": f"Failed to read ROIs: {str(e)}"}).encode('utf-8'))

    def handle_save_rois(self, data):
        hdr_path = data.get("hdr_path")
        rois = data.get("rois")
        if not hdr_path or rois is None:
            self._set_headers(status=400)
            self.wfile.write(json.dumps({"error": "hdr_path and rois are required"}).encode('utf-8'))
            return
            
        json_path = os.path.splitext(hdr_path)[0] + ".json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(rois, f, indent=4, ensure_ascii=False)
            self._set_headers()
            self.wfile.write(json.dumps({"status": "ok", "message": f"ROIs saved to {json_path}"}).encode('utf-8'))
        except Exception as e:
            self._set_headers(status=500)
            self.wfile.write(json.dumps({"error": f"Failed to save ROIs: {str(e)}"}).encode('utf-8'))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperspectral Data Preview Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=17474, help="Port to run server on")
    parser.add_argument("--key", default="microplastic_secret", help="Secret key for authentication")
    args = parser.parse_args()

    HyperspectralServerHandler.api_key = args.key

    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, HyperspectralServerHandler)
    print(f"Starting server on {args.host}:{args.port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    main()
