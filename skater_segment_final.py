import sys
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

def segment_skater(image_path, output_path="silhouette.png", n_clusters=10):

    print(f"\nLoading: {image_path}")
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    h, w, _ = img_array.shape
    total_pixels = h * w

    # ── Step 1: KMeans ────────────────────────────────────────────────────
    pixels = img_array.reshape(-1, 3)
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    label_map = labels.reshape(h, w)

    # ── Step 2: Mark ICE clusters (bright + low saturation) ───────────────
    def ice_score(center):
        r, g, b = center / 255.0
        brightness = (r + g + b) / 3.0
        mx, mn = max(r, g, b), min(r, g, b)
        saturation = (mx - mn) / mx if mx > 0 else 0
        return brightness * (1 - saturation)

    print("\nCluster classification:")
    ice_clusters = set()
    for i, center in enumerate(centers):
        score = ice_score(center)
        size_ratio = float(np.sum(labels == i)) / total_pixels
        tag = "ICE" if score > 0.59 else "other"  # 0.59 ≈ brightness*sat threshold
        if score > 0.59:
            ice_clusters.add(i)
        print(f"  Cluster {i:2d}: ice_score={score:.3f}  size={size_ratio:.3f}  → {tag}")

    # Fallback: at least the brightest cluster is ice
    if not ice_clusters:
        best = max(range(n_clusters), key=lambda i: ice_score(centers[i]))
        ice_clusters.add(best)

    print(f"\nIce clusters: {ice_clusters}")

    # ── Step 3: Non-ice mask → connected components ───────────────────────
    non_ice = np.zeros((h, w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            if label_map[row, col] not in ice_clusters:
                non_ice[row, col] = 255

    min_size = max(300, int(total_pixels * 0.002))
    labeled, num = ndimage.label(non_ice)

    # ── Step 4: Score each component → pick skater ────────────────────────
    # Skater = high color variance + relatively tall + large
    print(f"\nComponent scoring (min_size={min_size}):")
    candidates = []
    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        size = int(np.sum(comp))
        if size < min_size:
            continue

        rows_idx = np.where(np.any(comp, axis=1))[0]
        cols_idx = np.where(np.any(comp, axis=0))[0]
        bbox_h = rows_idx[-1] - rows_idx[0] + 1
        bbox_w = cols_idx[-1] - cols_idx[0] + 1
        aspect_ratio = bbox_w / max(bbox_h, 1)
        solidity = size / max(bbox_h * bbox_w, 1)

        comp_pixels = img_array[comp].astype(np.float32)
        color_var = float(np.mean(np.var(comp_pixels, axis=0)))

        # Skater score: colorful + tall + irregular
        score = color_var * (1 / (aspect_ratio + 0.1)) * (1 - solidity * 0.3) * np.log(size + 1)

        print(f"  Comp {comp_id:3d}: size={size:6d}  ar={aspect_ratio:.2f}  "
              f"sol={solidity:.2f}  colorvar={color_var:6.1f}  score={score:.1f}")

        candidates.append({
            'id': comp_id, 'score': score, 'size': size,
            'center_r': (rows_idx[0] + rows_idx[-1]) / 2,
            'center_c': (cols_idx[0] + cols_idx[-1]) / 2,
            'diag': np.sqrt(bbox_h**2 + bbox_w**2),
        })

    if not candidates:
        print("No candidates found.")
        Image.fromarray(np.ones((h, w), dtype=np.uint8) * 255, "L").save(output_path)
        return

    candidates.sort(key=lambda x: -x['score'])
    best = candidates[0]
    print(f"\nBest skater component: {best['id']} (score={best['score']:.1f})")

    # Include nearby smaller components (detached skate blades etc.)
    selected = {best['id']}
    for c in candidates[1:]:
        dist = np.sqrt((c['center_r'] - best['center_r'])**2 +
                       (c['center_c'] - best['center_c'])**2)
        if dist < best['diag'] * 0.85:
            selected.add(c['id'])

    # ── Step 5: Build output — WHITE background, BLACK skater ─────────────
    output = np.ones((h, w), dtype=np.uint8) * 255
    for comp_id in selected:
        output[labeled == comp_id] = 0

    # ── Step 6: Remove full-width horizontal line artifacts ───────────────
    # A human body cannot produce a perfectly straight line spanning the
    # full width of the image. Any row where black pixels form a single
    # continuous run covering > 75% of image width is an artifact.
    cleaned = output.copy()
    removed_rows = 0
    artifact_seed_pixels = []

    for row_idx in range(h):
        row = output[row_idx]
        black_pixels = np.where(row == 0)[0]
        if len(black_pixels) == 0:
            continue
        run_length = black_pixels[-1] - black_pixels[0] + 1
        is_continuous = (run_length == len(black_pixels))
        coverage = run_length / w
        if is_continuous and coverage > 0.75:
            for col_idx in black_pixels:
                artifact_seed_pixels.append((row_idx, int(col_idx)))
            cleaned[row_idx] = 255
            removed_rows += 1

    if removed_rows > 0:
        print(f"Removed {removed_rows} horizontal artifact line(s).")

    # ── Region growing from artifact pixels ───────────────────────────────
    # Flood-fill outward from removed artifact pixels: any adjacent black
    # pixel whose RGB color is close to the seed is also an artifact → white.
    if artifact_seed_pixels:
        color_threshold = 40.0
        visited = (cleaned == 255)  # already-white = already visited
        queue = list(artifact_seed_pixels)
        for (r, c) in queue:
            visited[r, c] = True

        grown = 0
        while queue:
            r, c = queue.pop()
            seed_color = img_array[r, c]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if cleaned[nr, nc] == 0:
                        dist = float(np.linalg.norm(seed_color - img_array[nr, nc]))
                        if dist < color_threshold:
                            cleaned[nr, nc] = 255
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                            grown += 1

        print(f"Region growing removed {grown} additional artifact pixel(s).")

    output = cleaned

    # ── Step 7: ICE flood fill (fixed color reference) ────────────────────
    # Grow ice from border inward — but compare each candidate pixel against
    # the MEAN ice cluster color (not the propagating seed) to prevent drift
    # into skin/costume colors.
    print("Running ice flood fill...")
    ice_pixels = pixels[np.isin(labels, list(ice_clusters))]
    mean_ice_color = ice_pixels.mean(axis=0)  # fixed reference color

    ice_visited = np.zeros((h, w), dtype=bool)
    ice_queue = []
    for c in range(w):
        for r in [0, h - 1]:
            if output[r, c] == 255 and not ice_visited[r, c]:
                ice_visited[r, c] = True
                ice_queue.append((r, c))
    for r in range(h):
        for c in [0, w - 1]:
            if output[r, c] == 255 and not ice_visited[r, c]:
                ice_visited[r, c] = True
                ice_queue.append((r, c))

    ice_grown = 0
    ice_color_threshold = 25.0
    while ice_queue:
        r, c = ice_queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not ice_visited[nr, nc]:
                ice_visited[nr, nc] = True
                neighbor = img_array[nr, nc]
                # Always compare against fixed mean ice color, not seed
                dist = float(np.linalg.norm(neighbor - mean_ice_color))
                if output[nr, nc] == 255:
                    ice_queue.append((nr, nc))  # already ice, keep traversing
                elif dist < ice_color_threshold:
                    output[nr, nc] = 255
                    ice_grown += 1
                    ice_queue.append((nr, nc))
    print(f"  Ice flood fill reclaimed {ice_grown} pixel(s).")

    # ── Step 8: SKATER flood fill (fixed color reference) ─────────────────
    # Grow skater outward from its core — compare against mean skater color.
    print("Running skater flood fill...")
    skater_pixels_arr = np.argwhere(output == 0)
    skater_grown = 0
    if len(skater_pixels_arr) > 0:
        skater_rgb = img_array[output == 0].reshape(-1, 3)
        mean_skater_color = skater_rgb.mean(axis=0)

        sr_min, sc_min = skater_pixels_arr.min(axis=0)
        sr_max, sc_max = skater_pixels_arr.max(axis=0)
        r_pad = max(1, (sr_max - sr_min) // 4)
        c_pad = max(1, (sc_max - sc_min) // 4)
        inner_mask = (
            (skater_pixels_arr[:, 0] >= sr_min + r_pad) &
            (skater_pixels_arr[:, 0] <= sr_max - r_pad) &
            (skater_pixels_arr[:, 1] >= sc_min + c_pad) &
            (skater_pixels_arr[:, 1] <= sc_max - c_pad)
        )
        seeds = skater_pixels_arr[inner_mask][::max(1, inner_mask.sum() // 500)]

        sk_visited = (output == 255).copy()
        sk_queue = []
        for r, c in seeds:
            if not sk_visited[r, c]:
                sk_visited[r, c] = True
                sk_queue.append((int(r), int(c)))

        skater_color_threshold = 30.0
        while sk_queue:
            r, c = sk_queue.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not sk_visited[nr, nc]:
                    sk_visited[nr, nc] = True
                    if output[nr, nc] == 255:
                        neighbor = img_array[nr, nc]
                        dist = float(np.linalg.norm(neighbor - mean_skater_color))
                        if dist < skater_color_threshold:
                            output[nr, nc] = 0
                            skater_grown += 1
                            sk_queue.append((nr, nc))
    print(f"  Skater flood fill reclaimed {skater_grown} pixel(s).")

    # ── Step 9: Fill small holes inside skater ────────────────────────────
    # White islands completely enclosed by skater pixels (stones on costume,
    # skin patches, etc.) → fill them black.
    print("Filling holes inside skater...")
    inverted = (output == 255).astype(np.uint8)
    labeled_inv, num_inv = ndimage.label(inverted)

    border_ids = set()
    for c in range(w):
        border_ids.add(int(labeled_inv[0, c]))
        border_ids.add(int(labeled_inv[h-1, c]))
    for r in range(h):
        border_ids.add(int(labeled_inv[r, 0]))
        border_ids.add(int(labeled_inv[r, w-1]))

    max_hole_size = 150  # max ~12x12 px — only small costume stones/dots
    holes_filled = 0
    for comp_id in range(1, num_inv + 1):
        if comp_id in border_ids:
            continue
        size = int(np.sum(labeled_inv == comp_id))
        if size <= max_hole_size:
            output[labeled_inv == comp_id] = 0
            holes_filled += size
    print(f"  Filled {holes_filled} hole pixel(s) inside skater.")

    Image.fromarray(output, "L").save(output_path)
    print(f"Saved: {output_path}")

    # Comparison image
    comp_path = output_path.replace(".png", "_comparison.png")
    comparison = Image.new("RGB", (w * 2, h))
    comparison.paste(img, (0, 0))
    comparison.paste(Image.fromarray(output).convert("RGB"), (w, 0))
    comparison.save(comp_path)
    print(f"Comparison: {comp_path}")

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python skater_segment_v4.py <image> [output] [n_clusters]")
        sys.exit(1)

    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "silhouette.png"
    n_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    segment_skater(image_path, output_path, n_clusters)
