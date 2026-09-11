import numpy as np
import cv2

def extract_wafer_features(wafer_map):
    """
    Extracts domain-specific hand-crafted geometric and spatial features
    from a single 2D wafer map.
    
    Wafer Map encoding:
      0: Background (outside wafer)
      1: Normal die (good)
      2: Defective die (fail)
      
    Returns:
      dict of feature_name -> float value (~28 features)
    """
    features = {}
    
    # Masks
    valid_mask = (wafer_map > 0)
    defect_mask = (wafer_map == 2).astype(np.uint8)
    
    total_dies = int(np.sum(valid_mask))
    total_defects = int(np.sum(defect_mask))
    
    if total_dies == 0:
        return {f"f_{i}": 0.0 for i in range(28)}
    
    # 1. Defect Rates
    features['defect_ratio'] = float(total_defects) / float(total_dies)
    features['total_defects'] = float(total_defects)
    
    # Wafer geometric center & radius
    y_indices, x_indices = np.where(valid_mask)
    center_y = float(np.mean(y_indices))
    center_x = float(np.mean(x_indices))
    
    distances_all = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    radius = float(np.max(distances_all)) if len(distances_all) > 0 else 1.0
    if radius == 0:
        radius = 1.0

    # 2. Defect Centroid Offset from Center
    if total_defects > 0:
        def_y, def_x = np.where(defect_mask > 0)
        def_cy = float(np.mean(def_y))
        def_cx = float(np.mean(def_x))
        centroid_dist = np.sqrt((def_cx - center_x)**2 + (def_cy - center_y)**2)
        features['defect_centroid_offset_norm'] = float(centroid_dist) / radius
    else:
        features['defect_centroid_offset_norm'] = 0.0

    # 3. Radial Concentric Ring Densities
    # Inner (Center): 0 to 0.33R, Middle (Donut): 0.33R to 0.67R, Outer (Edge-Ring): 0.67R to 1.0R
    r1_dies = np.sum(distances_all <= 0.33 * radius)
    r2_dies = np.sum((distances_all > 0.33 * radius) & (distances_all <= 0.67 * radius))
    r3_dies = np.sum(distances_all > 0.67 * radius)
    
    if total_defects > 0:
        def_dists = np.sqrt((def_x - center_x)**2 + (def_y - center_y)**2)
        r1_defs = np.sum(def_dists <= 0.33 * radius)
        r2_defs = np.sum((def_dists > 0.33 * radius) & (def_dists <= 0.67 * radius))
        r3_defs = np.sum(def_dists > 0.67 * radius)
    else:
        r1_defs = r2_defs = r3_defs = 0

    features['radial_density_inner'] = float(r1_defs) / float(r1_dies) if r1_dies > 0 else 0.0
    features['radial_density_middle'] = float(r2_defs) / float(r2_dies) if r2_dies > 0 else 0.0
    features['radial_density_outer'] = float(r3_defs) / float(r3_dies) if r3_dies > 0 else 0.0
    
    features['radial_inner_to_outer_ratio'] = features['radial_density_inner'] / (features['radial_density_outer'] + 1e-6)
    features['radial_middle_to_inner_ratio'] = features['radial_density_middle'] / (features['radial_density_inner'] + 1e-6)

    # 4. Angular Sector Densities (8 Octants) - To detect localized defects (Loc, Edge-Loc)
    angles_all = np.arctan2(y_indices - center_y, x_indices - center_x) % (2 * np.pi)
    if total_defects > 0:
        angles_defs = np.arctan2(def_y - center_y, def_x - center_x) % (2 * np.pi)
    else:
        angles_defs = np.array([])

    octant_densities = []
    num_octants = 8
    octant_step = 2 * np.pi / num_octants
    for o in range(num_octants):
        start_a = o * octant_step
        end_a = (o + 1) * octant_step
        dies_in_oct = np.sum((angles_all >= start_a) & (angles_all < end_a))
        defs_in_oct = np.sum((angles_defs >= start_a) & (angles_defs < end_a)) if len(angles_defs) > 0 else 0
        density = float(defs_in_oct) / float(dies_in_oct) if dies_in_oct > 0 else 0.0
        octant_densities.append(density)
        features[f'angular_density_octant_{o}'] = density
        
    features['angular_density_max'] = float(np.max(octant_densities))
    features['angular_density_std'] = float(np.std(octant_densities))

    # 5. Hu Moments (7 Scale/Rotation/Translation Invariant Moments)
    moments = cv2.moments(defect_mask)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(7):
        val = hu[i]
        if val != 0:
            log_hu = -1.0 * np.copysign(1.0, val) * np.log10(np.abs(val) + 1e-12)
        else:
            log_hu = 0.0
        features[f'hu_moment_{i+1}'] = float(log_hu)

    # 6. Connected Components & Scratch Features
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(defect_mask, connectivity=8)
    features['num_defect_clusters'] = float(num_labels - 1)
    
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        widths = stats[1:, cv2.CC_STAT_WIDTH]
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        
        max_idx = int(np.argmax(areas))
        max_cluster_area = float(areas[max_idx])
        
        features['max_cluster_area_ratio'] = max_cluster_area / float(total_dies)
        features['max_cluster_fraction_of_defects'] = max_cluster_area / float(total_defects)
        
        w = float(widths[max_idx])
        h = float(heights[max_idx])
        features['max_cluster_aspect_ratio'] = max(w, h) / (min(w, h) + 1e-6)
    else:
        features['max_cluster_area_ratio'] = 0.0
        features['max_cluster_fraction_of_defects'] = 0.0
        features['max_cluster_aspect_ratio'] = 1.0

    return features
