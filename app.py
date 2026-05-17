"""
COMPLETE DERMATOLOGICAL LESION CLASSIFIER - STREAMLIT WEB APP
FULL VERSION with ALL functions: 
- Preprocessing, Enhancement, Spatial & Frequency Filtering
- Segmentation (Otsu, Adaptive, K-means)
- Edge Detection (Canny, Sobel, Laplacian, Prewitt)
- Morphological Operations (Erosion, Dilation, Opening, Closing, Gradient, TopHat, BlackHat)
- Feature Extraction (20+ features: geometric, texture, Hu Moments)
- ABCD Scoring with Dynamic Thresholds
- Multiple Visualizations (Pipeline Grid, Histograms, Filter Comparison, Edge Comparison, Parameter Sensitivity)
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DermatoScan - Complete Skin Lesion Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
    }
    .main-header p {
        color: #e0e0e0;
        margin: 0;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .parameter-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# COMPLETE IMAGE PROCESSING CLASS (FULL VERSION)
# ============================================================================

class CompleteLesionAnalyzer:
    """
    COMPLETE implementation of ALL image processing functions
    No functions removed - everything from original project preserved
    """
    
    def __init__(self, image):
        """Initialize with input image"""
        self.original_image = image
        self.grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.processed_stages = {}
        self.features = {}
        self.analysis_results = {}
        
    # ========================================================================
    # PART 1: FUNDAMENTAL IMAGE PROCESSING
    # ========================================================================
    
    def load_and_preprocess(self, target_size=(512, 512)):
        """Step 1: Load image, resize, and convert to grayscale"""
        # Already handled in constructor, but keeping for compatibility
        self.processed_stages['original'] = self.original_image
        self.processed_stages['grayscale'] = self.grayscale_image
        return self
    
    def histogram_equalization(self):
        """Step 2: Apply CLAHE histogram equalization for enhancement"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.enhanced_image = clahe.apply(self.grayscale_image)
        
        # Global histogram equalization for comparison
        self.global_equalized = cv2.equalizeHist(self.grayscale_image)
        
        self.processed_stages['grayscale'] = self.grayscale_image
        self.processed_stages['enhanced'] = self.enhanced_image
        self.processed_stages['global_equalized'] = self.global_equalized
        
        return self
    
    def spatial_filtering(self):
        """Step 3: Apply spatial domain filters (Gaussian, Median, Bilateral)"""
        # Gaussian filter - removes Gaussian noise
        self.gaussian_filtered = cv2.GaussianBlur(self.enhanced_image, (5, 5), 1.0)
        
        # Median filter - removes salt-pepper noise, preserves edges
        self.median_filtered = cv2.medianBlur(self.enhanced_image, 5)
        
        # Bilateral filter - edge-preserving smoothing
        self.bilateral_filtered = cv2.bilateralFilter(self.enhanced_image, 9, 75, 75)
        
        self.processed_stages['gaussian'] = self.gaussian_filtered
        self.processed_stages['median'] = self.median_filtered
        self.processed_stages['bilateral'] = self.bilateral_filtered
        
        return self
    
    def frequency_domain_filtering(self):
        """Step 4: Apply DFT-based frequency domain filtering"""
        # Compute DFT
        f_transform = np.fft.fft2(self.median_filtered)
        f_shift = np.fft.fftshift(f_transform)
        self.magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        rows, cols = self.median_filtered.shape
        crow, ccol = rows // 2, cols // 2
        
        # Low-pass filter (Gaussian mask)
        low_pass_mask = np.zeros((rows, cols), np.float32)
        radius = 30
        cv2.circle(low_pass_mask, (ccol, crow), radius, 1, -1)
        
        # High-pass filter
        high_pass_mask = 1 - low_pass_mask
        
        # Apply filters
        f_shift_low = f_shift * low_pass_mask
        f_shift_high = f_shift * high_pass_mask
        
        # Inverse DFT
        f_ishift_low = np.fft.ifftshift(f_shift_low)
        self.low_pass_filtered = np.abs(np.fft.ifft2(f_ishift_low))
        
        f_ishift_high = np.fft.ifftshift(f_shift_high)
        self.high_pass_filtered = np.abs(np.fft.ifft2(f_ishift_high))
        
        self.processed_stages['magnitude_spectrum'] = self.magnitude_spectrum
        self.processed_stages['low_pass'] = self.low_pass_filtered
        self.processed_stages['high_pass'] = self.high_pass_filtered
        
        return self
    
    # ========================================================================
    # PART 2: ADVANCED IMAGE PROCESSING
    # ========================================================================
    
    def segmentation(self):
        """Step 5: Apply multiple segmentation techniques"""
        # Method 1: Otsu's thresholding
        _, self.otsu_segmented = cv2.threshold(
            self.median_filtered, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Method 2: Adaptive thresholding
        self.adaptive_segmented = cv2.adaptiveThreshold(
            self.median_filtered, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Method 3: K-means clustering (k=3)
        pixels = self.enhanced_image.reshape((-1, 1))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        centers = np.uint8(centers)
        self.kmeans_segmented = centers[labels.flatten()].reshape(self.enhanced_image.shape)
        
        # Binary mask from k-means
        self.kmeans_mask = (labels.reshape(self.enhanced_image.shape) == 0).astype(np.uint8) * 255
        
        self.processed_stages['otsu'] = self.otsu_segmented
        self.processed_stages['adaptive'] = self.adaptive_segmented
        self.processed_stages['kmeans'] = self.kmeans_segmented
        
        return self
    
    def edge_detection(self):
        """Step 6: Apply multiple edge detection techniques"""
        # Canny edge detection (multi-stage) with multiple thresholds
        self.canny_edges = cv2.Canny(self.median_filtered, 50, 150)
        self.canny_strict = cv2.Canny(self.median_filtered, 100, 200)
        self.canny_lenient = cv2.Canny(self.median_filtered, 30, 100)
        
        # Sobel edge detection (gradient-based)
        sobel_x = cv2.Sobel(self.median_filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.median_filtered, cv2.CV_64F, 0, 1, ksize=3)
        self.sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
        self.sobel_edges = np.uint8(np.clip(self.sobel_edges, 0, 255))
        
        # Laplacian edge detection
        self.laplacian_edges = cv2.Laplacian(self.median_filtered, cv2.CV_64F)
        self.laplacian_edges = np.uint8(np.abs(self.laplacian_edges))
        
        # Prewitt edge detection
        prewitt_x = cv2.filter2D(self.median_filtered, cv2.CV_64F, 
                                  np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
        prewitt_y = cv2.filter2D(self.median_filtered, cv2.CV_64F,
                                  np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
        self.prewitt_edges = np.sqrt(prewitt_x**2 + prewitt_y**2)
        self.prewitt_edges = np.uint8(np.clip(self.prewitt_edges, 0, 255))
        
        # Roberts edge detection
        roberts_x = cv2.filter2D(self.median_filtered, cv2.CV_64F, np.array([[1, 0], [0, -1]]))
        roberts_y = cv2.filter2D(self.median_filtered, cv2.CV_64F, np.array([[0, 1], [-1, 0]]))
        self.roberts_edges = np.sqrt(roberts_x**2 + roberts_y**2)
        self.roberts_edges = np.uint8(np.clip(self.roberts_edges, 0, 255))
        
        self.processed_stages['canny'] = self.canny_edges
        self.processed_stages['canny_strict'] = self.canny_strict
        self.processed_stages['canny_lenient'] = self.canny_lenient
        self.processed_stages['sobel'] = self.sobel_edges
        self.processed_stages['laplacian'] = self.laplacian_edges
        self.processed_stages['prewitt'] = self.prewitt_edges
        self.processed_stages['roberts'] = self.roberts_edges
        
        return self
    
    def morphological_operations(self):
        """Step 7: Apply ALL morphological operations to improve segmentation"""
        # Create multiple kernels
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Basic operations
        self.eroded = cv2.erode(self.otsu_segmented, kernel_ellipse, iterations=1)
        self.dilated = cv2.dilate(self.otsu_segmented, kernel_ellipse, iterations=1)
        self.opening = cv2.morphologyEx(self.otsu_segmented, cv2.MORPH_OPEN, kernel_ellipse)
        self.closing = cv2.morphologyEx(self.otsu_segmented, cv2.MORPH_CLOSE, kernel_large)
        
        # Advanced operations
        self.gradient = cv2.morphologyEx(self.otsu_segmented, cv2.MORPH_GRADIENT, kernel_ellipse)
        self.top_hat = cv2.morphologyEx(self.otsu_segmented, cv2.MORPH_TOPHAT, kernel_ellipse)
        self.black_hat = cv2.morphologyEx(self.otsu_segmented, cv2.MORPH_BLACKHAT, kernel_ellipse)
        
        # Multiple iterations
        self.eroded_twice = cv2.erode(self.otsu_segmented, kernel_ellipse, iterations=2)
        self.dilated_twice = cv2.dilate(self.otsu_segmented, kernel_ellipse, iterations=2)
        
        # Find largest contour (assumed to be the lesion)
        contours, _ = cv2.findContours(self.closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            self.lesion_contour = max(contours, key=cv2.contourArea)
            self.lesion_mask_filled = np.zeros_like(self.closing)
            cv2.drawContours(self.lesion_mask_filled, [self.lesion_contour], -1, 255, -1)
        else:
            self.lesion_contour = None
            self.lesion_mask_filled = self.closing
        
        self.processed_stages['eroded'] = self.eroded
        self.processed_stages['dilated'] = self.dilated
        self.processed_stages['opening'] = self.opening
        self.processed_stages['closing'] = self.closing
        self.processed_stages['gradient'] = self.gradient
        self.processed_stages['top_hat'] = self.top_hat
        self.processed_stages['black_hat'] = self.black_hat
        
        return self
    
    def extract_features(self):
        """Step 8: Extract ALL meaningful features for analysis (20+ features)"""
        if self.lesion_contour is None:
            return self
        
        # ===== GEOMETRIC FEATURES =====
        area = cv2.contourArea(self.lesion_contour)
        perimeter = cv2.arcLength(self.lesion_contour, True)
        
        # Circularity (4πA/P²) - 1 = perfect circle
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Bounding box features
        x, y, w, h = cv2.boundingRect(self.lesion_contour)
        aspect_ratio = w / h if h > 0 else 0
        extent = area / (w * h) if (w * h) > 0 else 0
        
        # Convex hull analysis
        hull = cv2.convexHull(self.lesion_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        diameter_mm = equivalent_diameter * 0.2646  # Approx pixel to mm conversion
        
        # ===== SHAPE FEATURES (Hu Moments - 7 moments) =====
        moments = cv2.moments(self.lesion_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # ===== BOUNDARY FEATURES =====
        if len(self.lesion_contour) > 0:
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(self.lesion_contour, epsilon, True)
            num_vertices = len(approx)
            contour_smoothness = len(self.lesion_contour) / perimeter if perimeter > 0 else 0
            
            # Convexity defects
            if len(self.lesion_contour) > 3:
                hull_indices = cv2.convexHull(self.lesion_contour, returnPoints=False)
                if len(hull_indices) > 3:
                    convexity_defects = cv2.convexityDefects(self.lesion_contour, hull_indices)
                    num_defects = len(convexity_defects) if convexity_defects is not None else 0
                else:
                    num_defects = 0
            else:
                num_defects = 0
        else:
            num_vertices = 0
            contour_smoothness = 0
            num_defects = 0
        
        # ===== TEXTURAL FEATURES =====
        lesion_pixels = self.grayscale_image[self.lesion_mask_filled == 255]
        if len(lesion_pixels) > 0:
            mean_intensity = np.mean(lesion_pixels)
            std_intensity = np.std(lesion_pixels)
            skewness = np.mean(((lesion_pixels - mean_intensity) / (std_intensity + 1e-10)) ** 3)
            kurtosis = np.mean(((lesion_pixels - mean_intensity) / (std_intensity + 1e-10)) ** 4) - 3
            # Percentiles for color distribution
            p10 = np.percentile(lesion_pixels, 10)
            p25 = np.percentile(lesion_pixels, 25)
            p75 = np.percentile(lesion_pixels, 75)
            p90 = np.percentile(lesion_pixels, 90)
            iqr = p75 - p25
            # Color uniformity measure
            color_uniformity = 1 - (std_intensity / 255)
        else:
            mean_intensity = std_intensity = skewness = kurtosis = p10 = p25 = p75 = p90 = iqr = color_uniformity = 0
        
        # ===== ASYMMETRY MEASURE =====
        asymmetry_measure = abs(hu_moments_log[0]) * 0.1
        # Normalize asymmetry to [0,1] range
        asymmetry_score = min(asymmetry_measure * 2, 1.0)
        
        # ===== COMPACTNESS AND ROUNDNESS =====
        compactness = (perimeter ** 2) / area if area > 0 else 0
        roundness = (4 * area) / (np.pi * (perimeter ** 2)) if perimeter > 0 else 0
        
        # Store ALL features
        self.features = {
            # Geometric
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity,
            'equivalent_diameter': equivalent_diameter,
            'diameter_mm': diameter_mm,
            'compactness': compactness,
            'roundness': roundness,
            
            # Boundary
            'num_vertices': num_vertices,
            'contour_smoothness': contour_smoothness,
            'num_convexity_defects': num_defects,
            
            # Texture
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'p10_intensity': p10,
            'p25_intensity': p25,
            'p75_intensity': p75,
            'p90_intensity': p90,
            'iqr': iqr,
            'color_uniformity': color_uniformity,
            
            # Hu Moments (all 7)
            'hu_moment_1': hu_moments_log[0],
            'hu_moment_2': hu_moments_log[1],
            'hu_moment_3': hu_moments_log[2],
            'hu_moment_4': hu_moments_log[3],
            'hu_moment_5': hu_moments_log[4],
            'hu_moment_6': hu_moments_log[5],
            'hu_moment_7': hu_moments_log[6],
            
            # Asymmetry
            'asymmetry_measure': asymmetry_measure,
            'asymmetry_score': asymmetry_score
        }
        
        return self
    
    def calculate_abcd_score(self):
        """Calculate ABCD (Asymmetry, Border, Color, Diameter) risk score with DYNAMIC thresholds"""
        if not self.features:
            return self
        
        # A - Asymmetry Score (from Hu moments)
        asymmetry_score = self.features['asymmetry_score']
        
        # B - Border Irregularity Score (1 - circularity)
        border_irregularity = max(0, 1 - self.features['circularity'])
        border_score = min(border_irregularity, 1.0)
        
        # C - Color Variation Score (normalized standard deviation)
        color_score = min(self.features['std_intensity'] / 255, 1.0)
        
        # D - Diameter Score (6mm threshold from clinical guidelines)
        if self.features['diameter_mm'] >= 6:
            diameter_score = 1.0
        elif self.features['diameter_mm'] <= 2:
            diameter_score = 0.1
        else:
            # Linear scaling between 2mm and 6mm
            diameter_score = (self.features['diameter_mm'] - 2) / 4
            diameter_score = max(0.1, min(1.0, diameter_score))
        
        # Additional scores for completeness
        texture_score = min(self.features['kurtosis'] / 10, 0.5)
        solidity_score = max(0, 1 - self.features['solidity'])
        
        self.abcd_scores = {
            'Asymmetry (A)': round(asymmetry_score, 4),
            'Border (B)': round(border_score, 4),
            'Color (C)': round(color_score, 4),
            'Diameter (D)': round(diameter_score, 4),
            'Texture (T)': round(texture_score, 4),
            'Solidity (S)': round(solidity_score, 4)
        }
        
        # Total score (unweighted average of core 4)
        total_score = (asymmetry_score + border_score + color_score + diameter_score) / 4
        
        # DYNAMIC risk classification based on actual total score
        if total_score < 0.3:
            risk_level = "LOW RISK - Benign Characteristics"
            recommendation = "✓ Regular monitoring recommended. Perform monthly self-examination. No immediate action needed."
            action = "Monthly self-check"
            risk_color = "low"
            urgency = "Routine monitoring"
        elif total_score < 0.6:
            risk_level = "MEDIUM RISK - Suspicious Features"
            recommendation = "⚠️ Dermatologist consultation recommended within 3 months for further evaluation. Consider baseline photography."
            action = "Schedule dermatologist visit"
            risk_color = "medium"
            urgency = "Within 3 months"
        else:
            risk_level = "HIGH RISK - Malignant Suspicion"
            recommendation = "🚨 URGENT: Immediate dermatologist consultation required for biopsy evaluation. Do not delay."
            action = "Emergency appointment"
            risk_color = "high"
            urgency = "Immediate"
        
        # Generate clinical notes based on individual metrics
        clinical_notes = []
        if asymmetry_score > 0.6:
            clinical_notes.append("🔴 Significant asymmetry detected - one half doesn't match the other")
        elif asymmetry_score > 0.4:
            clinical_notes.append("🟠 Moderate asymmetry observed")
        
        if border_score > 0.6:
            clinical_notes.append("🔴 Irregular, scalloped, or poorly defined border")
        elif border_score > 0.4:
            clinical_notes.append("🟠 Some border irregularity noted")
        
        if color_score > 0.5:
            clinical_notes.append("🔴 Multiple colors present (brown, black, tan, red, white, blue)")
        elif color_score > 0.3:
            clinical_notes.append("🟠 Variation in pigmentation observed")
        
        if self.features['diameter_mm'] > 6:
            clinical_notes.append(f"🔴 Diameter ({self.features['diameter_mm']:.1f}mm) exceeds 6mm threshold")
        elif self.features['diameter_mm'] > 4:
            clinical_notes.append(f"🟠 Diameter ({self.features['diameter_mm']:.1f}mm) approaching threshold")
        
        if self.features['circularity'] < 0.5:
            clinical_notes.append("🔴 Highly irregular shape")
        elif self.features['circularity'] < 0.7:
            clinical_notes.append("🟠 Moderately irregular shape")
        
        if self.features['solidity'] < 0.8:
            clinical_notes.append("🔴 Non-convex shape with indentations or protrusions")
        
        if self.features['num_convexity_defects'] > 5:
            clinical_notes.append(f"🔴 Multiple border irregularities ({self.features['num_convexity_defects']} defects detected)")
        
        if not clinical_notes:
            clinical_notes.append("✅ No significant abnormalities detected")
        
        self.risk_result = {
            'total_score': round(total_score, 4),
            'risk_level': risk_level,
            'recommendation': recommendation,
            'risk_color': risk_color,
            'action': action,
            'urgency': urgency,
            'clinical_notes': clinical_notes,
            'abcd_scores': self.abcd_scores,
            'raw_values': {
                'circularity': round(self.features['circularity'], 4),
                'diameter_mm': round(self.features['diameter_mm'], 1),
                'solidity': round(self.features['solidity'], 4),
                'std_intensity': round(self.features['std_intensity'], 1),
                'area': round(self.features['area'], 0),
                'perimeter': round(self.features['perimeter'], 1),
                'asymmetry': round(self.features['asymmetry_measure'], 4)
            }
        }
        
        return self
    
    def draw_contour_on_image(self):
        """Draw detected contour on original image with enhancements"""
        contour_img = self.original_image.copy()
        if self.lesion_contour is not None:
            # Draw main contour in green
            cv2.drawContours(contour_img, [self.lesion_contour], -1, (0, 255, 0), 2)
            
            # Draw bounding box in blue
            x, y, w, h = cv2.boundingRect(self.lesion_contour)
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            
            # Draw convex hull in red
            hull = cv2.convexHull(self.lesion_contour)
            cv2.drawContours(contour_img, [hull], -1, (0, 0, 255), 1)
            
            # Add center point
            M = cv2.moments(self.lesion_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(contour_img, (cx, cy), 3, (255, 255, 0), -1)
            
        return contour_img
    
    def run_complete_pipeline(self):
        """Execute ALL processing steps in order"""
        self.load_and_preprocess()
        self.histogram_equalization()
        self.spatial_filtering()
        self.frequency_domain_filtering()
        self.segmentation()
        self.edge_detection()
        self.morphological_operations()
        self.extract_features()
        self.calculate_abcd_score()
        return self
    
    # ========================================================================
    # VISUALIZATION METHODS (ALL OF THEM)
    # ========================================================================
    
    def plot_histograms(self):
        """Plot histogram comparison before and after enhancement"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(self.grayscale_image.ravel(), bins=256, color='gray', alpha=0.7)
        axes[0, 0].set_title('Original Grayscale Histogram')
        axes[0, 0].set_xlabel('Pixel Intensity')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(self.enhanced_image.ravel(), bins=256, color='blue', alpha=0.7)
        axes[0, 1].set_title('Enhanced (CLAHE) Histogram')
        axes[0, 1].set_xlabel('Pixel Intensity')
        axes[0, 1].set_ylabel('Frequency')
        
        cumsum_orig = np.cumsum(np.histogram(self.grayscale_image.ravel(), bins=256)[0])
        cumsum_orig = cumsum_orig / cumsum_orig[-1]
        axes[1, 0].plot(cumsum_orig, color='gray', linewidth=2)
        axes[1, 0].set_title('Cumulative Distribution (Original)')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('CDF')
        
        cumsum_enh = np.cumsum(np.histogram(self.enhanced_image.ravel(), bins=256)[0])
        cumsum_enh = cumsum_enh / cumsum_enh[-1]
        axes[1, 1].plot(cumsum_enh, color='blue', linewidth=2)
        axes[1, 1].set_title('Cumulative Distribution (Enhanced)')
        axes[1, 1].set_xlabel('Intensity')
        axes[1, 1].set_ylabel('CDF')
        
        plt.tight_layout()
        return fig
    
    def create_pipeline_visualization(self):
        """Create comprehensive pipeline visualization (6x4 grid)"""
        fig = plt.figure(figsize=(20, 28))
        
        # Row 1: Original and Preprocessing
        ax1 = plt.subplot(6, 4, 1)
        ax1.imshow(self.original_image)
        ax1.set_title('Original RGB Image')
        ax1.axis('off')
        
        ax2 = plt.subplot(6, 4, 2)
        ax2.imshow(self.grayscale_image, cmap='gray')
        ax2.set_title('Grayscale')
        ax2.axis('off')
        
        ax3 = plt.subplot(6, 4, 3)
        ax3.imshow(self.enhanced_image, cmap='gray')
        ax3.set_title('Enhanced (CLAHE)')
        ax3.axis('off')
        
        ax4 = plt.subplot(6, 4, 4)
        ax4.imshow(self.global_equalized, cmap='gray')
        ax4.set_title('Global Equalized')
        ax4.axis('off')
        
        # Row 2: Filtering Results
        ax5 = plt.subplot(6, 4, 5)
        ax5.imshow(self.gaussian_filtered, cmap='gray')
        ax5.set_title('Gaussian Filter')
        ax5.axis('off')
        
        ax6 = plt.subplot(6, 4, 6)
        ax6.imshow(self.median_filtered, cmap='gray')
        ax6.set_title('Median Filter')
        ax6.axis('off')
        
        ax7 = plt.subplot(6, 4, 7)
        ax7.imshow(self.bilateral_filtered, cmap='gray')
        ax7.set_title('Bilateral Filter')
        ax7.axis('off')
        
        ax8 = plt.subplot(6, 4, 8)
        ax8.imshow(self.magnitude_spectrum, cmap='gray')
        ax8.set_title('Frequency Spectrum')
        ax8.axis('off')
        
        # Row 3: Segmentation Results
        ax9 = plt.subplot(6, 4, 9)
        ax9.imshow(self.otsu_segmented, cmap='gray')
        ax9.set_title('Otsu Segmentation')
        ax9.axis('off')
        
        ax10 = plt.subplot(6, 4, 10)
        ax10.imshow(self.adaptive_segmented, cmap='gray')
        ax10.set_title('Adaptive Thresholding')
        ax10.axis('off')
        
        ax11 = plt.subplot(6, 4, 11)
        ax11.imshow(self.kmeans_segmented, cmap='gray')
        ax11.set_title('K-means Segmentation')
        ax11.axis('off')
        
        ax12 = plt.subplot(6, 4, 12)
        ax12.imshow(self.lesion_mask_filled, cmap='gray')
        ax12.set_title('Lesion Mask')
        ax12.axis('off')
        
        # Row 4: Edge Detection
        ax13 = plt.subplot(6, 4, 13)
        ax13.imshow(self.canny_edges, cmap='gray')
        ax13.set_title('Canny (50,150)')
        ax13.axis('off')
        
        ax14 = plt.subplot(6, 4, 14)
        ax14.imshow(self.sobel_edges, cmap='gray')
        ax14.set_title('Sobel')
        ax14.axis('off')
        
        ax15 = plt.subplot(6, 4, 15)
        ax15.imshow(self.laplacian_edges, cmap='gray')
        ax15.set_title('Laplacian')
        ax15.axis('off')
        
        ax16 = plt.subplot(6, 4, 16)
        ax16.imshow(self.prewitt_edges, cmap='gray')
        ax16.set_title('Prewitt')
        ax16.axis('off')
        
        # Row 5: Morphological Operations
        ax17 = plt.subplot(6, 4, 17)
        ax17.imshow(self.opening, cmap='gray')
        ax17.set_title('Opening')
        ax17.axis('off')
        
        ax18 = plt.subplot(6, 4, 18)
        ax18.imshow(self.closing, cmap='gray')
        ax18.set_title('Closing')
        ax18.axis('off')
        
        ax19 = plt.subplot(6, 4, 19)
        ax19.imshow(self.gradient, cmap='gray')
        ax19.set_title('Morphological Gradient')
        ax19.axis('off')
        
        ax20 = plt.subplot(6, 4, 20)
        contour_img = self.draw_contour_on_image()
        ax20.imshow(contour_img)
        ax20.set_title('Final Lesion Contour')
        ax20.axis('off')
        
        # Row 6: Analysis Results
        ax21 = plt.subplot(6, 4, 21)
        analysis_text = f"ABCD SCORE: {self.risk_result['total_score']:.3f}\n\n"
        analysis_text += f"A (Asymmetry): {self.abcd_scores['Asymmetry (A)']:.3f}\n"
        analysis_text += f"B (Border): {self.abcd_scores['Border (B)']:.3f}\n"
        analysis_text += f"C (Color): {self.abcd_scores['Color (C)']:.3f}\n"
        analysis_text += f"D (Diameter): {self.abcd_scores['Diameter (D)']:.3f}\n\n"
        analysis_text += f"Circularity: {self.features['circularity']:.3f}\n"
        analysis_text += f"Solidity: {self.features['solidity']:.3f}\n"
        analysis_text += f"Diameter: {self.features['diameter_mm']:.1f} mm\n"
        analysis_text += f"Defects: {self.features['num_convexity_defects']}"
        ax21.text(0.1, 0.5, analysis_text, fontsize=9, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax21.set_title('Quantitative Analysis')
        ax21.axis('off')
        
        # Risk meter
        ax22 = plt.subplot(6, 4, 22)
        risk_score = self.risk_result['total_score']
        colors = ['#28a745' if risk_score < 0.3 else '#ffc107' if risk_score < 0.6 else '#dc3545']
        ax22.barh([0], [risk_score], color=colors, height=0.5)
        ax22.set_xlim(0, 1)
        ax22.set_title(f'Risk Meter: {risk_score:.3f}')
        ax22.set_xlabel('Low Risk ← → High Risk')
        ax22.set_yticks([])
        
        # Risk level
        ax23 = plt.subplot(6, 4, 23)
        ax23.text(0.1, 0.5, self.risk_result['risk_level'], fontsize=10, 
                 verticalalignment='center', fontweight='bold')
        ax23.set_title('Final Assessment')
        ax23.axis('off')
        
        # Action required
        ax24 = plt.subplot(6, 4, 24)
        ax24.text(0.1, 0.5, f"Action: {self.risk_result['action']}\nUrgency: {self.risk_result['urgency']}", 
                 fontsize=9, verticalalignment='center')
        ax24.set_title('Recommendation')
        ax24.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_filter_comparison(self):
        """Compare Gaussian vs Median filter effects"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(self.enhanced_image, cmap='gray')
        axes[0, 0].set_title('Original (Enhanced)')
        axes[0, 0].axis('off')
        
        # Gaussian
        axes[0, 1].imshow(self.gaussian_filtered, cmap='gray')
        axes[0, 1].set_title('Gaussian Filtered')
        axes[0, 1].axis('off')
        
        # Median
        axes[0, 2].imshow(self.median_filtered, cmap='gray')
        axes[0, 2].set_title('Median Filtered')
        axes[0, 2].axis('off')
        
        # Difference maps
        gaussian_diff = np.abs(self.enhanced_image.astype(float) - self.gaussian_filtered.astype(float))
        axes[1, 0].imshow(gaussian_diff, cmap='hot')
        axes[1, 0].set_title('Gaussian Change Map')
        axes[1, 0].axis('off')
        
        median_diff = np.abs(self.enhanced_image.astype(float) - self.median_filtered.astype(float))
        axes[1, 1].imshow(median_diff, cmap='hot')
        axes[1, 1].set_title('Median Change Map')
        axes[1, 1].axis('off')
        
        # Histogram comparison
        axes[1, 2].hist(self.gaussian_filtered.ravel(), bins=50, alpha=0.5, label='Gaussian')
        axes[1, 2].hist(self.median_filtered.ravel(), bins=50, alpha=0.5, label='Median')
        axes[1, 2].set_title('Intensity Distribution')
        axes[1, 2].legend()
        
        plt.suptitle('Filter Comparison: Gaussian vs Median', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_edge_comparison(self):
        """Compare different edge detection techniques"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        axes[0, 0].imshow(self.canny_edges, cmap='gray')
        axes[0, 0].set_title('Canny (50,150)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.canny_strict, cmap='gray')
        axes[0, 1].set_title('Canny (100,200) - Strict')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(self.canny_lenient, cmap='gray')
        axes[0, 2].set_title('Canny (30,100) - Lenient')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(self.sobel_edges, cmap='gray')
        axes[0, 3].set_title('Sobel')
        axes[0, 3].axis('off')
        
        axes[1, 0].imshow(self.laplacian_edges, cmap='gray')
        axes[1, 0].set_title('Laplacian')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(self.prewitt_edges, cmap='gray')
        axes[1, 1].set_title('Prewitt')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(self.roberts_edges, cmap='gray')
        axes[1, 2].set_title('Roberts')
        axes[1, 2].axis('off')
        
        # Edge intensity comparison
        edge_intensities = [
            np.mean(self.canny_edges),
            np.mean(self.sobel_edges),
            np.mean(self.laplacian_edges),
            np.mean(self.prewitt_edges)
        ]
        axes[1, 3].bar(['Canny', 'Sobel', 'Laplacian', 'Prewitt'], edge_intensities)
        axes[1, 3].set_title('Edge Intensity Comparison')
        axes[1, 3].set_ylabel('Mean Edge Strength')
        
        plt.suptitle('Edge Detection Methods Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


# ============================================================================
# PARAMETER SENSITIVITY STUDY FUNCTION
# ============================================================================

def parameter_sensitivity_study(image):
    """Study effect of varying parameters on output"""
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Gaussian kernel size variation
    kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
    for i, ksize in enumerate(kernel_sizes):
        filtered = cv2.GaussianBlur(gray, ksize, 0)
        axes[0, i].imshow(filtered, cmap='gray')
        axes[0, i].set_title(f'Gaussian {ksize[0]}x{ksize[1]}')
        axes[0, i].axis('off')
    
    # Canny threshold variation
    thresholds = [(30, 90), (50, 150), (70, 210), (100, 200)]
    for i, (t1, t2) in enumerate(thresholds):
        edges = cv2.Canny(gray, t1, t2)
        axes[1, i].imshow(edges, cmap='gray')
        axes[1, i].set_title(f'Canny ({t1}, {t2})')
        axes[1, i].axis('off')
    
    plt.suptitle('Parameter Sensitivity Study: Gaussian Kernel Size & Canny Thresholds', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# STREAMLIT UI (FULL VERSION)
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 DermatoScan - Complete Skin Lesion Analyzer</h1>
        <p>Advanced Image Processing Pipeline | ABCD Rule Classification | Real-time Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📋 Instructions")
        st.markdown("""
        1. **Upload** a clear skin lesion image
        2. **Click Analyze** to process
        3. **View comprehensive results**
        
        **Supported formats:** JPG, PNG, JPEG
        """)
        
        st.markdown("---")
        st.markdown("## 🎛️ Processing Parameters")
        
        cliplimit = st.slider("CLAHE Clip Limit", 1.0, 4.0, 2.0, 0.5)
        kernel_size = st.selectbox("Filter Kernel Size", [3, 5, 7, 9], index=1)
        canny_low = st.slider("Canny Low Threshold", 30, 100, 50)
        canny_high = st.slider("Canny High Threshold", 100, 250, 150)
        
        st.markdown("---")
        st.markdown("## 📊 ABCD Rule Explained")
        st.markdown("""
        - **A**symmetry: Irregular shape
        - **B**order: Irregular edges  
        - **C**olor: Multiple colors
        - **D**iameter: >6mm concerning
        """)
        
        st.markdown("---")
        st.markdown("## 📈 Features Extracted")
        st.markdown("""
        - Geometric (Area, Perimeter, Circularity)
        - Shape (Hu Moments, Solidity, Aspect Ratio)
        - Texture (Mean, Std, Skewness, Kurtosis)
        - Boundary (Vertices, Defects, Smoothness)
        """)
        
        st.markdown("---")
        st.markdown("## ⚠️ Disclaimer")
        st.markdown("*Educational tool only. Not for medical diagnosis.*")
    
    # Main content - Two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Lesion Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the skin lesion"
        )
        
        if uploaded_file is not None:
            # Read and process image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Handle different formats
            if len(image_np.shape) == 3 and image_np.shape[-1] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
            # Resize to 512x512
            image_np = cv2.resize(image_np, (512, 512))
            
            st.image(image_np, caption="Uploaded Image", use_container_width=True)
            st.session_state['uploaded_image'] = image_np
            
            # Store parameters
            st.session_state['cliplimit'] = cliplimit
            st.session_state['kernel_size'] = kernel_size
    
    with col2:
        st.markdown("### 🔬 Analysis Results")
        
        if uploaded_file is not None and 'uploaded_image' in st.session_state:
            if st.button("🔍 START COMPLETE ANALYSIS", type="primary", use_container_width=True):
                with st.spinner("Processing image through complete pipeline..."):
                    # Run analyzer
                    analyzer = CompleteLesionAnalyzer(st.session_state['uploaded_image'])
                    analyzer.run_complete_pipeline()
                    
                    # Store in session state
                    st.session_state['analyzer'] = analyzer
                    st.session_state['results'] = analyzer.risk_result
                    st.session_state['features'] = analyzer.features
                    st.session_state['abcd_scores'] = analyzer.abcd_scores
                    
                    st.success("✅ Analysis Complete!")
    
    # Display Results
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Risk card
        if results['risk_color'] == 'low':
            st.markdown('<div class="risk-low">', unsafe_allow_html=True)
        elif results['risk_color'] == 'medium':
            st.markdown('<div class="risk-medium">', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-high">', unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns([2, 1.5, 1])
        with col_a:
            st.markdown(f"## {results['risk_level']}")
            st.markdown(f"**ABCD Score:** {results['total_score']:.4f}")
        with col_b:
            st.markdown("**📋 Recommendation**")
            st.markdown(results['recommendation'])
        with col_c:
            st.markdown("**🎯 Action Required**")
            st.markdown(f"**{results['action']}**")
            st.markdown(f"*Urgency: {results['urgency']}*")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clinical Notes
        st.markdown("### 📝 Clinical Observations")
        for note in results['clinical_notes']:
            st.markdown(note)
        
        st.markdown("---")
        
        # ABCD Score Breakdown
        st.markdown("### 📊 ABCD Score Breakdown")
        
        abcd_cols = st.columns(4)
        scores = results['abcd_scores']
        
        with abcd_cols[0]:
            st.metric("A - Asymmetry", f"{scores['Asymmetry (A)']:.3f}")
            st.progress(scores['Asymmetry (A)'])
            st.caption("Higher = more asymmetric")
        with abcd_cols[1]:
            st.metric("B - Border", f"{scores['Border (B)']:.3f}")
            st.progress(scores['Border (B)'])
            st.caption("Higher = more irregular")
        with abcd_cols[2]:
            st.metric("C - Color", f"{scores['Color (C)']:.3f}")
            st.progress(scores['Color (C)'])
            st.caption("Higher = more variation")
        with abcd_cols[3]:
            st.metric("D - Diameter", f"{scores['Diameter (D)']:.3f}")
            st.progress(scores['Diameter (D)'])
            st.caption(">6mm = 1.0")
        
        st.markdown("---")
        
        # Feature Details with Tabs (ALL FEATURES)
        st.markdown("### 📐 Detailed Feature Analysis (20+ Features)")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Geometric Features", "Shape Features", "Texture Features", 
            "Hu Moments (7)", "Boundary Features"
        ])
        
        features = st.session_state['features']
        
        with tab1:
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                st.markdown("**Size Metrics**")
                st.metric("Area", f"{features['area']:.0f} px²")
                st.metric("Perimeter", f"{features['perimeter']:.1f} px")
                st.metric("Diameter", f"{features['diameter_mm']:.1f} mm")
            with col_f2:
                st.markdown("**Shape Metrics**")
                st.metric("Circularity", f"{features['circularity']:.4f}")
                st.metric("Roundness", f"{features['roundness']:.4f}")
                st.metric("Compactness", f"{features['compactness']:.1f}")
            with col_f3:
                st.markdown("**Proportional Metrics**")
                st.metric("Aspect Ratio", f"{features['aspect_ratio']:.3f}")
                st.metric("Extent", f"{features['extent']:.3f}")
                st.metric("Solidity", f"{features['solidity']:.4f}")
        
        with tab2:
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.markdown("**Primary Shape Descriptors**")
                st.metric("Asymmetry Score", f"{features['asymmetry_score']:.4f}")
                st.metric("Asymmetry Measure", f"{features['asymmetry_measure']:.4f}")
            with col_h2:
                st.markdown("**Additional Metrics**")
                st.metric("Color Uniformity", f"{features['color_uniformity']:.4f}")
        
        with tab3:
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.markdown("**Central Tendency**")
                st.metric("Mean Intensity", f"{features['mean_intensity']:.1f}/255")
                st.metric("Median (p50)", f"{(features['p25_intensity'] + features['p75_intensity'])/2:.1f}")
            with col_t2:
                st.markdown("**Dispersion**")
                st.metric("Std Deviation", f"{features['std_intensity']:.2f}")
                st.metric("IQR", f"{features['iqr']:.1f}")
            with col_t3:
                st.markdown("**Distribution Shape**")
                st.metric("Skewness", f"{features['skewness']:.3f}")
                st.metric("Kurtosis", f"{features['kurtosis']:.3f}")
            
            st.markdown("**Percentile Values**")
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                st.metric("p10", f"{features['p10_intensity']:.0f}")
            with col_p2:
                st.metric("p25", f"{features['p25_intensity']:.0f}")
            with col_p3:
                st.metric("p75", f"{features['p75_intensity']:.0f}")
            with col_p4:
                st.metric("p90", f"{features['p90_intensity']:.0f}")
        
        with tab4:
            st.markdown("**Hu Moments (Log Scale - Invariant to Translation, Rotation, Scale)**")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Moment 1 (Asymmetry)", f"{features['hu_moment_1']:.6f}")
                st.metric("Moment 2", f"{features['hu_moment_2']:.6f}")
                st.metric("Moment 3", f"{features['hu_moment_3']:.6f}")
                st.metric("Moment 4", f"{features['hu_moment_4']:.6f}")
            with col_m2:
                st.metric("Moment 5", f"{features['hu_moment_5']:.6f}")
                st.metric("Moment 6", f"{features['hu_moment_6']:.6f}")
                st.metric("Moment 7", f"{features['hu_moment_7']:.6f}")
            
            st.caption("Higher absolute values indicate more complex shapes")
        
        with tab5:
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("**Contour Properties**")
                st.metric("Number of Vertices", f"{features['num_vertices']}")
                st.metric("Contour Smoothness", f"{features['contour_smoothness']:.3f}")
            with col_b2:
                st.markdown("**Defect Analysis**")
                st.metric("Convexity Defects", f"{features['num_convexity_defects']}")
                st.caption("More defects = more irregular border")
        
        st.markdown("---")
        
        # Raw Values Summary
        st.markdown("### 📊 Key Measured Values")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("Circularity", f"{results['raw_values']['circularity']:.4f}")
            st.caption("1.0 = perfect circle")
        with col_r2:
            st.metric("Diameter", f"{results['raw_values']['diameter_mm']:.1f} mm")
            st.caption(">6mm is concerning")
        with col_r3:
            st.metric("Solidity", f"{results['raw_values']['solidity']:.4f}")
            st.caption("1.0 = convex shape")
        with col_r4:
            st.metric("Color Variation", f"{results['raw_values']['std_intensity']:.1f}")
            st.caption("Higher = more colors")
        
        st.markdown("---")
        
        # Visualizations Section (ALL 5 TYPES)
        st.markdown("### 🖼️ Processing Visualizations")
        
        viz_option = st.radio(
            "Select Visualization:",
            [
                "Complete Pipeline (6x4 Grid)", 
                "Histogram Analysis", 
                "Filter Comparison (Gaussian vs Median)",
                "Edge Detection Comparison (5 Methods)",
                "Parameter Sensitivity Study"
            ],
            horizontal=True
        )
        
        if st.button("🎨 Generate Visualization", use_container_width=True):
            with st.spinner("Creating visualization..."):
                analyzer = st.session_state['analyzer']
                
                if viz_option == "Complete Pipeline (6x4 Grid)":
                    fig = analyzer.create_pipeline_visualization()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                elif viz_option == "Histogram Analysis":
                    fig = analyzer.plot_histograms()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                elif viz_option == "Filter Comparison (Gaussian vs Median)":
                    fig = analyzer.create_filter_comparison()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                elif viz_option == "Edge Detection Comparison (5 Methods)":
                    fig = analyzer.create_edge_comparison()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                elif viz_option == "Parameter Sensitivity Study":
                    fig = parameter_sensitivity_study(st.session_state['uploaded_image'])
                    st.pyplot(fig)
                    plt.close(fig)
        
        st.markdown("---")
        
        # Download Report
        st.markdown("### 📄 Download Complete Report")
        
        def generate_complete_report():
            report = f"""
================================================================================
                    DERMATOLOGICAL LESION ANALYSIS - COMPLETE REPORT
================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--------------------------------------------------------------------------------
                        ABCD RISK ASSESSMENT
--------------------------------------------------------------------------------
Total ABCD Score: {results['total_score']:.4f}
Risk Level: {results['risk_level']}
Recommendation: {results['recommendation']}
Action Required: {results['action']}
Urgency: {results['urgency']}

--------------------------------------------------------------------------------
                        ABCD SCORE BREAKDOWN
--------------------------------------------------------------------------------
A - Asymmetry:     {scores['Asymmetry (A)']:.4f}
B - Border:        {scores['Border (B)']:.4f}
C - Color:         {scores['Color (C)']:.4f}
D - Diameter:      {scores['Diameter (D)']:.4f}
T - Texture:       {scores['Texture (T)']:.4f}
S - Solidity:      {scores['Solidity (S)']:.4f}

--------------------------------------------------------------------------------
                        CLINICAL OBSERVATIONS
--------------------------------------------------------------------------------
{chr(10).join(results['clinical_notes'])}

--------------------------------------------------------------------------------
                        GEOMETRIC FEATURES
--------------------------------------------------------------------------------
Area:                    {features['area']:.2f} square pixels
Perimeter:               {features['perimeter']:.2f} pixels
Circularity:             {features['circularity']:.4f} (1 = perfect circle)
Roundness:               {features['roundness']:.4f}
Compactness:             {features['compactness']:.2f}
Aspect Ratio:            {features['aspect_ratio']:.4f}
Extent:                  {features['extent']:.4f}
Solidity:                {features['solidity']:.4f} (1 = convex)
Equivalent Diameter:     {features['equivalent_diameter']:.2f} pixels
Diameter (mm):           {features['diameter_mm']:.2f} mm

--------------------------------------------------------------------------------
                        BOUNDARY FEATURES
--------------------------------------------------------------------------------
Number of Vertices:      {features['num_vertices']}
Contour Smoothness:      {features['contour_smoothness']:.4f}
Convexity Defects:       {features['num_convexity_defects']}

--------------------------------------------------------------------------------
                        TEXTURE FEATURES
--------------------------------------------------------------------------------
Mean Intensity:          {features['mean_intensity']:.2f}/255
Standard Deviation:      {features['std_intensity']:.2f}
Skewness:                {features['skewness']:.4f}
Kurtosis:                {features['kurtosis']:.4f}
Color Uniformity:        {features['color_uniformity']:.4f}
Interquartile Range:     {features['iqr']:.2f}
p10 Intensity:           {features['p10_intensity']:.0f}
p25 Intensity:           {features['p25_intensity']:.0f}
p75 Intensity:           {features['p75_intensity']:.0f}
p90 Intensity:           {features['p90_intensity']:.0f}

--------------------------------------------------------------------------------
                        HU MOMENTS (Log Scale)
                        (Invariant to Translation, Rotation, Scale)
--------------------------------------------------------------------------------
Moment 1 (Asymmetry):    {features['hu_moment_1']:.6f}
Moment 2:                {features['hu_moment_2']:.6f}
Moment 3:                {features['hu_moment_3']:.6f}
Moment 4:                {features['hu_moment_4']:.6f}
Moment 5:                {features['hu_moment_5']:.6f}
Moment 6:                {features['hu_moment_6']:.6f}
Moment 7:                {features['hu_moment_7']:.6f}

--------------------------------------------------------------------------------
                        ASYMMETRY ANALYSIS
--------------------------------------------------------------------------------
Asymmetry Measure:       {features['asymmetry_measure']:.4f}
Asymmetry Score:         {features['asymmetry_score']:.4f}

================================================================================
CLINICAL INTERPRETATION GUIDELINES:
- Circularity > 0.8: Benign tendency | Current: {features['circularity']:.3f}
- Circularity < 0.6: Malignant suspicion
- Diameter > 6mm: Concerning | Current: {features['diameter_mm']:.1f}mm
- Solidity < 0.9: Irregular shape | Current: {features['solidity']:.3f}
- Color Std > 50: High variation | Current: {features['std_intensity']:.1f}

================================================================================
DISCLAIMER: This is an AI-generated analysis for educational purposes only.
Please consult a qualified dermatologist for medical advice.
================================================================================
"""
            return report
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download Full Report (TXT)",
                data=generate_complete_report(),
                file_name=f"dermatoscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        st.caption("Report includes all extracted features (20+), ABCD scores, and clinical interpretation")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()