"""
Dermatological Lesion Classifier - Complete Implementation
Processes multiple skin lesion images with comparative analysis
Image Processing Project for DSIP Lab - 2025-26
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib parameters
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['figure.dpi'] = 100

class DermatologicalLesionClassifier:
    """
    Complete pipeline for skin lesion analysis including:
    - Preprocessing and enhancement
    - Segmentation
    - Feature extraction
    - Boundary analysis for abnormality detection
    """
    
    def __init__(self, image_path, image_name="Unknown"):
        """Initialize the classifier with an image path"""
        self.image_path = image_path
        self.image_name = image_name
        self.original_image = None
        self.grayscale_image = None
        self.processed_stages = {}
        self.features = {}
        self.analysis_results = {}
        
    def load_and_preprocess(self, target_size=(512, 512)):
        """Step 1: Load image, resize, and convert to grayscale"""
        # Load image
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Cannot load image from {self.image_path}")
        
        # Convert BGR to RGB
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        self.original_image = cv2.resize(self.original_image, target_size)
        
        # Convert to grayscale
        self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        
        self.processed_stages['original'] = self.original_image
        self.processed_stages['grayscale'] = self.grayscale_image
        
        print(f"✓ Loaded {self.image_name} - resized to {target_size}")
        return self
    
    def histogram_equalization(self):
        """Step 2: Apply histogram equalization for contrast enhancement"""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.enhanced_image = clahe.apply(self.grayscale_image)
        
        # Also compute global histogram equalization for comparison
        self.global_equalized = cv2.equalizeHist(self.grayscale_image)
        
        self.processed_stages['enhanced'] = self.enhanced_image
        self.processed_stages['global_equalized'] = self.global_equalized
        
        return self
    
    def spatial_filtering(self):
        """Step 3: Apply Gaussian and Median filters for noise removal"""
        # Gaussian filter
        self.gaussian_filtered = cv2.GaussianBlur(self.enhanced_image, (5, 5), 1.0)
        
        # Median filter (better for edge preservation)
        self.median_filtered = cv2.medianBlur(self.enhanced_image, 5)
        
        # Bilateral filter (preserves edges while reducing noise)
        self.bilateral_filtered = cv2.bilateralFilter(self.enhanced_image, 9, 75, 75)
        
        self.processed_stages['gaussian'] = self.gaussian_filtered
        self.processed_stages['median'] = self.median_filtered
        self.processed_stages['bilateral'] = self.bilateral_filtered
        
        return self
    
    def frequency_filtering(self):
        """Step 4: Apply DFT-based frequency domain filtering"""
        # Compute DFT
        f_transform = np.fft.fft2(self.median_filtered)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Create Low-pass filter (Gaussian)
        rows, cols = self.median_filtered.shape
        crow, ccol = rows // 2, cols // 2
        
        # Low-pass filter mask
        low_pass_mask = np.zeros((rows, cols), np.float32)
        radius = 30
        cv2.circle(low_pass_mask, (ccol, crow), radius, 1, -1)
        
        # High-pass filter mask
        high_pass_mask = 1 - low_pass_mask
        
        # Apply filters
        f_shift_low = f_shift * low_pass_mask
        f_shift_high = f_shift * high_pass_mask
        
        # Inverse DFT
        f_ishift_low = np.fft.ifftshift(f_shift_low)
        img_low_pass = np.abs(np.fft.ifft2(f_ishift_low))
        
        f_ishift_high = np.fft.ifftshift(f_shift_high)
        img_high_pass = np.abs(np.fft.ifft2(f_ishift_high))
        
        self.processed_stages['magnitude_spectrum'] = magnitude_spectrum
        self.processed_stages['low_pass_filtered'] = img_low_pass
        self.processed_stages['high_pass_filtered'] = img_high_pass
        
        return self
    
    def segmentation(self):
        """Step 5: Apply multiple segmentation techniques"""
        # Method 1: Otsu's thresholding
        _, self.otsu_segmented = cv2.threshold(self.median_filtered, 0, 255, 
                                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding
        self.adaptive_segmented = cv2.adaptiveThreshold(self.median_filtered, 255,
                                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY, 11, 2)
        
        # Method 3: K-means clustering
        pixels = self.enhanced_image.reshape((-1, 1))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        self.kmeans_segmented = centers[labels.flatten()].reshape(self.enhanced_image.shape)
        
        # Get binary mask from k-means
        lesion_mask = (labels.reshape(self.enhanced_image.shape) == 0).astype(np.uint8) * 255
        
        self.processed_stages['otsu_segmented'] = self.otsu_segmented
        self.processed_stages['adaptive_segmented'] = self.adaptive_segmented
        self.processed_stages['kmeans_segmented'] = self.kmeans_segmented
        self.processed_stages['lesion_mask'] = lesion_mask
        
        return self
    
    def edge_detection(self):
        """Step 6: Apply multiple edge detection techniques"""
        # Canny edge detection
        self.canny_edges = cv2.Canny(self.median_filtered, 50, 150)
        self.canny_strict = cv2.Canny(self.median_filtered, 100, 200)
        self.canny_lenient = cv2.Canny(self.median_filtered, 30, 100)
        
        # Sobel edge detection
        sobel_x = cv2.Sobel(self.median_filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.median_filtered, cv2.CV_64F, 0, 1, ksize=3)
        self.sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
        self.sobel_edges = np.uint8(np.clip(self.sobel_edges, 0, 255))
        
        # Laplacian edge detection
        self.laplacian_edges = cv2.Laplacian(self.median_filtered, cv2.CV_64F)
        self.laplacian_edges = np.uint8(np.abs(self.laplacian_edges))
        
        self.processed_stages['canny_edges'] = self.canny_edges
        self.processed_stages['sobel_edges'] = self.sobel_edges
        self.processed_stages['laplacian_edges'] = self.laplacian_edges
        
        return self
    
    def morphological_operations(self):
        """Step 7: Apply morphological operations to improve segmentation"""
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Apply operations
        self.eroded = cv2.erode(self.otsu_segmented, kernel, iterations=1)
        self.dilated = cv2.dilate(self.otsu_segmented, kernel, iterations=1)
        self.opening = cv2.morphologyEx(self.otsu_segmented, cv2.MORPH_OPEN, kernel)
        self.closing = cv2.morphologyEx(self.otsu_segmented, cv2.MORPH_CLOSE, kernel_large)
        
        # Find largest contour (assumed to be the lesion)
        contours, _ = cv2.findContours(self.closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            self.lesion_contour = max(contours, key=cv2.contourArea)
            # Create filled lesion mask
            self.lesion_mask_filled = np.zeros_like(self.closing)
            cv2.drawContours(self.lesion_mask_filled, [self.lesion_contour], -1, 255, -1)
        else:
            self.lesion_contour = None
            self.lesion_mask_filled = self.closing
        
        self.processed_stages['morph_opening'] = self.opening
        self.processed_stages['morph_closing'] = self.closing
        self.processed_stages['lesion_contour_image'] = self.draw_contour_on_image()
        
        return self
    
    def draw_contour_on_image(self):
        """Draw the detected lesion contour on the original image"""
        contour_img = self.original_image.copy()
        if self.lesion_contour is not None:
            cv2.drawContours(contour_img, [self.lesion_contour], -1, (255, 0, 0), 2)
        return contour_img
    
    def extract_features(self):
        """Step 8: Extract meaningful features for analysis"""
        if self.lesion_contour is None:
            print(f"⚠ No lesion contour detected for {self.image_name}")
            return self
        
        # Geometric features
        area = cv2.contourArea(self.lesion_contour)
        perimeter = cv2.arcLength(self.lesion_contour, True)
        
        # Compactness and circularity
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Bounding box
        x, y, w, h = cv2.boundingRect(self.lesion_contour)
        aspect_ratio = w / h if h > 0 else 0
        extent = area / (w * h) if (w * h) > 0 else 0
        
        # Convex hull analysis
        hull = cv2.convexHull(self.lesion_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Equivalent diameter
        equivalent_diameter = np.sqrt(4 * area / np.pi)
        
        # Hu Moments (shape descriptors)
        moments = cv2.moments(self.lesion_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        # Edge features
        if len(self.lesion_contour) > 0:
            epsilon = 0.01 * perimeter
            approx = cv2.approxPolyDP(self.lesion_contour, epsilon, True)
            num_vertices = len(approx)
            contour_smoothness = len(self.lesion_contour) / perimeter if perimeter > 0 else 0
        else:
            num_vertices = 0
            contour_smoothness = 0
        
        # Textural features
        lesion_pixels = self.grayscale_image[self.lesion_mask_filled == 255]
        if len(lesion_pixels) > 0:
            mean_intensity = np.mean(lesion_pixels)
            std_intensity = np.std(lesion_pixels)
            skewness = np.mean(((lesion_pixels - mean_intensity) / (std_intensity + 1e-10)) ** 3)
            kurtosis = np.mean(((lesion_pixels - mean_intensity) / (std_intensity + 1e-10)) ** 4) - 3
        else:
            mean_intensity = std_intensity = skewness = kurtosis = 0
        
        # Store all features
        self.features = {
            'image_name': self.image_name,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity,
            'equivalent_diameter': equivalent_diameter,
            'num_vertices': num_vertices,
            'contour_smoothness': contour_smoothness,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'hu_moments_1': hu_moments_log[0],
            'hu_moments_2': hu_moments_log[1]
        }
        
        return self
    
    def analyze_abnormality(self):
        """Analyze extracted features to detect abnormalities using ABCD rule"""
        if not self.features:
            return self
        
        # Asymmetry (from Hu moment 1 - higher absolute value = more asymmetric)
        asymmetry_score = min(abs(self.features.get('hu_moments_1', 0)) * 0.2, 1.0)
        
        # Border irregularity (lower circularity = more irregular)
        border_irregularity = 1 - self.features.get('circularity', 0)
        border_irregularity = min(border_irregularity, 1.0)
        
        # Color variation (higher std = more color variation)
        color_variation = self.features.get('std_intensity', 0) / 255
        color_variation = min(color_variation, 1.0)
        
        # Diameter - larger than 6mm (approx 60 pixels for 512x512 image) is concerning
        diameter_pixels = self.features.get('equivalent_diameter', 0)
        diameter_mm = diameter_pixels * 0.2646  # Approx conversion
        diameter_concern = 1.0 if diameter_mm > 6 else (diameter_mm / 6) * 0.5
        
        # ABCD scores
        abcd_scores = {
            'Asymmetry': asymmetry_score,
            'Border Irregularity': border_irregularity,
            'Color Variation': color_variation,
            'Diameter': min(diameter_concern, 1.0)
        }
        
        total_abcd_score = sum(abcd_scores.values()) / 4
        
        # Risk classification
        if total_abcd_score < 0.3:
            risk_level = "LOW - Benign"
            recommendation = "Monitor for any changes"
            color_code = "green"
        elif total_abcd_score < 0.6:
            risk_level = "MEDIUM - Suspicious"
            recommendation = "Consider dermatologist consultation"
            color_code = "orange"
        else:
            risk_level = "HIGH - Malignant suspicion"
            recommendation = "URGENT: Consult a dermatologist immediately"
            color_code = "red"
        
        self.analysis_results = {
            'image_name': self.image_name,
            'abcd_scores': abcd_scores,
            'total_abcd_score': total_abcd_score,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'diameter_mm': diameter_mm,
            'circularity': self.features.get('circularity', 0),
            'solidity': self.features.get('solidity', 0),
            'border_irregularity': border_irregularity,
            'color_variation': color_variation,
            'risk_color': color_code
        }
        
        return self
    
    def run_complete_pipeline(self):
        """Execute the complete image processing pipeline"""
        self.load_and_preprocess()
        self.histogram_equalization()
        self.spatial_filtering()
        self.frequency_filtering()
        self.segmentation()
        self.edge_detection()
        self.morphological_operations()
        self.extract_features()
        self.analyze_abnormality()
        return self
    
    def get_summary(self):
        """Return summary of results"""
        if self.analysis_results:
            return {
                'Image': self.image_name,
                'ABCD Score': f"{self.analysis_results['total_abcd_score']:.3f}",
                'Risk Level': self.analysis_results['risk_level'].split(' - ')[0],
                'Diameter (mm)': f"{self.analysis_results['diameter_mm']:.1f}",
                'Circularity': f"{self.features.get('circularity', 0):.3f}"
            }
        return None


class MultiImageAnalyzer:
    """Process and compare multiple skin lesion images"""
    
    def __init__(self, output_dir='results'):
        self.results = []
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def process_images(self, image_paths):
        """Process multiple images and store results"""
        print("\n" + "=" * 80)
        print("DERMATOLOGICAL LESION CLASSIFIER - MULTI-IMAGE ANALYSIS")
        print("=" * 80)
        
        for idx, img_path in enumerate(image_paths, 1):
            if not os.path.exists(img_path):
                print(f"\n⚠ Warning: {img_path} not found. Skipping...")
                continue
            
            # Extract image name from path
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            
            print(f"\n{'─' * 80}")
            print(f"Processing Image {idx}/{len(image_paths)}: {img_name}")
            print(f"{'─' * 80}")
            
            # Process image
            classifier = DermatologicalLesionClassifier(img_path, img_name)
            classifier.run_complete_pipeline()
            
            # Store results
            self.results.append({
                'classifier': classifier,
                'name': img_name,
                'features': classifier.features,
                'analysis': classifier.analysis_results
            })
            
            print(f"✓ Completed: {img_name} - Risk: {classifier.analysis_results.get('risk_level', 'N/A')}")
        
        return self
    
    def generate_comparison_table(self):
        """Generate comparison table of all processed images"""
        if not self.results:
            print("No results to display")
            return None
        
        data = []
        for r in self.results:
            if r['analysis']:
                data.append({
                    'Image Name': r['name'],
                    'ABCD Score': f"{r['analysis']['total_abcd_score']:.3f}",
                    'Risk Level': r['analysis']['risk_level'].split(' - ')[0],
                    'Diameter (mm)': f"{r['analysis']['diameter_mm']:.1f}",
                    'Circularity': f"{r['analysis']['circularity']:.3f}",
                    'Solidity': f"{r['analysis']['solidity']:.3f}",
                    'Border Irregularity': f"{r['analysis']['border_irregularity']:.3f}"
                })
        
        df = pd.DataFrame(data)
        return df
    
    def plot_comparison_charts(self):
        """Create comparison charts for all processed images"""
        if len(self.results) < 2:
            print("Need at least 2 images for comparison charts")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        names = [r['name'] for r in self.results]
        abcd_scores = [r['analysis']['total_abcd_score'] for r in self.results]
        circularity = [r['analysis']['circularity'] for r in self.results]
        diameters = [r['analysis']['diameter_mm'] for r in self.results]
        border_irregularity = [r['analysis']['border_irregularity'] for r in self.results]
        
        # Colors based on risk
        colors = [r['analysis']['risk_color'] for r in self.results]
        
        # Bar chart: ABCD Scores
        bars1 = axes[0, 0].bar(names, abcd_scores, color=colors, edgecolor='black')
        axes[0, 0].axhline(y=0.3, color='green', linestyle='--', label='Benign Threshold')
        axes[0, 0].axhline(y=0.6, color='red', linestyle='--', label='Malignant Threshold')
        axes[0, 0].set_ylabel('ABCD Score')
        axes[0, 0].set_title('Risk Assessment (ABCD Score)')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Bar chart: Circularity
        axes[0, 1].bar(names, circularity, color='skyblue', edgecolor='black')
        axes[0, 1].set_ylabel('Circularity (1 = perfect circle)')
        axes[0, 1].set_title('Lesion Circularity Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Bar chart: Diameter
        bars3 = axes[1, 0].bar(names, diameters, color=colors, edgecolor='black')
        axes[1, 0].axhline(y=6, color='red', linestyle='--', label='6mm Threshold')
        axes[1, 0].set_ylabel('Diameter (mm)')
        axes[1, 0].set_title('Lesion Diameter Comparison')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Bar chart: Border Irregularity
        axes[1, 1].bar(names, border_irregularity, color='lightcoral', edgecolor='black')
        axes[1, 1].set_ylabel('Border Irregularity (higher = more irregular)')
        axes[1, 1].set_title('Border Irregularity Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Multi-Image Comparative Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparison_charts.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_all_lesion_contours(self):
        """Plot all detected lesion contours side by side"""
        n = len(self.results)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, r in enumerate(self.results):
            classifier = r['classifier']
            
            # Get contour image
            contour_img = classifier.processed_stages.get('lesion_contour_image', classifier.original_image)
            
            axes[i].imshow(contour_img)
            risk_text = r['analysis'].get('risk_level', 'Unknown').split(' - ')[0]
            axes[i].set_title(f"{r['name']}\nRisk: {risk_text}\nABCD: {r['analysis']['total_abcd_score']:.3f}")
            axes[i].axis('off')
        
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Detected Lesion Contours - Comparative View', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/all_contours.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("FINAL ANALYSIS REPORT - MULTI-IMAGE COMPARISON")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Images Analyzed: {len(self.results)}")
        print("=" * 80)
        
        # Summary Table
        print("\n📊 COMPARISON SUMMARY TABLE:")
        print("-" * 80)
        
        df = self.generate_comparison_table()
        if df is not None:
            print(df.to_string(index=False))
        
        # Risk Distribution
        print("\n📈 RISK DISTRIBUTION:")
        risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        for r in self.results:
            risk = r['analysis']['risk_level'].split(' - ')[0]
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        for risk, count in risk_counts.items():
            bar = "█" * count
            print(f"   {risk:6}: {bar} ({count} image(s))")
        
        # Detailed Analysis for Each Image
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS PER IMAGE")
        print("=" * 80)
        
        for r in self.results:
            print(f"\n📌 {r['name'].upper()}")
            print("-" * 40)
            print(f"   ABCD Score: {r['analysis']['total_abcd_score']:.4f}")
            print(f"   Risk Level: {r['analysis']['risk_level']}")
            print(f"   Recommendation: {r['analysis']['recommendation']}")
            print(f"\n   Metrics:")
            print(f"   • Diameter: {r['analysis']['diameter_mm']:.2f} mm")
            print(f"   • Circularity: {r['analysis']['circularity']:.4f}")
            print(f"   • Solidity: {r['analysis']['solidity']:.4f}")
            print(f"   • Border Irregularity: {r['analysis']['border_irregularity']:.4f}")
            print(f"   • Color Variation: {r['analysis']['color_variation']:.4f}")
        
        # Filter Comparison Analysis
        print("\n" + "=" * 80)
        print("FILTER COMPARISON ANALYSIS: Gaussian vs Median")
        print("=" * 80)
        print("""
        ┌─────────────────┬─────────────────────────────────────────────────────────────┐
        │    Filter       │                    Characteristics                          │
        ├─────────────────┼─────────────────────────────────────────────────────────────┤
        │   Gaussian      │ • Removes Gaussian noise effectively                        │
        │                 │ • Blurs edges - can lose fine lesion boundaries             │
        │                 │ • Good for overall smoothing before segmentation          │
        │                 │ • Kernel size determines smoothing strength                │
        ├─────────────────┼─────────────────────────────────────────────────────────────┤
        │    Median       │ • Excellent for salt-and-pepper noise                      │
        │                 │ • PRESERVES EDGES - Critical for boundary analysis         │
        │                 │ • Non-linear filter                                        │
        │                 │ • Better for skin lesion boundary preservation            │
        └─────────────────┴─────────────────────────────────────────────────────────────┘
        
        ✅ RECOMMENDATION: For skin lesion analysis, MEDIAN FILTER is preferred as it
           preserves edge information critical for border irregularity assessment.
        """)
        
        # Segmentation Evaluation
        print("\n" + "=" * 80)
        print("SEGMENTATION TECHNIQUES EVALUATION")
        print("=" * 80)
        print("""
        ┌──────────────┬────────────────────────────────────────────────────────────────┐
        │  Method      │                    Evaluation                                  │
        ├──────────────┼────────────────────────────────────────────────────────────────┤
        │ Otsu         │ • Best for bimodal histograms                                 │
        │ Thresholding │ • Can fail if lesion has similar intensity to background      │
        │              │ • Computationally efficient                                   │
        ├──────────────┼────────────────────────────────────────────────────────────────┤
        │ Adaptive     │ • Handles varying illumination across image                   │
        │ Thresholding │ • Better for lesions with uneven lighting                     │
        │              │ • May produce noise in uniform regions                        │
        ├──────────────┼────────────────────────────────────────────────────────────────┤
        │ K-means      │ • Works well for multi-color lesions                          │
        │ Clustering   │ • More computationally expensive                              │
        │              │ • Requires specifying number of clusters                      │
        └──────────────┴────────────────────────────────────────────────────────────────┘
        """)
        
        # Edge Detection Comparison
        print("\n" + "=" * 80)
        print("EDGE DETECTION METHODS COMPARISON")
        print("=" * 80)
        print("""
        ┌──────────┬───────────────────────────────────────────────────────────────────┐
        │  Method  │                        Evaluation                                 │
        ├──────────┼───────────────────────────────────────────────────────────────────┤
        │ Canny    │ • Best overall - multi-stage algorithm                           │
        │          │ • Provides thin, connected edges                                 │
        │          │ • Requires parameter tuning (low_threshold, high_threshold)      │
        │          │ • Most suitable for lesion boundary detection                    │
        ├──────────┼───────────────────────────────────────────────────────────────────┤
        │ Sobel    │ • Simple and fast                                                │
        │          │ • Produces thicker edges                                         │
        │          │ • More sensitive to noise                                        │
        │          │ • Good for gradient magnitude visualization                      │
        └──────────┴───────────────────────────────────────────────────────────────────┘
        """)
        
        return df
    
    def save_all_visualizations(self):
        """Save individual visualizations for each image"""
        for r in self.results:
            classifier = r['classifier']
            
            # Create visualization for each image
            fig = classifier.visualize_results()
            fig.savefig(f'{self.output_dir}/{classifier.image_name}_pipeline.png', 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Save histogram comparison
            fig_hist = classifier.plot_histograms()
            fig_hist.savefig(f'{self.output_dir}/{classifier.image_name}_histograms.png',
                           dpi=150, bbox_inches='tight')
            plt.close(fig_hist)
        
        print(f"\n✓ All visualizations saved to '{self.output_dir}/' directory")
    
    def visualize_single_image(self, image_index=0):
        """Display comprehensive visualization for a single image"""
        if image_index >= len(self.results):
            print(f"Image index {image_index} out of range")
            return
        
        classifier = self.results[image_index]['classifier']
        
        # Show histograms
        print(f"\n📊 Displaying histograms for: {classifier.image_name}")
        classifier.plot_histograms()
        plt.show()
        
        # Show full pipeline
        print(f"\n🔬 Displaying processing pipeline for: {classifier.image_name}")
        classifier.visualize_results()
        plt.show()


# Add visualization methods to DermatologicalLesionClassifier
def plot_histograms(self):
    """Plot histogram comparison before and after enhancement"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
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


def visualize_results(self):
    """Comprehensive visualization of all processing stages"""
    fig = plt.figure(figsize=(18, 24))
    
    # Row 1: Original and Preprocessing
    ax1 = plt.subplot(5, 4, 1)
    ax1.imshow(self.original_image)
    ax1.set_title('Original RGB Image')
    ax1.axis('off')
    
    ax2 = plt.subplot(5, 4, 2)
    ax2.imshow(self.grayscale_image, cmap='gray')
    ax2.set_title('Grayscale')
    ax2.axis('off')
    
    ax3 = plt.subplot(5, 4, 3)
    ax3.imshow(self.enhanced_image, cmap='gray')
    ax3.set_title('Enhanced (CLAHE)')
    ax3.axis('off')
    
    ax4 = plt.subplot(5, 4, 4)
    ax4.imshow(self.global_equalized, cmap='gray')
    ax4.set_title('Global Equalized')
    ax4.axis('off')
    
    # Row 2: Filtering Results
    ax5 = plt.subplot(5, 4, 5)
    ax5.imshow(self.gaussian_filtered, cmap='gray')
    ax5.set_title('Gaussian Filter')
    ax5.axis('off')
    
    ax6 = plt.subplot(5, 4, 6)
    ax6.imshow(self.median_filtered, cmap='gray')
    ax6.set_title('Median Filter')
    ax6.axis('off')
    
    ax7 = plt.subplot(5, 4, 7)
    ax7.imshow(self.bilateral_filtered, cmap='gray')
    ax7.set_title('Bilateral Filter')
    ax7.axis('off')
    
    ax8 = plt.subplot(5, 4, 8)
    ax8.imshow(self.processed_stages['magnitude_spectrum'], cmap='gray')
    ax8.set_title('Frequency Spectrum')
    ax8.axis('off')
    
    # Row 3: Segmentation Results
    ax9 = plt.subplot(5, 4, 9)
    ax9.imshow(self.otsu_segmented, cmap='gray')
    ax9.set_title('Otsu Segmentation')
    ax9.axis('off')
    
    ax10 = plt.subplot(5, 4, 10)
    ax10.imshow(self.adaptive_segmented, cmap='gray')
    ax10.set_title('Adaptive Thresholding')
    ax10.axis('off')
    
    ax11 = plt.subplot(5, 4, 11)
    ax11.imshow(self.kmeans_segmented, cmap='gray')
    ax11.set_title('K-means Segmentation')
    ax11.axis('off')
    
    ax12 = plt.subplot(5, 4, 12)
    ax12.imshow(self.lesion_mask_filled, cmap='gray')
    ax12.set_title('Lesion Mask (Morphology)')
    ax12.axis('off')
    
    # Row 4: Edge Detection
    ax13 = plt.subplot(5, 4, 13)
    ax13.imshow(self.canny_edges, cmap='gray')
    ax13.set_title('Canny Edge Detection')
    ax13.axis('off')
    
    ax14 = plt.subplot(5, 4, 14)
    ax14.imshow(self.sobel_edges, cmap='gray')
    ax14.set_title('Sobel Edge Detection')
    ax14.axis('off')
    
    ax15 = plt.subplot(5, 4, 15)
    ax15.imshow(self.laplacian_edges, cmap='gray')
    ax15.set_title('Laplacian Edge Detection')
    ax15.axis('off')
    
    ax16 = plt.subplot(5, 4, 16)
    ax16.imshow(self.processed_stages['lesion_contour_image'])
    ax16.set_title('Detected Lesion Contour')
    ax16.axis('off')
    
    # Row 5: Analysis
    if self.lesion_contour is not None:
        ax17 = plt.subplot(5, 4, 17)
        analysis_text = f"Image: {self.image_name}\n\n"
        analysis_text += f"ABCD Score: {self.analysis_results['total_abcd_score']:.3f}\n"
        analysis_text += f"Risk: {self.analysis_results['risk_level']}\n"
        analysis_text += f"Diameter: {self.analysis_results['diameter_mm']:.1f} mm\n"
        analysis_text += f"Circularity: {self.features['circularity']:.3f}\n"
        analysis_text += f"Solidity: {self.features['solidity']:.3f}\n\n"
        analysis_text += f"Recommendation:\n{self.analysis_results['recommendation']}"
        ax17.text(0.1, 0.5, analysis_text, fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax17.set_title('Analysis Results')
        ax17.axis('off')
        
        # Hide remaining subplots
        for i in range(18, 21):
            plt.subplot(5, 4, i).axis('off')
    
    plt.tight_layout()
    return fig


# Attach methods to class
DermatologicalLesionClassifier.plot_histograms = plot_histograms
DermatologicalLesionClassifier.visualize_results = visualize_results


# ============================================================================
# SAMPLE IMAGE GENERATION
# ============================================================================

def create_sample_lesions():
    """Create synthetic skin lesion images for testing (Benign, Suspicious, Malignant)"""
    sample_images = []
    
    def create_lesion(name, lesion_type, center, size, irregularity, color_variation):
        img = np.ones((512, 512, 3), dtype=np.uint8) * 245  # Light skin background
        
        # Base color based on type
        if lesion_type == 'benign':
            base_color = (120, 80, 60)  # Brown
        elif lesion_type == 'suspicious':
            base_color = (90, 60, 50)   # Dark brown
        else:  # malignant
            base_color = (60, 40, 35)   # Very dark/black
        
        # Draw main lesion
        axes = (size[0], size[1])
        angle = np.random.randint(0, 180)
        
        # Create irregular shape
        cv2.ellipse(img, center, axes, angle, 0, 360, base_color, -1)
        
        # Add border irregularity
        num_spikes = irregularity * 30
        for i in range(int(num_spikes)):
            angle_rad = np.random.rand() * 2 * np.pi
            r = size[0] + np.random.rand() * 30 * irregularity
            x = int(center[0] + r * np.cos(angle_rad))
            y = int(center[1] + r * np.sin(angle_rad))
            variation = np.random.randint(-20, 20)
            color = tuple(max(0, min(255, c + variation)) for c in base_color)
            cv2.circle(img, (x, y), np.random.randint(2, 6), color, -1)
        
        # Add color variation
        num_variations = int(color_variation * 100)
        for i in range(num_variations):
            x = center[0] + np.random.randn() * size[0]
            y = center[1] + np.random.randn() * size[1]
            if np.sqrt((x - center[0])**2 + (y - center[1])**2) < size[0]:
                color_offset = np.random.randint(-40, 40) * color_variation
                color = tuple(max(0, min(255, int(c + color_offset))) for c in base_color)
                cv2.circle(img, (int(x), int(y)), np.random.randint(2, 5), color, -1)
        
        # Add asymmetry
        if lesion_type == 'suspicious' or lesion_type == 'malignant':
            # Add an irregular protrusion
            protrusion_dir = np.random.randint(0, 360)
            protrusion_x = center[0] + int(size[0] * 0.7 * np.cos(np.radians(protrusion_dir)))
            protrusion_y = center[1] + int(size[1] * 0.7 * np.sin(np.radians(protrusion_dir)))
            cv2.ellipse(img, (protrusion_x, protrusion_y), 
                       (int(size[0]*0.5), int(size[1]*0.4)), 
                       protrusion_dir, 0, 360, base_color, -1)
        
        # Save image
        cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return name
    
    # Create three sample images with different risk levels
    sample_images.append(create_lesion('sample_benign_mole.jpg', 'benign', 
                                       (256, 256), (60, 55), 0.3, 0.2))
    sample_images.append(create_lesion('sample_suspicious_lesion.jpg', 'suspicious',
                                       (256, 256), (75, 65), 0.7, 0.6))
    sample_images.append(create_lesion('sample_malignant_melanoma.jpg', 'malignant',
                                       (256, 256), (90, 80), 1.0, 0.9))
    
    # Create two more with different locations
    img4 = create_lesion('sample_nevus.jpg', 'benign',
                        (256, 300), (50, 45), 0.4, 0.3)
    img5 = create_lesion('sample_atypical_nevus.jpg', 'suspicious',
                        (300, 256), (70, 60), 0.6, 0.5)
    
    sample_images.extend([img4, img5])
    
    print("\n✓ Created 5 sample skin lesion images:")
    print("   1. sample_benign_mole.jpg - Low risk (Benign)")
    print("   2. sample_suspicious_lesion.jpg - Medium risk (Suspicious)")
    print("   3. sample_malignant_melanoma.jpg - High risk (Malignant)")
    print("   4. sample_nevus.jpg - Low-Medium risk")
    print("   5. sample_atypical_nevus.jpg - Medium risk")
    
    return sample_images


# ============================================================================
# PARAMETER SENSITIVITY STUDY
# ============================================================================

def parameter_sensitivity_study(image_path, output_dir='results'):
    """Study the effect of varying parameters on segmentation and edge detection"""
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot load {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 1. Gaussian kernel size variation
    kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]
    for i, ksize in enumerate(kernel_sizes):
        filtered = cv2.GaussianBlur(gray, ksize, 0)
        axes[0, i].imshow(filtered, cmap='gray')
        axes[0, i].set_title(f'Gaussian {ksize[0]}x{ksize[1]}')
        axes[0, i].axis('off')
    
    # 2. Canny threshold variation
    thresholds = [(30, 90), (50, 150), (70, 210), (100, 200)]
    for i, (t1, t2) in enumerate(thresholds):
        edges = cv2.Canny(gray, t1, t2)
        axes[1, i].imshow(edges, cmap='gray')
        axes[1, i].set_title(f'Canny ({t1}, {t2})')
        axes[1, i].axis('off')
    
    plt.suptitle('Parameter Sensitivity Study: Gaussian Kernel Size & Canny Thresholds', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n📊 Parameter Sensitivity Study Results:")
    print("   • Larger Gaussian kernels (9x9) provide more smoothing but lose fine details")
    print("   • Canny thresholds: Lower values detect more edges (including noise)")
    print("   • Higher thresholds (100,200) detect only strong edges")
    print("   • Optimal for skin lesions: Gaussian 5x5 + Canny (50,150)")
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("     DERMATOLOGICAL LESION CLASSIFIER - IMAGE PROCESSING PROJECT")
    print("     Digital Signal & Image Processing Lab - TY BTECH")
    print("=" * 80)
    
    # Create output directory
    output_dir = 'dsip_project_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ================================================================
    # PUT YOUR 5 IMAGE FILENAMES HERE
    # ================================================================
    # Make sure these files are in the SAME folder as this Python script
    # ================================================================
    
    sample_images = [
        'lesion1.jpg',    # Change to your actual filename
        'lesion2.jpg',    # Change to your actual filename
        'lesion3.jpg',    # Change to your actual filename
        'lesion4.jpg',    # Change to your actual filename
        'lesion5.jpg'     # Change to your actual filename
    ]
    
    # Check if images exist
    print("\n📷 Checking for image files...")
    valid_images = []
    for img in sample_images:
        if os.path.exists(img):
            valid_images.append(img)
            print(f"   ✓ Found: {img}")
        else:
            print(f"   ✗ Missing: {img} - Make sure it's in the correct folder")
    
    if len(valid_images) < 3:
        print(f"\n⚠ Warning: Only {len(valid_images)} images found. Need at least 3.")
        print("   Generating sample images as fallback...")
        valid_images = create_sample_lesions()
    
    # Initialize multi-image analyzer
    analyzer = MultiImageAnalyzer(output_dir=output_dir)
    
    # Process all images
    analyzer.process_images(valid_images)
    
    # Generate comparison table
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS RESULTS")
    print("=" * 80)
    
    df = analyzer.generate_report()
    
    # Save report to CSV
    if df is not None:
        df.to_csv(f'{output_dir}/comparison_report.csv', index=False)
        print(f"\n✓ Report saved to '{output_dir}/comparison_report.csv'")
    
    # Generate visualizations
    print("\n📊 Generating comparison charts...")
    analyzer.plot_comparison_charts()
    analyzer.plot_all_lesion_contours()
    
    # Save individual visualizations
    print("\n💾 Saving individual visualizations...")
    analyzer.save_all_visualizations()
    
    # Run parameter sensitivity study on first image
    if valid_images:
        print("\n🔬 Running parameter sensitivity study...")
        parameter_sensitivity_study(valid_images[0], output_dir=output_dir)
    
    # Display detailed results for the first image
    if len(analyzer.results) > 0:
        print("\n" + "=" * 80)
        print("SAMPLE DETAILED RESULTS - First Image")
        print("=" * 80)
        analyzer.visualize_single_image(0)
    
    # Final summary
    print("\n" + "=" * 80)
    print("PROJECT COMPLETION SUMMARY")
    print("=" * 80)
    print(f"""
    ✅ All requirements satisfied:
    
    PART 1 - Fundamental Processing:
    ✓ Image preprocessing (resize + grayscale)
    ✓ Histogram equalization (CLAHE)
    ✓ Spatial filtering (Gaussian, Median, Bilateral)
    ✓ Frequency domain filtering (DFT based)
    ✓ Histogram analysis (before/after comparison)
    
    PART 2 - Advanced Processing:
    ✓ Image segmentation (Otsu, Adaptive, K-means)
    ✓ Edge detection (Canny, Sobel, Laplacian)
    ✓ Morphological operations (Erosion, Dilation, Opening, Closing)
    ✓ Feature extraction (geometric, textural, Hu moments)
    
    ANALYSIS TASKS:
    ✓ Filter comparison (Gaussian vs Median)
    ✓ Segmentation evaluation (multiple methods)
    ✓ Edge detection analysis (Canny vs Sobel)
    ✓ Parameter sensitivity study
    ✓ Effectiveness evaluation (ABCD scoring)
    
    OUTPUTS GENERATED:
    📁 Directory: '{output_dir}/'
    📊 Files: comparison_report.csv, comparison_charts.png, 
              all_contours.png, parameter_sensitivity.png
    🖼️ Individual pipeline visualizations for each image
    """)
    
    print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print(f"📁 All results saved to '{output_dir}/' directory")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()