import cv2
import numpy as np
from skimage import measure, morphology, segmentation
from scipy import ndimage as ndi
import pandas as pd

class ParticleAnalyzer:
    def __init__(self):
        pass

    def preprocess(self, image):
        """
        Zero-parameter preprocessing pipeline.
        Converts to grayscale, applies blurring, and uses Otsu's binarization.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Denoising
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu's Thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological cleaning
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        return opening

    def segment(self, binary):
        """
        Advanced Watershed Segmentation to separate touching particles.
        """
        # Distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Threshold distance transform to get seeds
        _, sur_mask = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sur_mask = np.uint8(sur_mask)
        
        # Find unknown region
        unknown = cv2.subtract(binary, sur_mask)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sur_mask)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Watershed logic needs color image or dummy
        dummy_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(dummy_img, markers)
        
        # Cleaned labels
        labels = np.copy(markers)
        labels[markers == -1] = 0 # Boundaries
        labels[markers == 1] = 0  # Background
        
        return labels

    def calculate_metrics(self, labels, binary):
        """
        Calculates standard and novel metrics.
        """
        props = measure.regionprops(labels)
        data = []
        
        total_area = np.sum(binary > 0)
        mean_area = np.mean([p.area for p in props]) if props else 0
        num_particles = len(props)
        
        for p in props:
            # Standard metrics
            area = p.area
            perimeter = p.perimeter
            eccentricity = p.eccentricity
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
            
            # Novel Metric: Shape Complexity Score
            # Ratio of actual perimeter to equivalent circular perimeter
            eq_diameter = p.equivalent_diameter
            circular_perimeter = np.pi * eq_diameter
            complexity_score = perimeter / circular_perimeter if circular_perimeter > 0 else 1
            
            # State Classification
            # Green = Isolated/Circular, Red = Agglomerated/Complex, Yellow = Edge case
            state = "Green"
            if circularity < 0.7 or complexity_score > 1.3:
                state = "Red"
            
            # Simple edge detection
            minr, minc, maxr, maxl = p.bbox
            if minr == 0 or minc == 0 or maxr == labels.shape[0] or maxl == labels.shape[1]:
                state = "Yellow"

            data.append({
                "ID": p.label,
                "Area": area,
                "Perimeter": round(perimeter, 2),
                "Circularity": round(circularity, 2),
                "Complexity": round(complexity_score, 2),
                "State": state
            })
            
        # Aggregate Metric: Particle Aggregation Index (PAI)
        # We can define PAI as the ratio of 'Red' particle area to total area
        red_area = sum([d['Area'] for d in data if d['State'] == "Red"])
        pai = red_area / total_area if total_area > 0 else 0
        
        return pd.DataFrame(data), pai

    def get_colored_output(self, image, labels, df):
        """
        Generates classification image.
        """
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()
            
        color_map = {
            "Green": (0, 255, 0),   # BGR
            "Red": (0, 0, 255),
            "Yellow": (0, 255, 255)
        }
        
        for _, row in df.iterrows():
            mask = (labels == row['ID']).astype(np.uint8)
            color = color_map.get(row['State'], (255, 255, 255))
            
            # Find contours and draw
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, cnts, -1, color, 2)
            
        return output
        
if __name__ == "__main__":
    # Test stub
    import matplotlib.pyplot as plt
    
    # Create synthetic image if needed
    img = np.zeros((400, 400), dtype=np.uint8)
    # Some "particles"
    cv2.circle(img, (100, 100), 20, 255, -1) # Isolated
    cv2.circle(img, (200, 200), 25, 255, -1) # Agglomerated 1
    cv2.circle(img, (220, 220), 25, 255, -1) # Agglomerated 2
    cv2.rectangle(img, (300, 300), (350, 320), 255, -1) # Jagged/Edge
    
    analyzer = ParticleAnalyzer()
    binary = analyzer.preprocess(img)
    labels = analyzer.segment(binary)
    df, pai = analyzer.calculate_metrics(labels, binary)
    output = analyzer.get_colored_output(img, labels, df)
    
    print(f"PAI: {pai:.2f}")
    print(df.head())
