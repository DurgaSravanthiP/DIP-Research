import cv2
import numpy as np
from analyzer import ParticleAnalyzer

# Generate the same synthetic test case for the walkthrough
img = np.zeros((400, 400), dtype=np.uint8)
# Background noise
noise = np.random.randint(0, 30, (400, 400), dtype=np.uint8)
img = cv2.add(img, noise)

# Particles
cv2.circle(img, (100, 100), 20, (200, 200, 200), -1) # Isolated
cv2.circle(img, (250, 250), 25, (200, 200, 200), -1) # Cluster Part 1
cv2.circle(img, (275, 275), 25, (200, 200, 200), -1) # Cluster Part 2
cv2.rectangle(img, (50, 300), (100, 310), (200, 200, 200), -1) # Edge case

analyzer = ParticleAnalyzer()
binary = analyzer.preprocess(img)
labels = analyzer.segment(binary)
df, pai = analyzer.calculate_metrics(labels, binary)
output = analyzer.get_colored_output(img, labels, df)

cv2.imwrite("C:\\Users\\sofia\\.gemini\\antigravity\\brain\\8ff2a1c2-1433-4473-95f1-9641f7754956\\verification_result.png", output)
print("Result saved!")
