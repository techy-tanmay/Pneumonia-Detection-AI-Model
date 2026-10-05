PneumoVision — Sample Chest Radiographs
==========================================

This folder contains verified test radiographs for evaluating the PneumoVision AI localization system:

1. sample_chest_radiograph.dcm
   - Format: True Medical DICOM (.dcm), 16-bit pixel allocation (12-bit stored)
   - Photometric Interpretation: MONOCHROME2
   - Content: Chest radiograph presenting opacity region
   - Use: Tests end-to-end DICOM parsing, percentile clipping, and web display conversion

2. sample_pneumonia_right_opacity.png
   - Format: PNG (1024 x 1024)
   - Content: Synthetic radiograph presenting alveolar infiltrate/opacity in the Right Mid/Lower zone

3. sample_bilateral_opacities.jpg
   - Format: JPEG (1024 x 1024)
   - Content: Bilateral perihilar and basilar infiltrates

4. sample_normal_chest.png
   - Format: PNG (1024 x 1024)
   - Content: Unremarkable chest radiograph with clear lung fields (tests no-detection case)

To test:
Upload any of these files into the PneumoVision web interface or click "Load Sample" directly from the dashboard.
