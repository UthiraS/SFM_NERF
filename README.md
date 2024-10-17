
# Classical Structure from Motion (SfM) Pipeline

This project implements a classical Structure from Motion (SfM) pipeline to reconstruct 3D structures from multiple 2D images.

## Overview

Structure from Motion is a photogrammetric technique that estimates 3D structures from 2D image sequences. This implementation follows traditional SfM approaches, utilizing:

- Feature detection and matching
- Epipolar geometry calculations
- Non-linear triangulation
- Perspective n-point (PnP) algorithms
- Bundle adjustment

## Dataset

The pipeline operates on a dataset of multiple views of a building, as shown in Fig. 1 of the project description. Ensure your input images are organized in a suitable directory structure.

## Implementation

The SfM pipeline consists of the following key steps:

1. Feature extraction and matching across image pairs
2. Estimating fundamental/essential matrices
3. Recovering camera poses 
4. Triangulating 3D points
5. Registering new views using PnP
6. Refining the reconstruction via bundle adjustment

## Usage

1. Place your input images in the designated data directory
2. Run the main SfM script:

```
python run_sfm.py
```

3. The reconstructed 3D point cloud and camera poses will be saved in the output directory

## Requirements

- Python 3.x
- OpenCV
- NumPy
- SciPy

## Future Work

- Implement dense reconstruction techniques
- Add visualization tools for the sparse point cloud
- Optimize for larger datasets



