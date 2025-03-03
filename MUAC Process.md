### **Step-by-Step Solution**

1. **Segmentation & Preprocessing**
    
    - **Background Removal**: Use thresholding or a SAM_VIT model to segment the child from the white backdrop.
    - **Image Alignment**: Ensure consistency in scale and orientation across views (if possible).
        
2. **2D Pose Estimation**
    - **Key point Detection**: Use a pose estimation model (Either MMPose/OpenPose) to locate shoulder, elbow, and wrist joints in each image.
    - **Noise Reduction**: Average key points across multiple images of the same view to minimize detection errors.
        
3. **3D Key point Triangulation**
    
    - **Camera Calibration**: If possible, calibrate cameras using a checkerboard or use Structure from Motion (SfM) to estimate intrinsic/extrinsic parameters.
    - **Triangulate 3D Points**: Use 2D key points from multiple views to compute 3D positions of shoulder and elbow joints.
        
4. **Identify Measurement Point**
    
    - **Midpoint Calculation**: Determine the midpoint between the shoulder and elbow in 3D space to locate the upper arm's circumference measurement site.
        
5. **3D Reconstruction & Measurement**
    
    - **Visual Hull/Shape-from-Silhouette**: Reconstruct a 3D model using segmented silhouettes (tools: COLMAP, MeshLab). (Final Tool not decided yet)
        
    - **Multi-View Stereo**: For higher accuracy, generate a detailed 3D point cloud or mesh.
        
    - **Cross-Section Analysis**: Slice the 3D model at the midpoint and compute the perimeter of the cross-section.
        
6. **Option 5 Alternative: Elliptical Approximation (If 3D reconstruction is too expensive)**
    
    - **Orthogonal Measurements**: From front and side views, measure the arm’s width (major axis) and depth (minor axis).
    - **Circumference Formula**: Use Ramanujan’s approximation for ellipse perimeter:
        
        C≈π(a+b)[1+3h10+4−3h],h=(a−b)2(a+b)2C≈π(a+b)[1+10+4−3h​3h​],h=(a+b)2(a−b)2​
        
        where aa and bb are semi-axes.
        
7. **Scale Conversion**
    - **Reference Object**: Use a known-size object in the scene (e.g., ruler) or the child’s height (anthropometric scaling) to convert pixel measurements to real-world units.
        

### **Tools & Libraries**

- **Pose Estimation**: OpenPose, MMPose.
    
- **3D Reconstruction**: COLMAP, OpenMVG + OpenMVS, MeshLab.
    
- **Segmentation**: OpenCV (thresholding), SAM VIT (Self-Attention Mechanism for Vision Transformers)
- **Math**: NumPy for elliptical approximation.
### **Validation**

- Cross-check results against manual measurements.
    
- Test consistency across different views and subjects.
    

### **Why This Works**

- **Multi-View Geometry**: Combines 2D data from multiple angles to infer 3D structure, bypassing occlusion issues.
    
- **Elliptical Assumption**: Simplifies measurement when 3D reconstruction is impractical, leveraging orthogonal views for accuracy.

### Current Progress

- [X] Automate segmentation for kiddies images
- [x] Pose Estimation using MMPose (Failed)
	- Failed due to cpython setup being time consuming
- [ ] Pose Estimation using OpenPose
	- Use ControlNets to setup openpose
- [ ] Midpoint estimation using Numpy
- [ ] Option 1: Circumference calculation using [Ramanujan's formula](https://www.researchgate.net/publication/2120536_Ramanujan's_Perimeter_of_an_Ellipse)
- [ ] Option 2: 3D Reconstruction using various angles
	- COLMAP
	- Meshlab
