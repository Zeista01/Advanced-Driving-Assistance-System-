# The Simulation - CARLA 

We are using a software called CARLA for various simulation validations.

## Car Follower in CARLA

Car follower is a simple system which follows a lead car while maintaining constant distance.

### Results:

![Car Follower](https://github.com/Zeista01/Advanced-Driving-Assistance-System-/blob/main/Results/carfollower.gif)

## Pure Pursuit in CARLA

Pure Pursuit Control (PPC) is a geometric path-tracking algorithm used in autonomous driving. This implementation applies PPC to a bicycle model in CARLA.

### Methodology
1. **Trajectory Generation**: Define waypoints as (x, y) coordinates using a dummy vehicle on autopilot.
2. **Visualization in CARLA**: Display trajectory markers using inbuilt CARLA functions.
3. **Vehicle Spawning**: Initialize a vehicle at the trajectory start.
4. **Camera Setup**: Mount a camera for observation.
5. **Pure Pursuit Controller**:
   - Set a lookahead distance to find the lookahead point on the trajectory.
  - Compute steering angle using:
  
  $$
  \delta = \tan^{-1}\left(\frac{2L \sin(\alpha)}{L_d}\right)
  $$

  - Apply steering command.

6. **Bicycle Model**:

   - Update vehicle state using:

  $$
  \dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{\theta} = \frac{v}{L} \tan(\delta)
  $$

7. **Control Loop**: Continuously update the lookahead based on the current position of the vehicle apply requires control.


### Results:

| ![Waypoints](https://github.com/Zeista01/Advanced-Driving-Assistance-System-/blob/main/Results/trajectory.png?raw=true) | ![PPC](https://github.com/Zeista01/Advanced-Driving-Assistance-System-/blob/main/Results/output2.gif?raw=true) |
|:---:|:---:|
| **The Trajectory** | **Tracing the Path by Pure Pursuit Control** |





### Conclusion
This PPC implementation enables path tracking in CARLA. Adjusting lookahead distance and speed optimizes performance.

*References:*
- CARLA Simulator Docs
- Pure Pursuit Algorithm



---

## Lane Segmentation using U-Net Architecture

Lane segmentation was implemented using a U-Net architecture to identify lane markings in images from the CARLA driving simulator, leveraging the "Lane Detection for CARLA Driving Simulator" dataset from Kaggle. The approach involved training a deep learning model to segment lane regions from camera images.

### Methodology
1. **Data Preparation**: Utilized the Kaggle dataset, containing RGB images and corresponding segmentation masks from CARLA, located in /kaggle/input/lane-detection-for-carla-driving-simulator/, loaded via PyTorch DataLoaders.
2. **Model Architecture**: Employed a U-Net with an encoder-decoder structure, featuring convolutional layers for feature extraction and upsampling layers for pixel-wise segmentation.
3. **Training**: Trained the model on a GPU (NVIDIA Tesla T4) using a combined Dice and Binary Cross-Entropy loss function, optimized with AdamW and a learning rate scheduler (ReduceLROnPlateau).
4. **Evaluation**: Assessed performance using metrics like Mean Intersection over Union (MIoU) and Dice Coefficient, with visualization of predicted masks against ground truth. Inference speed was measured, achieving an average of 0.0127 seconds per image (78.99 FPS).
5. **Model Saving**: Loaded the best model weights from best_model.pth and saved the final trained model's state dictionary to /kaggle/working/final_lane_segmentation_model.pth for reuse.

### Results
The trained U-Net model achieved promising lane segmentation performance. Visualizations showed accurate lane detection in validation images, with predicted masks closely aligning with ground truth. Quantitative metrics (e.g., MIoU and Dice) improved over epochs, indicating effective learning. The model demonstrated real-time capability with an estimated 78.99 FPS on the NVIDIA Tesla T4 GPU. (Note: Specific metric values and images to be added based on training output.)

![Results of U-Net](https://github.com/Zeista01/Advanced-Driving-Assistance-System-/blob/main/Results/MIoU%20and%20loss.png?raw=true)

![Results of U-Net](https://github.com/Zeista01/Advanced-Driving-Assistance-System-/blob/main/Results/lane%20seg.png?raw=true)


### Carla Simulation:

![Dashboard Camera](Results/output_combined.gif)

---
*References:*
- CARLA Simulator Docs
- Pure Pursuit Algorithm
- Lyft-Udacity Challenge Dataset
- U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)

