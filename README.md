# The Simulation - CARLA 

We are using a software called CARLA for various simulation validations.

## Car Follower in CARLA

Car follower is a simple system which follows a lead car while maintaining constant distance.

### Results:
https://drive.google.com/file/d/1A9dlfvb626ndAPMwntn-QjQ-9_mYQjx7/view?usp=drive_link


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


![PPC](https://github.com/Zeista01/Advanced-Driving-Assistance-System-/blob/main/Results/ppc.gif)


### Conclusion
This PPC implementation enables path tracking in CARLA. Adjusting lookahead distance and speed optimizes performance.

---
*References:*
- CARLA Simulator Docs
- Pure Pursuit Algorithm

