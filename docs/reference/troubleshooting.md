---
sidebar_position: 3
---

# Troubleshooting Guide

This guide provides solutions for common issues encountered when working with ROS 2 and the humanoid robotics examples.

## ROS 2 Setup Issues

### ROS 2 Commands Not Found

**Problem:** Commands like `ros2 run`, `ros2 topic`, etc. return "command not found"

**Solutions:**
1. **Source ROS 2 installation:**
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Add to bashrc for permanent setup:**
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Verify ROS 2 installation:**
   ```bash
   printenv | grep ROS
   ```

### Python Package Import Errors

**Problem:** Python nodes fail with import errors like "No module named 'rclpy'"

**Solutions:**
1. **Check Python environment:**
   ```bash
   python3 -c "import rclpy; print(rclpy.__version__)"
   ```

2. **Activate virtual environment if used:**
   ```bash
   source venv/bin/activate
   ```

3. **Install missing packages:**
   ```bash
   pip3 install rclpy
   # Or install via apt for system packages:
   sudo apt install python3-ros-foxy-rclpy
   ```

### Package Build Issues

**Problem:** `colcon build` fails with compilation errors

**Solutions:**
1. **Clean previous build:**
   ```bash
   rm -rf build/ install/ log/
   ```

2. **Check for missing dependencies:**
   ```bash
   rosdep install --from-paths src --ignore-src -r -y
   ```

3. **Build with more verbose output:**
   ```bash
   colcon build --packages-select package_name --event-handlers console_direct+
   ```

## Node Communication Issues

### Nodes Cannot Communicate

**Problem:** Publisher and subscriber nodes cannot communicate with each other

**Solutions:**
1. **Verify topic names match:**
   ```bash
   ros2 topic list
   ros2 topic info /topic_name
   ```

2. **Check for matching message types:**
   ```bash
   ros2 topic type /topic_name
   ```

3. **Verify nodes are on the same ROS domain:**
   ```bash
   echo $ROS_DOMAIN_ID
   # Set if needed: export ROS_DOMAIN_ID=0
   ```

4. **Check network configuration:**
   ```bash
   # For multi-machine communication
   export ROS_LOCALHOST_ONLY=0
   export ROS_IP=your_ip_address
   ```

### Service Not Available

**Problem:** Service client reports "service not available"

**Solutions:**
1. **Verify service server is running:**
   ```bash
   ros2 service list
   ros2 service info /service_name
   ```

2. **Check service type matches:**
   ```bash
   ros2 service type /service_name
   ```

3. **Wait for service in client code:**
   ```python
   while not client.wait_for_service(timeout_sec=1.0):
       node.get_logger().info('Service not available, waiting again...')
   ```

## Launch File Issues

### Launch File Fails to Start

**Problem:** `ros2 launch` command fails with errors

**Solutions:**
1. **Verify package name and launch file exist:**
   ```bash
   ros2 pkg executables package_name
   ```

2. **Check launch file syntax:**
   ```bash
   python3 -m py_compile path/to/launch/file.py
   ```

3. **Run with verbose output:**
   ```bash
   ros2 launch package_name launch_file.py --event-handlers console_direct+
   ```

## Parameter Issues

### Parameters Not Loading

**Problem:** Node parameters are not loaded from YAML file

**Solutions:**
1. **Verify parameter file path:**
   ```bash
   # Use absolute path or package-relative path
   ros2 run package node --ros-args --params-file $(ros2 pkg prefix package_name)/share/package_name/config/params.yaml
   ```

2. **Check parameter file syntax:**
   ```bash
   python3 -c "import yaml; print(yaml.safe_load(open('params.yaml')))"
   ```

3. **Verify node declares parameters:**
   ```python
   self.declare_parameter('param_name', default_value)
   ```

## Gazebo and Simulation Issues

### Gazebo Won't Start

**Problem:** Gazebo simulation fails to launch

**Solutions:**
1. **Check graphics drivers:**
   ```bash
   glxinfo | grep "OpenGL renderer"
   ```

2. **Install GPU drivers if needed:**
   ```bash
   sudo apt install nvidia-driver-XXX  # For NVIDIA GPUs
   ```

3. **Set proper display settings:**
   ```bash
   export DISPLAY=:0
   ```

### URDF Model Issues

**Problem:** Robot model doesn't appear correctly in Gazebo

**Solutions:**
1. **Validate URDF syntax:**
   ```bash
   check_urdf /path/to/robot.urdf
   ```

2. **Check for missing meshes:**
   ```bash
   # Verify mesh paths are correct and files exist
   find . -name "*.stl" -o -name "*.dae" -o -name "*.obj"
   ```

3. **Test URDF in RViz:**
   ```bash
   ros2 run rviz2 rviz2
   # Add RobotModel display and set robot description
   ```

## Performance Issues

### High CPU Usage

**Problem:** ROS 2 nodes consume excessive CPU resources

**Solutions:**
1. **Reduce timer frequencies:**
   ```python
   # Instead of 100Hz, try 10Hz for non-critical tasks
   self.timer = self.create_timer(0.1, callback)  # 10Hz
   ```

2. **Optimize message publishing:**
   ```python
   # Don't publish if data hasn't changed significantly
   if abs(new_value - old_value) > threshold:
       publisher.publish(msg)
   ```

3. **Use appropriate QoS settings:**
   ```python
   # For high-frequency topics, consider reducing history depth
   qos = QoSProfile(depth=1)
   ```

## Memory Issues

### Memory Leaks

**Problem:** Process memory usage grows over time

**Solutions:**
1. **Properly destroy nodes:**
   ```python
   def destroy_node(self):
       # Clean up resources before destroying
       self.destroy_publisher(publisher)
       self.destroy_subscription(subscription)
       super().destroy_node()
   ```

2. **Monitor memory usage:**
   ```bash
   # Monitor specific process
   pidstat -p $(pgrep -f node_name) -r 1
   ```

## Network Issues

### Multi-Robot Communication

**Problem:** Multiple robots interfere with each other

**Solutions:**
1. **Use different ROS domain IDs:**
   ```bash
   # Robot 1
   export ROS_DOMAIN_ID=0
   # Robot 2
   export ROS_DOMAIN_ID=1
   ```

2. **Use namespaces:**
   ```bash
   ros2 run package node --ros-args --remap __ns:=/robot1
   ```

## Common Error Messages

### "Unable to load plugin" Error

**Problem:** Error loading ROS 2 plugins

**Solutions:**
1. **Update plugin cache:**
   ```bash
   update-ros-package-index
   ```

2. **Check plugin installation:**
   ```bash
   # For RViz plugins
   apt list --installed | grep rviz
   ```

### "Clock" Topic Issues

**Problem:** Warning about /clock topic in simulation

**Solutions:**
1. **Use sim time parameter:**
   ```python
   self.use_sim_time = self.get_parameter('use_sim_time').value
   ```

2. **Launch with use_sim_time:**
   ```bash
   ros2 param set /node_name use_sim_time true
   ```

## Testing and Validation

### Running Validation Scripts

**Problem:** Validation scripts fail

**Solutions:**
1. **Ensure all dependencies are installed:**
   ```bash
   rosdep install --from-paths . --ignore-src -r -y
   ```

2. **Run with proper environment:**
   ```bash
   source install/setup.bash
   python3 test_script.py
   ```

3. **Check for race conditions in tests:**
   ```python
   # Add sufficient delays for message propagation
   time.sleep(1.0)
   ```

## Getting Help

### Useful Commands for Debugging

```bash
# List all active nodes
ros2 node list

# Get information about a specific node
ros2 node info /node_name

# Monitor topic statistics
ros2 topic hz /topic_name

# Echo topic messages
ros2 topic echo /topic_name

# Call a service
ros2 service call /service_name service_type "{request: values}"

# List all parameters for a node
ros2 param list /node_name
```

### Additional Resources

- [ROS 2 Documentation](https://docs.ros.org/)
- [ROS Answers](https://answers.ros.org/)
- [ROS Discourse](https://discourse.ros.org/)
- [Gazebo Documentation](http://gazebosim.org/tutorials)