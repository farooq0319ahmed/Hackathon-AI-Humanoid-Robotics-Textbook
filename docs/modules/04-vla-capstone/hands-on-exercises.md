---
sidebar_position: 6
---

# Hands-On Exercises: Multi-Modal Human-Robot Interaction

## Exercise 1: Voice Command Processing Pipeline

### Objective
Implement a complete voice command processing pipeline that converts spoken commands into robot actions using OpenAI Whisper and ROS 2 integration.

### Prerequisites
- OpenAI API key with Whisper access
- ROS 2 Humble Hawksbill
- Audio input device (microphone)
- NVIDIA GPU for optimal processing

### Exercise Steps

1. **Setup Voice Recognition Node**
   ```python
   # Create a ROS 2 package for voice processing
   mkdir -p ~/vla_ws/src/voice_processing
   cd ~/vla_ws/src/voice_processing
   ros2 pkg create --build-type ament_python voice_ros

   # Install required dependencies
   pip3 install openai speechrecognition pyaudio
   ```

2. **Create Voice Recognition Node**
   ```python
   # voice_ros/voice_recognition_node.py
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from audio_common_msgs.msg import AudioData
   import openai
   import speech_recognition as sr
   import pyaudio
   import threading
   import queue

   class VoiceRecognitionNode(Node):
       def __init__(self):
           super().__init__('voice_recognition_node')

           # Get OpenAI API key from parameters
           self.api_key = self.declare_parameter('openai_api_key', '').value
           if self.api_key:
               openai.api_key = self.api_key
           else:
               self.get_logger().warn("No OpenAI API key provided, using offline recognition")

           # Audio processing setup
           self.recognizer = sr.Recognizer()
           self.recognizer.energy_threshold = 400
           self.microphone = sr.Microphone()

           # Create publishers
           self.transcript_pub = self.create_publisher(
               String, '/speech_transcript', 10
           )
           self.command_pub = self.create_publisher(
               String, '/parsed_command', 10
           )

           # Start audio listening thread
           self.audio_queue = queue.Queue()
           self.listening_thread = threading.Thread(target=self.listen_audio)
           self.listening_thread.start()

           self.get_logger().info("Voice recognition node initialized")

       def listen_audio(self):
           """Continuously listen for audio input"""
           with self.microphone as source:
               self.recognizer.adjust_for_ambient_noise(source)

           while rclpy.ok():
               try:
                   with self.microphone as source:
                       audio = self.recognizer.listen(source, timeout=1.0)

                   # Process audio with Whisper or offline recognition
                   transcript = self.recognize_speech(audio)
                   if transcript:
                       self.process_transcript(transcript)

               except sr.WaitTimeoutError:
                   continue
               except Exception as e:
                   self.get_logger().error(f"Audio processing error: {e}")

       def recognize_speech(self, audio):
           """Recognize speech using Whisper or offline method"""
           if self.api_key:  # Use OpenAI Whisper
               try:
                   # Save audio to temporary file for Whisper API
                   import io
                   audio_data = audio.get_wav_data()
                   audio_file = io.BytesIO(audio_data)
                   audio_file.name = "temp.wav"

                   transcript = openai.Audio.transcribe(
                       model="whisper-1",
                       file=audio_file
                   )
                   return transcript.text
               except Exception as e:
                   self.get_logger().error(f"Whisper API error: {e}")
                   return None
           else:  # Use offline recognition
               try:
                   return self.recognizer.recognize_sphinx(audio)
               except sr.UnknownValueError:
                   return None
               except sr.RequestError:
                   return None

       def process_transcript(self, transcript):
           """Process transcript and publish command"""
           self.get_logger().info(f"Recognized: {transcript}")

           # Publish transcript
           transcript_msg = String()
           transcript_msg.data = transcript
           self.transcript_pub.publish(transcript_msg)

           # Parse command and publish
           command = self.parse_command(transcript)
           if command:
               command_msg = String()
               command_msg.data = command
               self.command_pub.publish(command_msg)

       def parse_command(self, transcript):
           """Parse natural language command from transcript"""
           transcript_lower = transcript.lower()

           if "move to" in transcript_lower or "go to" in transcript_lower:
               return "NAVIGATE_TO_LOCATION"
           elif "pick up" in transcript_lower or "grasp" in transcript_lower:
               return "PICK_UP_OBJECT"
           elif "stop" in transcript_lower:
               return "STOP_ROBOT"
           elif "hello" in transcript_lower or "hi" in transcript_lower:
               return "GREET_HUMAN"

           return None

   def main(args=None):
       rclpy.init(args=args)
       node = VoiceRecognitionNode()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

3. **Create Launch File**
   ```xml
   <!-- voice_ros/launch/voice_recognition.launch.xml -->
   <launch>
       <node pkg="voice_ros" exec="voice_recognition_node" name="voice_recognition" output="screen">
           <param name="openai_api_key" value="$(var openai_api_key)"/>
       </node>
   </launch>
   ```

4. **Test Voice Recognition**
   ```bash
   # Build the package
   cd ~/vla_ws
   colcon build --packages-select voice_ros

   # Source the workspace
   source install/setup.bash

   # Launch with your OpenAI API key
   export OPENAI_API_KEY="your-api-key-here"
   ros2 launch voice_ros voice_recognition.launch.xml openai_api_key:=$OPENAI_API_KEY

   # Test by speaking commands and monitoring topics
   ros2 topic echo /speech_transcript
   ros2 topic echo /parsed_command
   ```

### Expected Outcomes
- Voice commands are captured and transcribed accurately
- Natural language commands are parsed into robot actions
- Transcripts and commands are published to appropriate ROS topics
- System handles both online (Whisper) and offline recognition

## Exercise 2: Cognitive Planning with LLM Integration

### Objective
Implement cognitive planning using OpenAI GPT models to decompose high-level commands into executable action sequences.

### Prerequisites
- OpenAI API key with GPT access
- ROS 2 Humble with action libraries
- Understanding of action execution frameworks

### Exercise Steps

1. **Create Cognitive Planning Node**
   ```python
   # cognitive_planning/cognitive_planner_node.py
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   from action_msgs.msg import GoalStatus
   import openai
   import json
   import asyncio
   from concurrent.futures import ThreadPoolExecutor

   class CognitivePlannerNode(Node):
       def __init__(self):
           super().__init__('cognitive_planner')

           # Initialize OpenAI client
           self.api_key = self.declare_parameter('openai_api_key', '').value
           if self.api_key:
               openai.api_key = self.api_key

           # Create subscribers
           self.command_sub = self.create_subscription(
               String, '/high_level_command', self.command_callback, 10
           )

           # Create publishers
           self.plan_pub = self.create_publisher(
               String, '/cognitive_plan', 10
           )
           self.action_pub = self.create_publisher(
               String, '/low_level_action', 10
           )
           self.status_pub = self.create_publisher(
               String, '/planning_status', 10
           )

           # Thread pool for async operations
           self.executor = ThreadPoolExecutor(max_workers=2)

           self.get_logger().info("Cognitive planner initialized")

       def command_callback(self, msg):
           """Process high-level command"""
           command = msg.data
           self.get_logger().info(f"Processing command: {command}")

           # Plan asynchronously
           future = asyncio.run_coroutine_threadsafe(
               self.plan_command_async(command),
               asyncio.get_event_loop()
           )

       async def plan_command_async(self, command):
           """Plan command using LLM"""
           try:
               plan = await self.generate_plan_with_llm(command)

               if plan:
                   # Publish plan
                   plan_msg = String()
                   plan_msg.data = json.dumps(plan)
                   self.plan_pub.publish(plan_msg)

                   # Execute plan
                   await self.execute_plan(plan)

           except Exception as e:
               self.get_logger().error(f"Planning error: {e}")
               self.publish_status(f"Planning failed: {str(e)}")

       async def generate_plan_with_llm(self, command):
           """Generate plan using LLM"""
           system_prompt = """
           You are a cognitive planning system for a humanoid robot. Decompose the given command
           into a sequence of executable actions. Consider the robot's capabilities:

           Navigation: Move to locations, avoid obstacles, path planning
           Manipulation: Pick up objects, place objects, open doors
           Perception: Detect objects, recognize humans, map environment
           Communication: Speak, gesture, acknowledge

           Available actions:
           - NAVIGATE_TO_LOCATION: Move to specific location
           - DETECT_OBJECT: Find specific object in environment
           - PICK_UP_OBJECT: Grasp an object
           - PLACE_OBJECT: Release object at location
           - FOLLOW_HUMAN: Follow a person
           - SPEAK: Verbal communication
           - GREET_HUMAN: Acknowledge and greet person

           Response format: JSON with 'actions' array containing action objects with 'name' and 'parameters'.
           """

           try:
               response = await openai.ChatCompletion.acreate(
                   model="gpt-3.5-turbo",
                   messages=[
                       {"role": "system", "content": system_prompt},
                       {"role": "user", "content": f"Command: {command}"}
                   ],
                   temperature=0.1,
                   max_tokens=500
               )

               plan_json = response.choices[0].message.content
               # Clean up response if it contains markdown formatting
               if plan_json.startswith("```json"):
                   plan_json = plan_json[7:-3]  # Remove markdown markers
               elif plan_json.startswith("```"):
                   plan_json = plan_json[3:-3]

               plan = json.loads(plan_json)
               return plan

           except Exception as e:
               self.get_logger().error(f"LLM planning error: {e}")
               return self.fallback_plan(command)

       def fallback_plan(self, command):
           """Create fallback plan when LLM fails"""
           command_lower = command.lower()

           if "navigate" in command_lower or "go to" in command_lower:
               return {
                   "actions": [
                       {"name": "NAVIGATE_TO_LOCATION", "parameters": {"target": "unknown"}}
                   ]
               }
           elif "pick up" in command_lower or "grasp" in command_lower:
               return {
                   "actions": [
                       {"name": "DETECT_OBJECT", "parameters": {"object": "unknown"}},
                       {"name": "PICK_UP_OBJECT", "parameters": {"object": "unknown"}}
                   ]
               }
           else:
               return {
                   "actions": [
                       {"name": "SPEAK", "parameters": {"text": f"Processing: {command}"}}
                   ]
               }

       async def execute_plan(self, plan):
           """Execute the planned actions"""
           actions = plan.get('actions', [])
           self.get_logger().info(f"Executing plan with {len(actions)} actions")

           for i, action in enumerate(actions):
               self.get_logger().info(f"Executing action {i+1}/{len(actions)}: {action['name']}")

               # Publish action
               action_msg = String()
               action_msg.data = json.dumps(action)
               self.action_pub.publish(action_msg)

               # Wait for action completion (simplified)
               await asyncio.sleep(2.0)  # Simulate action execution time

           self.publish_status("Plan completed successfully")

       def publish_status(self, status):
           """Publish planning status"""
           status_msg = String()
           status_msg.data = status
           self.status_pub.publish(status_msg)

   def main(args=None):
       rclpy.init(args=args)
       node = CognitivePlannerNode()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Create Launch File**
   ```xml
   <!-- cognitive_planning/launch/cognitive_planning.launch.xml -->
   <launch>
       <node pkg="cognitive_planning" exec="cognitive_planner_node" name="cognitive_planner" output="screen">
           <param name="openai_api_key" value="$(var openai_api_key)"/>
       </node>
   </launch>
   ```

3. **Test Cognitive Planning**
   ```bash
   # Build the package
   cd ~/vla_ws
   colcon build --packages-select cognitive_planning

   # Source and launch
   source install/setup.bash
   ros2 launch cognitive_planning cognitive_planning.launch.xml openai_api_key:=$OPENAI_API_KEY

   # Test with sample commands
   ros2 topic pub /high_level_command std_msgs/String "data: 'go to the kitchen and pick up the cup'"
   ```

### Expected Outcomes
- High-level commands are decomposed into action sequences
- LLM generates appropriate action plans for robot capabilities
- Plans are executed in proper sequence
- System provides fallback planning when LLM fails

## Exercise 3: Multi-Modal Perception Integration

### Objective
Integrate visual, audio, and gesture perception into a unified multi-modal understanding system.

### Prerequisites
- Camera for visual input
- Microphone for audio input
- OpenCV and MediaPipe for gesture recognition
- ROS 2 Humble

### Exercise Steps

1. **Create Multi-Modal Perception Node**
   ```python
   # multi_modal_perception/multi_modal_node.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, AudioData
   from std_msgs.msg import String
   from cv_bridge import CvBridge
   import cv2
   import mediapipe as mp
   import numpy as np
   import threading
   import queue

   class MultiModalPerceptionNode(Node):
       def __init__(self):
           super().__init__('multi_modal_perception')

           # Initialize CV bridge
           self.bridge = CvBridge()

           # Initialize MediaPipe for gesture recognition
           self.mp_hands = mp.solutions.hands
           self.mp_pose = mp.solutions.pose
           self.hands = self.mp_hands.Hands(
               static_image_mode=False,
               max_num_hands=2,
               min_detection_confidence=0.7
           )
           self.pose = self.mp_pose.Pose(
               static_image_mode=False,
               model_complexity=1,
               min_detection_confidence=0.7
           )

           # Create subscribers
           self.image_sub = self.create_subscription(
               Image, '/camera/rgb/image_raw', self.image_callback, 10
           )
           self.audio_sub = self.create_subscription(
               AudioData, '/audio_input', self.audio_callback, 10
           )

           # Create publishers
           self.perception_pub = self.create_publisher(
               String, '/multi_modal_perception', 10
           )
           self.gesture_pub = self.create_publisher(
               String, '/detected_gestures', 10
           )
           self.visual_pub = self.create_publisher(
               String, '/visual_analysis', 10
           )

           # Processing queues
           self.image_queue = queue.Queue(maxsize=5)
           self.audio_queue = queue.Queue(maxsize=5)

           # Processing threads
           self.image_thread = threading.Thread(target=self.process_images)
           self.image_thread.start()

           self.get_logger().info("Multi-modal perception node initialized")

       def image_callback(self, msg):
           """Process image data"""
           try:
               cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

               # Add to processing queue
               try:
                   self.image_queue.put_nowait(cv_image)
               except queue.Full:
                   # Drop oldest if queue full
                   try:
                       self.image_queue.get_nowait()
                   except queue.Empty:
                       pass
                   self.image_queue.put_nowait(cv_image)

           except Exception as e:
               self.get_logger().error(f"Image processing error: {e}")

       def audio_callback(self, msg):
           """Process audio data"""
           try:
               # Process audio data
               audio_features = self.extract_audio_features(msg.data)

               # Publish audio analysis
               audio_msg = String()
               audio_msg.data = str(audio_features)
               self.perception_pub.publish(audio_msg)

           except Exception as e:
               self.get_logger().error(f"Audio processing error: {e}")

       def process_images(self):
           """Process images in separate thread"""
           while rclpy.ok():
               try:
                   cv_image = self.image_queue.get(timeout=1.0)

                   # Process with MediaPipe
                   rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                   hand_results = self.hands.process(rgb_image)
                   pose_results = self.pose.process(rgb_image)

                   # Extract features
                   features = {
                       'hands': self.extract_hand_features(hand_results),
                       'pose': self.extract_pose_features(pose_results),
                       'faces': self.detect_faces(cv_image)
                   }

                   # Publish features
                   features_msg = String()
                   features_msg.data = str(features)
                   self.visual_pub.publish(features_msg)

                   # Detect and publish gestures
                   gestures = self.detect_gestures(features)
                   if gestures:
                       gesture_msg = String()
                       gesture_msg.data = str(gestures)
                       self.gesture_pub.publish(gesture_msg)

                   # Create multi-modal fusion
                   multi_modal_data = {
                       'visual_features': features,
                       'gestures': gestures,
                       'timestamp': self.get_clock().now().nanoseconds / 1e9
                   }

                   fusion_msg = String()
                   fusion_msg.data = str(multi_modal_data)
                   self.perception_pub.publish(fusion_msg)

               except queue.Empty:
                   continue
               except Exception as e:
                   self.get_logger().error(f"Image processing thread error: {e}")

       def extract_hand_features(self, results):
           """Extract hand features from MediaPipe results"""
           if not results.multi_hand_landmarks:
               return []

           hands_data = []
           for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
               hand_data = {
                   'handedness': results.multi_handedness[i].classification[0].index,
                   'landmarks': []
               }

               for landmark in hand_landmarks.landmark:
                   hand_data['landmarks'].append({
                       'x': landmark.x,
                       'y': landmark.y,
                       'z': landmark.z if landmark.z else 0.0
                   })

               hands_data.append(hand_data)

           return hands_data

       def extract_pose_features(self, results):
           """Extract pose features from MediaPipe results"""
           if not results.pose_landmarks:
               return {}

           pose_data = {'landmarks': []}
           for landmark in results.pose_landmarks.landmark:
               pose_data['landmarks'].append({
                   'x': landmark.x,
                   'y': landmark.y,
                   'z': landmark.z if landmark.z else 0.0,
                   'visibility': landmark.visibility
               })

           return pose_data

       def detect_faces(self, image):
           """Detect faces in image"""
           gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
           faces = face_cascade.detectMultiScale(gray, 1.1, 4)
           return [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} for x, y, w, h in faces]

       def detect_gestures(self, features):
           """Detect specific gestures from features"""
           gestures = []

           # Simple gesture detection based on hand positions
           for hand_data in features['hands']:
               landmarks = hand_data['landmarks']
               if len(landmarks) >= 5:
                   # Check for "stop" gesture (palm facing forward)
                   wrist = landmarks[0]
                   thumb = landmarks[1]
                   index_finger = landmarks[5]

                   # Calculate distances to detect gesture
                   thumb_index_dist = np.sqrt(
                       (thumb['x'] - index_finger['x'])**2 +
                       (thumb['y'] - index_finger['y'])**2 +
                       (thumb['z'] - index_finger['z'])**2
                   )

                   if thumb_index_dist > 0.1:  # Open palm
                       gestures.append({
                           'type': 'stop',
                           'confidence': 0.8,
                           'handedness': 'right' if hand_data['handedness'] == 1 else 'left'
                       })

           return gestures

       def extract_audio_features(self, audio_data):
           """Extract basic audio features"""
           # This is a simplified example
           # In practice, would use proper audio feature extraction
           return {
               'energy': sum(b*b for b in audio_data) / len(audio_data) if audio_data else 0,
               'length': len(audio_data),
               'timestamp': self.get_clock().now().nanoseconds / 1e9
           }

   def main(args=None):
       rclpy.init(args=args)
       node = MultiModalPerceptionNode()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Create Launch File**
   ```xml
   <!-- multi_modal_perception/launch/multi_modal.launch.xml -->
   <launch>
       <node pkg="multi_modal_perception" exec="multi_modal_node" name="multi_modal_perception" output="screen">
       </node>
   </launch>
   ```

3. **Test Multi-Modal Perception**
   ```bash
   # Build the package
   cd ~/vla_ws
   colcon build --packages-select multi_modal_perception

   # Source and launch
   source install/setup.bash
   ros2 launch multi_modal_perception multi_modal.launch.xml

   # Monitor the outputs
   ros2 topic echo /multi_modal_perception
   ros2 topic echo /detected_gestures
   ros2 topic echo /visual_analysis
   ```

### Expected Outcomes
- Visual features are extracted from camera input
- Hand gestures are detected using MediaPipe
- Audio features are processed from microphone input
- Multi-modal data is fused and published
- System can detect and respond to various human gestures

## Exercise 4: Capstone Integration Challenge

### Objective
Integrate all VLA components into a complete autonomous humanoid system that can respond to voice commands, perceive its environment, plan actions, and execute tasks.

### Prerequisites
- All previous exercises completed
- Simulation environment (Gazebo or Isaac Sim)
- Complete ROS 2 workspace with all modules

### Exercise Steps

1. **Create Capstone Integration Node**
   ```python
   # capstone_integration/capstone_system.py
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String, Bool
   from geometry_msgs.msg import Twist, PoseStamped
   from sensor_msgs.msg import Image, Imu, LaserScan
   import json
   import asyncio
   from concurrent.futures import ThreadPoolExecutor

   class CapstoneSystemNode(Node):
       def __init__(self):
           super().__init__('capstone_system')

           # System state
           self.system_state = {
               'active': False,
               'current_task': None,
               'safety_status': 'SAFE',
               'battery_level': 100.0
           }

           # Create subscribers for all inputs
           self.voice_sub = self.create_subscription(
               String, '/speech_transcript', self.voice_command_callback, 10
           )
           self.perception_sub = self.create_subscription(
               String, '/multi_modal_perception', self.perception_callback, 10
           )
           self.emergency_sub = self.create_subscription(
               Bool, '/emergency_stop', self.emergency_callback, 10
           )

           # Create publishers for all outputs
           self.navigation_pub = self.create_publisher(
               PoseStamped, '/goal_pose', 10
           )
           self.cmd_vel_pub = self.create_publisher(
               Twist, '/cmd_vel', 10
           )
           self.speech_pub = self.create_publisher(
               String, '/robot_speech', 10
           )
           self.system_status_pub = self.create_publisher(
               String, '/system_status', 10
           )

           # Thread pool for async operations
           self.executor = ThreadPoolExecutor(max_workers=4)

           # Initialize system components
           self.initialize_components()

           # Start main control loop
           self.control_timer = self.create_timer(0.1, self.main_control_loop)

           self.get_logger().info("Capstone system initialized and running")

       def initialize_components(self):
           """Initialize all system components"""
           self.get_logger().info("Initializing capstone system components...")
           # This would initialize all the subsystems
           self.system_state['active'] = True

       def voice_command_callback(self, msg):
           """Process voice commands"""
           try:
               command = json.loads(msg.data) if msg.data.startswith('{') else {'text': msg.data}
               self.get_logger().info(f"Received voice command: {command}")

               # Process command through cognitive planning
               asyncio.run_coroutine_threadsafe(
                   self.process_voice_command(command),
                   asyncio.get_event_loop()
               )

           except Exception as e:
               self.get_logger().error(f"Voice command processing error: {e}")

       async def process_voice_command(self, command):
           """Process voice command asynchronously"""
           command_text = command.get('text', command) if isinstance(command, dict) else command

           # Simple command interpretation
           if "go to" in command_text.lower() or "navigate" in command_text.lower():
               await self.execute_navigation_command(command_text)
           elif "pick up" in command_text.lower() or "grasp" in command_text.lower():
               await self.execute_manipulation_command(command_text)
           elif "stop" in command_text.lower():
               await self.execute_stop_command()
           else:
               await self.execute_generic_command(command_text)

       async def execute_navigation_command(self, command):
           """Execute navigation command"""
           # Extract location from command
           location = self.extract_location_from_command(command)
           if location:
               self.get_logger().info(f"Navigating to {location}")

               # Create and send navigation goal
               goal_pose = self.create_location_pose(location)
               self.navigation_pub.publish(goal_pose)

               # Provide feedback
               feedback = f"Navigating to {location}"
               self.speak_response(feedback)

       async def execute_manipulation_command(self, command):
           """Execute manipulation command"""
           # Extract object from command
           obj = self.extract_object_from_command(command)
           if obj:
               self.get_logger().info(f"Attempting to manipulate {obj}")

               # This would involve perception, planning, and execution
               feedback = f"Looking for {obj} to manipulate"
               self.speak_response(feedback)

       async def execute_stop_command(self):
           """Execute stop command"""
           self.get_logger().info("Stopping robot")
           cmd_vel = Twist()
           self.cmd_vel_pub.publish(cmd_vel)
           self.speak_response("Robot stopped")

       async def execute_generic_command(self, command):
           """Execute generic command"""
           self.get_logger().info(f"Processing generic command: {command}")
           self.speak_response(f"Processing command: {command}")

       def perception_callback(self, msg):
           """Process perception data"""
           try:
               perception_data = json.loads(msg.data) if msg.data.startswith('{') else {'data': msg.data}
               self.get_logger().info(f"Perception update: {len(str(perception_data))} chars")

               # Update system state based on perception
               self.update_system_with_perception(perception_data)

           except Exception as e:
               self.get_logger().error(f"Perception processing error: {e}")

       def update_system_with_perception(self, perception_data):
           """Update system state with perception data"""
           # This would update maps, detect obstacles, recognize objects, etc.
           pass

       def emergency_callback(self, msg):
           """Handle emergency stop"""
           if msg.data:
               self.get_logger().error("EMERGENCY STOP ACTIVATED")
               self.system_state['safety_status'] = 'EMERGENCY'
               self.emergency_stop()
           else:
               self.system_state['safety_status'] = 'SAFE'

       def emergency_stop(self):
           """Execute emergency stop"""
           cmd_vel = Twist()
           self.cmd_vel_pub.publish(cmd_vel)
           self.speak_response("Emergency stop activated")

       def main_control_loop(self):
           """Main system control loop"""
           # Update system status
           status_msg = String()
           status_msg.data = json.dumps({
               'state': self.system_state,
               'timestamp': self.get_clock().now().nanoseconds / 1e9
           })
           self.system_status_pub.publish(status_msg)

           # Monitor system health
           self.monitor_system_health()

       def monitor_system_health(self):
           """Monitor system health and safety"""
           # Check battery level
           if self.system_state['battery_level'] < 10:
               self.system_state['safety_status'] = 'BATTERY_LOW'
               self.return_to_base()

       def return_to_base(self):
           """Return to charging station"""
           self.get_logger().info("Returning to base due to low battery")
           # Implementation would navigate to base station

       def create_location_pose(self, location):
           """Create pose for known location"""
           # This would map location names to coordinates
           # For this example, using mock coordinates
           locations = {
               'kitchen': (1.0, 0.0, 0.0),
               'living room': (0.0, 1.0, 0.0),
               'bedroom': (-1.0, 0.0, 0.0),
               'office': (0.0, -1.0, 0.0)
           }

           coords = locations.get(location.lower(), (0.0, 0.0, 0.0))
           pose = PoseStamped()
           pose.header.frame_id = 'map'
           pose.header.stamp = self.get_clock().now().to_msg()
           pose.pose.position.x = coords[0]
           pose.pose.position.y = coords[1]
           pose.pose.position.z = 0.0
           pose.pose.orientation.w = 1.0

           return pose

       def extract_location_from_command(self, command):
           """Extract location from command"""
           command_lower = command.lower()
           locations = ['kitchen', 'living room', 'bedroom', 'office', 'bathroom']
           for loc in locations:
               if loc in command_lower:
                   return loc
           return None

       def extract_object_from_command(self, command):
           """Extract object from command"""
           command_lower = command.lower()
           objects = ['cup', 'book', 'ball', 'box', 'bottle', 'phone']
           for obj in objects:
               if obj in command_lower:
                   return obj
           return None

       def speak_response(self, text):
           """Publish speech response"""
           speech_msg = String()
           speech_msg.data = text
           self.speech_pub.publish(speech_msg)

   def main(args=None):
       rclpy.init(args=args)
       node = CapstoneSystemNode()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Create Complete Launch File**
   ```xml
   <!-- capstone_integration/launch/capstone_system.launch.xml -->
   <launch>
       <!-- Launch all system components -->
       <include file="$(find-pkg-share voice_ros)/launch/voice_recognition.launch.xml">
           <arg name="openai_api_key" value="$(var openai_api_key)"/>
       </include>

       <include file="$(find-pkg-share cognitive_planning)/launch/cognitive_planning.launch.xml">
           <arg name="openai_api_key" value="$(var openai_api_key)"/>
       </include>

       <include file="$(find-pkg-share multi_modal_perception)/launch/multi_modal.launch.xml"/>

       <!-- Launch capstone integration -->
       <node pkg="capstone_integration" exec="capstone_system" name="capstone_system" output="screen"/>
   </launch>
   ```

3. **Test Complete System**
   ```bash
   # Build all packages
   cd ~/vla_ws
   colcon build

   # Source and launch complete system
   source install/setup.bash
   ros2 launch capstone_integration capstone_system.launch.xml openai_api_key:=$OPENAI_API_KEY

   # Test the complete system by speaking commands like:
   # "Go to the kitchen"
   # "Pick up the cup"
   # "Stop the robot"

   # Monitor all system topics
   ros2 topic echo /system_status
   ros2 topic echo /robot_speech
   ros2 topic echo /goal_pose
   ```

### Expected Outcomes
- Complete VLA system processes voice commands and executes actions
- Multi-modal perception provides environmental awareness
- Cognitive planning decomposes high-level commands
- Navigation system moves robot to desired locations
- Safety systems monitor and respond to emergency conditions
- System demonstrates end-to-end autonomous behavior

## Exercise 5: Performance Optimization and Testing

### Objective
Optimize the complete VLA system for performance and conduct comprehensive testing.

### Exercise Steps

1. **Create Performance Monitoring Node**
   ```python
   # performance_monitor/performance_node.py
   import rclpy
   from rclpy.node import Node
   from std_msgs.msg import String
   import psutil
   import time
   import json

   class PerformanceMonitorNode(Node):
       def __init__(self):
           super().__init__('performance_monitor')

           self.metrics_pub = self.create_publisher(
               String, '/system_metrics', 10
           )

           # Start monitoring timer
           self.monitor_timer = self.create_timer(1.0, self.monitor_performance)

           self.get_logger().info("Performance monitor started")

       def monitor_performance(self):
           """Monitor system performance metrics"""
           metrics = {
               'cpu_percent': psutil.cpu_percent(interval=1),
               'memory_percent': psutil.virtual_memory().percent,
               'disk_usage': psutil.disk_usage('/').percent,
               'process_count': len(psutil.pids()),
               'timestamp': time.time()
           }

           # Publish metrics
           metrics_msg = String()
           metrics_msg.data = json.dumps(metrics)
           self.metrics_pub.publish(metrics_msg)

           # Log if thresholds exceeded
           if metrics['cpu_percent'] > 80:
               self.get_logger().warn(f"High CPU usage: {metrics['cpu_percent']}%")
           if metrics['memory_percent'] > 80:
               self.get_logger().warn(f"High memory usage: {metrics['memory_percent']}%")

   def main(args=None):
       rclpy.init(args=args)
       node = PerformanceMonitorNode()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Create System Test Suite**
   ```python
   # system_tests/test_suite.py
   import unittest
   import rclpy
   from std_msgs.msg import String
   import time

   class TestCapstoneSystem(unittest.TestCase):
       @classmethod
       def setUpClass(cls):
           rclpy.init()

       @classmethod
       def tearDownClass(cls):
           rclpy.shutdown()

       def test_voice_recognition(self):
           """Test voice recognition functionality"""
           node = rclpy.create_node('test_voice_node')
           publisher = node.create_publisher(String, '/test_input', 10)
           subscription = node.create_subscription(
               String, '/speech_transcript', lambda msg: setattr(self, 'transcript', msg.data), 10
           )

           # Publish test audio
           test_msg = String()
           test_msg.data = "test audio input"
           publisher.publish(test_msg)

           # Wait for response
           timeout = time.time() + 5.0  # 5 second timeout
           while not hasattr(self, 'transcript') and time.time() < timeout:
               rclpy.spin_once(node, timeout_sec=0.1)

           self.assertTrue(hasattr(self, 'transcript'))
           node.destroy_node()

       def test_cognitive_planning(self):
           """Test cognitive planning functionality"""
           node = rclpy.create_node('test_planning_node')
           publisher = node.create_publisher(String, '/high_level_command', 10)
           subscription = node.create_subscription(
               String, '/cognitive_plan', lambda msg: setattr(self, 'plan', msg.data), 10
           )

           # Publish test command
           test_cmd = String()
           test_cmd.data = "go to kitchen"
           publisher.publish(test_cmd)

           # Wait for plan
           timeout = time.time() + 5.0
           while not hasattr(self, 'plan') and time.time() < timeout:
               rclpy.spin_once(node, timeout_sec=0.1)

           self.assertTrue(hasattr(self, 'plan'))
           node.destroy_node()

       def test_system_integration(self):
           """Test complete system integration"""
           # This would test the full pipeline
           self.assertTrue(True)  # Placeholder for integration test

   if __name__ == '__main__':
       unittest.main()
   ```

3. **Run Performance Tests**
   ```bash
   # Run the test suite
   python3 -m pytest system_tests/test_suite.py -v

   # Monitor system during operation
   ros2 run performance_monitor performance_node
   ```

### Expected Outcomes
- System performance is monitored and optimized
- CPU and memory usage remain within acceptable limits
- All system components function correctly together
- Comprehensive testing validates system functionality
- Performance bottlenecks are identified and addressed

## Evaluation Criteria

### Performance Metrics
- **Voice Recognition**: Accuracy > 80%, latency < 2 seconds
- **Cognitive Planning**: Plan generation time < 5 seconds
- **Multi-Modal Perception**: Real-time processing (30 FPS)
- **System Integration**: End-to-end task completion > 90%

### Validation Tests
1. **Voice Command Test**: System correctly interprets and executes voice commands
2. **Navigation Test**: Robot successfully navigates to specified locations
3. **Manipulation Test**: Robot detects and manipulates objects when commanded
4. **Safety Test**: Emergency stop functions properly in all modes
5. **Integration Test**: Complete VLA pipeline works end-to-end

### Troubleshooting Guide
- Check ROS topic connections and message formats
- Verify OpenAI API key and permissions
- Monitor system resource usage
- Validate sensor calibration and data quality
- Test individual components before system integration