try:
    from ultralytics import YOLO
    import cv2
    import os
    import glob
    import random
    import time
    import numpy as np
    from datetime import datetime
    from typing import Dict, List, Tuple, Optional, Union
    import logging
    from dataclasses import dataclass
    from collections import defaultdict, Counter
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install required packages: pip install ultralytics opencv-python numpy")
    exit(1)

# Configuration
@dataclass
class Config:
    model_path: str = "yolo11x.pt"
    confidence_threshold: float = 0.5
    capture_interval: int = 8  # seconds
    display_duration: int = 4  # seconds
    camera_index: int = 2
    stream_url: Optional[str] = None  # e.g., rtsp/http URL from phone camera app
    resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30

# Initialize configuration
config = Config()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load detection model with error handling
try:
    model = YOLO(config.model_path)
    logger.info(f"Model loaded successfully: {config.model_path}")
except Exception as e:
    logger.error(f"Failed to load model {config.model_path}: {e}")
    exit(1)

# Object hierarchy mapping - optimized for fast lookup
object_hierarchy = {
    "stick": ["skis","baseball bat", "snowboard","tennis racket"],
    "ball":["frisbee","sports ball"],
    "person":["person"],
    "animal": ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
    "vehicle": ["car","airplane","bus","train","truck","boat"],
    "bike": ["bicycle","motorcycle"],
    "backpack": ["backpack","suitcase"],
    "umbrella": ["umbrella"],
    "handbag": ["handbag"],
    "tie": ["tie"],
    "miscellaneous":["bench","traffic light","fire hydrant","stop sign","parking meter","kite","skateboard","surfboard","vase","teddy bear","hair drier","toothbrush"],
    "bottle":["bottle"],
    "cup":["cup","wine glass"],
    "utensiles":["fork","knife","spoon"],
    "bowl":["bowl"],
    "food":["banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake"],
    "chair":["chair","couch"],
    "plant":["potted plant"],
    "bed":["bed"],
    "table":["dining table"],
    "toilet":["toilet"],
    "tv":["tv"],
    "laptop":["laptop"],
    "mouse":["mouse"],
    "remote":["remote"],
    "keyboard":["keyboard"],
    "phone":["cell phone"],
    "appliance":["microwave","oven","toaster","sink","refrigerator"],
    "book":["book"],
    "clock":["clock"],
    "scissors":["scissors"]
}

# Create reverse lookup for O(1) hierarchy mapping
class_to_hierarchy = {}
for hierarchy_category, objects in object_hierarchy.items():
    for obj in objects:
        class_to_hierarchy[obj] = hierarchy_category

# Cache for model names to avoid repeated access
_model_names_cache = None

def get_object_hierarchy(class_name: str) -> str:
    """
    Map a detected object class to its hierarchy category using O(1) lookup
    
    Args:
        class_name: The detected object class name
        
    Returns:
        str: The hierarchy category for the object
    """
    return class_to_hierarchy.get(class_name, "unknown")

def get_model_names():
    """
    Get model names with caching to avoid repeated access
    
    Returns:
        dict: Model class names mapping
    """
    global _model_names_cache
    if _model_names_cache is None:
        _model_names_cache = model.names
    return _model_names_cache

def display_object_hierarchy():
    """
    Display the complete object hierarchy structure
    """
    logger.info("OBJECT HIERARCHY STRUCTURE")
    
    for hierarchy_category, objects in object_hierarchy.items():
        logger.info(f"{hierarchy_category.upper()}:")
        for obj in objects:
            logger.info(f"  - {obj}")
    
    logger.info(f"Total hierarchy categories: {len(object_hierarchy)}")
    logger.info(f"Total objects: {sum(len(objects) for objects in object_hierarchy.values())}")

@dataclass
class Detection:
    """Optimized detection data structure"""
    class_name: str
    original_class: str
    hierarchy_category: str
    confidence: float
    center_x: float
    center_y: float
    bbox: List[float]
    
    def __post_init__(self):
        self.confidence = round(self.confidence, 2)
        self.center_x = round(self.center_x, 1)
        self.center_y = round(self.center_y, 1)

def process_detections(results) -> List[Detection]:
    """
    Process YOLO results into optimized detection objects
    
    Args:
        results: YOLO inference results
        
    Returns:
        List[Detection]: Processed detection objects
    """
    detections = []
    names = get_model_names()
    
    for r in results:
        if hasattr(r, "boxes") and len(r.boxes) > 0:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= config.confidence_threshold:
                    cls_id = int(box.cls[0])
                    original_class = names[cls_id]
                    hierarchy_category = get_object_hierarchy(original_class)
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detection = Detection(
                        class_name=hierarchy_category,
                        original_class=original_class,
                        hierarchy_category=hierarchy_category,
                        confidence=conf * 100,
                        center_x=center_x,
                        center_y=center_y,
                        bbox=[x1, y1, x2, y2]
                    )
                    detections.append(detection)
    
    return detections

def draw_detection_annotations(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """
    Draw detection annotations on frame
    
    Args:
        frame: Input frame
        detections: List of detections to draw
        
    Returns:
        np.ndarray: Annotated frame
    """
    annotated_frame = frame.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        
        # Draw rectangle
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Create label
        hierarchy_values = object_hierarchy.get(detection.hierarchy_category, [])
        if len(hierarchy_values) > 1 or detection.hierarchy_category != detection.original_class:
            label = f"{detection.hierarchy_category}: {detection.original_class} - {detection.confidence:.1f}%"
        else:
            label = f"{detection.hierarchy_category}: {detection.confidence:.1f}%"
        
        # Get text size and draw background
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (int(x1), int(y1) - text_height - baseline), 
                    (int(x1) + text_width, int(y1)), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(annotated_frame, label, (int(x1), int(y1) - baseline), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return annotated_frame

def test_hierarchy_labeling(image_path: str, show_popup: bool = True) -> List[Detection]:
    """
    Test the hierarchy labeling system on a single image
    
    Args:
        image_path: Path to the test image
        show_popup: Whether to show the annotated image
        
    Returns:
        List[Detection]: Detected objects
    """
    logger.info(f"Testing hierarchy labeling on {os.path.basename(image_path)}")
    
    try:
        # Perform inference
        results = model(image_path)
        
        # Process detections
        detections = process_detections(results)
        
        if detections:
            for detection in detections:
                logger.info(f"  - {detection.hierarchy_category} (was {detection.original_class}): {detection.confidence:.2f}% | Center: ({detection.center_x:.1f}, {detection.center_y:.1f})")
        else:
            logger.info("  No objects detected!")
        
        # Create annotated image
        if show_popup:
            annotated_frame = draw_detection_annotations(results[0].plot(), detections)
            
            window_title = f"Hierarchy Labels - {os.path.basename(image_path)}"
            cv2.imshow(window_title, annotated_frame)
            
            logger.info(f"  Showing image with hierarchy labels: {window_title}")
            logger.info("  Press any key to close...")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return detections
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return []

def get_hierarchy_statistics(all_results: List[Dict]) -> Dict[str, Dict]:
    """
    Get detailed statistics about object hierarchy distribution
    
    Args:
        all_results: List of classification results
        
    Returns:
        dict: Statistics about hierarchy distribution
    """
    hierarchy_stats = defaultdict(lambda: {
        'count': 0,
        'objects': set(),
        'subdatasets': set()
    })
    
    for result in all_results:
        if 'error' not in result:
            for detection in result['detections']:
                hierarchy_category = detection['hierarchy_category']
                hierarchy_stats[hierarchy_category]['count'] += 1
                hierarchy_stats[hierarchy_category]['objects'].add(detection['original_class'])
                hierarchy_stats[hierarchy_category]['subdatasets'].add(result['subdataset'])
    
    return dict(hierarchy_stats)

class WebcamClassifier:
    """
    Optimized webcam classifier with improved performance and memory management
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.cap = None
        self.all_detections = []
        self.capture_count = 0
        self.start_time = None
        self.is_paused = False
        self.pause_start_time = None
        self.paused_duration = 0
        self.latest_annotated_image = None
        
    def initialize_camera(self) -> bool:
        """Initialize camera with error handling"""
        try:
            source = self.config.stream_url if self.config.stream_url else self.config.camera_index
            self.cap = cv2.VideoCapture(source)
            
            if not self.cap.isOpened():
                logger.error(f"Could not open camera source {source}")
                return False
    
    # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Camera source {source} initialized successfully")
            logger.info(f"Resolution: {actual_width}x{actual_height}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released and windows closed")
    
    def _draw_stacked_text(
        self,
        frame: np.ndarray,
        lines: List[str],
        start_y: int = 30,
        line_height: int = 28,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Render multiple lines top-down so UI text never overlaps."""
        y = start_y
        for line in lines:
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            y += line_height
        return frame
    
    def process_frame_detection(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """Process frame for detections and create annotated image"""
        try:
            # Perform YOLO inference
            results = model(frame)
            
            # Process detections
            detections = process_detections(results)
            
            # Create annotated image
            annotated_frame = draw_detection_annotations(frame, detections)
            
            return detections, annotated_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], frame
    
    def add_info_overlay(self, frame: np.ndarray, capture_count: int, elapsed: float, is_paused: bool) -> np.ndarray:
        """Add information overlay to frame"""
        overlay_frame = frame.copy()
        
        if is_paused:
            lines = [
                "PAUSED - Press SPACEBAR to resume",
                "Press 'q' to quit",
            ]
            self._draw_stacked_text(overlay_frame, lines, color=(0, 0, 255))
        else:
            next_capture_in = self.config.capture_interval - (elapsed % self.config.capture_interval)
            lines = [
                f"Capture #{capture_count + 1} - Next in: {next_capture_in:.1f}s",
                "Press SPACEBAR to pause | 'q' to quit",
            ]
            self._draw_stacked_text(overlay_frame, lines, color=(0, 255, 0))
                
        return overlay_frame
                
    
    def run(self) -> List[Detection]:
        """Main classification loop"""
        logger.info("Starting auto-capture webcam classification")
        source_desc = self.config.stream_url if self.config.stream_url else f"camera index {self.config.camera_index}"
        logger.info(f"Using {source_desc}")
        logger.info(f"Capture interval: {self.config.capture_interval} seconds")
        
        if not self.initialize_camera():
            return []
        
        self.start_time = time.time()
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Could not read frame from camera")
                    break
                
                # Handle pause/resume timing
                current_time = time.time()
                if self.is_paused:
                    if self.pause_start_time is not None:
                        self.paused_duration += current_time - self.pause_start_time
                        self.pause_start_time = current_time
                else:
                    if self.pause_start_time is not None:
                        self.pause_start_time = None
                
                # Calculate elapsed time accounting for pauses
                elapsed = current_time - self.start_time - self.paused_duration
                
                # Check if it's time to capture
                if not self.is_paused and elapsed >= (self.capture_count + 1) * self.config.capture_interval:
                    logger.info(f"Auto-capturing image {self.capture_count + 1}...")
                    
                    # Process frame for detections
                    detections, annotated_frame = self.process_frame_detection(frame)
                    # Save latest annotated image (boxes only); UI text is added live later
                    self.latest_annotated_image = annotated_frame
                    
                    # Log capture (no saving)
                    
                    # Log detections
                    if detections:
                        for detection in detections:
                            logger.info(f"  - {detection.hierarchy_category}: {detection.original_class} - {detection.confidence:.2f}% | Center: ({detection.center_x:.1f}, {detection.center_y:.1f})")
                    else:
                        logger.info("  No objects detected")
                    
                    self.all_detections.extend(detections)
                    self.capture_count += 1
                
                # Display logic
                display_frame = self.get_display_frame(frame, elapsed)
                cv2.imshow("Classification Results & Live Camera", display_frame)
                
                # Handle keyboard input
                if not self.handle_keyboard_input():
                    break
            
            return self.all_detections
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return []
        
        finally:
            self.cleanup()
    
    def get_display_frame(self, frame: np.ndarray, elapsed: float) -> np.ndarray:
        """Get the appropriate frame to display"""
        if self.latest_annotated_image is not None:
            time_since_last_capture = elapsed % self.config.capture_interval
            
            if time_since_last_capture < self.config.display_duration:
                # Show captured image with results and fresh overlay text (no freezing)
                display_frame = self.latest_annotated_image.copy()
                display_frame = self.add_info_overlay(
                    display_frame, self.capture_count, elapsed, self.is_paused
                )
                self._draw_stacked_text(
                    display_frame,
                    [f"Capture #{self.capture_count} - Showing results"],
                    start_y=90,
                    color=(0, 255, 0),
                )
            else:
                # Show live camera feed
                display_frame = self.add_info_overlay(frame, self.capture_count, elapsed, self.is_paused)
        else:
            # Show waiting message for first capture
            display_frame = self.add_info_overlay(frame, self.capture_count, elapsed, self.is_paused)
            time_until_first = self.config.capture_interval - elapsed
            if time_until_first > 0:
                self._draw_stacked_text(
                    display_frame,
                    [f"Waiting for first capture... {time_until_first:.1f}s"],
                    start_y=90,
                    color=(0, 255, 0),
                )
        
        return display_frame
    
    def handle_keyboard_input(self) -> bool:
        """Handle keyboard input, return False to quit"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            logger.info("Auto-capture stopped by user")
            return False
        elif key == ord(' '):  # Spacebar
            current_time = time.time()
            if self.is_paused:
                self.is_paused = False
                self.pause_start_time = None
                logger.info("Resuming auto-capture...")
            else:
                self.is_paused = True
                self.pause_start_time = current_time
                logger.info("Pausing auto-capture...")
        
        return True
    
    def display_final_results(self):
        """Display final classification results"""
        logger.info(f"FINAL RESULTS:")
        logger.info(f"Total captures: {self.capture_count}")
        logger.info(f"Total objects detected: {len(self.all_detections)}")
        
        if self.all_detections:
            hierarchy_counts = Counter(detection.hierarchy_category for detection in self.all_detections)
            
            logger.info(f"Objects by hierarchy category:")
            for category, count in hierarchy_counts.most_common():
                logger.info(f"  - {category}: {count} objects")

def classify_image_from_webcam() -> List[Detection]:
    """
    Automatically capture and classify webcam images with optimized performance
    """
    classifier = WebcamClassifier(config)
    detections = classifier.run()
    classifier.display_final_results()
    return detections

def display_summary_results(all_results: List[Dict]) -> None:
    """
    Display a summary of classification results with optimized processing
    """
    logger.info("RANDOM IMAGE CLASSIFICATION SUMMARY")
    
    total_detections = 0
    class_counts = Counter()
    hierarchy_counts = Counter()
    subdataset_counts = Counter()
    
    logger.info(f"Processed {len(all_results)} random images:")
    
    for result in all_results:
        if 'error' not in result:
            subdataset = result['subdataset']
            subdataset_counts[subdataset] += 1
            
            total_detections += len(result['detections'])
            for detection in result['detections']:
                class_name = detection['class']
                original_class = detection['original_class']
                hierarchy_category = detection['hierarchy_category']
                
                class_counts[class_name] += 1
                hierarchy_counts[hierarchy_category] += 1
                
                logger.info(f"  {subdataset}/{result['image_name']}: {class_name} at center ({detection['center_x']:.1f}, {detection['center_y']:.1f}) - {detection['confidence']:.1f}%")
        else:
            logger.error(f"  {result['subdataset']}/{result['image_name']}: ERROR - {result['error']}")
    
    logger.info(f"Total detections: {total_detections}")
    
    if subdataset_counts:
        logger.info("Images by sport category:")
        for sport, count in subdataset_counts.most_common():
            logger.info(f"  - {sport}: {count} image(s)")
    
    if hierarchy_counts:
        logger.info("Objects by hierarchy category:")
        for hierarchy_category, count in hierarchy_counts.most_common():
            logger.info(f"  - {hierarchy_category}: {count} objects")
        
        # Display detailed hierarchy statistics
        hierarchy_stats = get_hierarchy_statistics(all_results)
        logger.info("Detailed hierarchy analysis:")
        for hierarchy_category, stats in sorted(hierarchy_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            logger.info(f"  {hierarchy_category.upper()}:")
            logger.info(f"    - Total objects: {stats['count']}")
            logger.info(f"    - Unique object types: {len(stats['objects'])}")
            logger.info(f"    - Found in sports: {', '.join(sorted(stats['subdatasets']))}")
            logger.info(f"    - Original YOLO classes: {', '.join(sorted(stats['objects']))}")
    
    if class_counts:
        logger.info("Most common hierarchy categories detected:")
        for hierarchy_name, count in class_counts.most_common(10):
            logger.info(f"  - {hierarchy_name}: {count} times")

def main():
    """Main execution function with optimized performance"""
    logger.info("WEBCAM OBJECT CLASSIFICATION WITH HIERARCHY LABELS")
    logger.info("This will capture an image from your webcam and classify it using YOLO")
    
    # Display object hierarchy structure
    display_object_hierarchy()
    
    # Run webcam classification with optimized performance
    detections = classify_image_from_webcam()
    
    # Display results summary
    if detections:
        logger.info("CLASSIFICATION COMPLETE!")
        logger.info(f"Detected {len(detections)} objects:")
        for detection in detections:
            logger.info(f"  - {detection.hierarchy_category} (confidence: {detection.confidence:.1f}%)")
    else:
        logger.info("No objects detected or classification failed")
    
    # Alternative: Test with existing images (uncomment to use)
    # test_image_path = '/Users/shaayeralam/Spatial AI(Robotic Dog)/YOLOv8 Working/dog.jpeg'
    # test_hierarchy_labeling(test_image_path, show_popup=True)
    
    # Alternative: Random sports dataset classification (uncomment to use)
    # dataset_path = '/Users/shaayeralam/Spatial AI(Robotic Dog)/YOLOv8 Working/data'
    # all_results = classify_random_images_from_dataset(dataset_path, num_images=5, show_popups=True)
    # display_summary_results(all_results)
    
    logger.info("Webcam classification complete!")

# Main execution
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nClassification interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Ensure cleanup
        cv2.destroyAllWindows()