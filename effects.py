import cv2
import numpy as np
import random
import math
import time

# Dictionary to store all registered effects
EFFECTS = {}

# Base Effect class
class Effect:
    def __init__(self):
        pass

    def process(self, frame):
        """Process a frame with this effect"""
        return frame

# Function to register an effect
def register_effect(name):
    def decorator(cls):
        EFFECTS[name.lower()] = cls
        return cls
    return decorator

# Function to get all registered effects
def get_all_effects():
    return EFFECTS

# Function to get an effect instance by name
def get_effect(name):
    if name.lower() in EFFECTS:
        return EFFECTS[name.lower()]()
    return None

# --- Basic Effects ---

@register_effect("original")
class OriginalEffect(Effect):
    """Original image with no modifications"""
    def process(self, frame):
        return frame

@register_effect("grayscale")
class GrayscaleEffect(Effect):
    """Convert image to grayscale"""
    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

@register_effect("edge")
class EdgeDetectionEffect(Effect):
    """Detect and highlight edges"""
    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

@register_effect("cartoon")
class CartoonEffect(Effect):
    """Apply cartoon-like effect"""
    def process(self, frame):
        # Bilateral filter to reduce noise but keep edges sharp
        color = cv2.bilateralFilter(frame, 9, 250, 250)
        # Convert to grayscale and apply median blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        # Use adaptive threshold for edges
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        # Convert back to color
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # Combine with color image
        return cv2.bitwise_and(color, edges_bgr)

@register_effect("blur")
class BlurEffect(Effect):
    """Apply Gaussian blur"""
    def process(self, frame):
        return cv2.GaussianBlur(frame, (15, 15), 0)

@register_effect("sharpen")
class SharpenEffect(Effect):
    """Sharpen the image"""
    def process(self, frame):
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(frame, -1, kernel)

@register_effect("emboss")
class EmbossEffect(Effect):
    """Apply emboss effect"""
    def process(self, frame):
        kernel = np.array([[0, -1, -1],
                          [1, 0, -1],
                          [1, 1, 0]])
        emboss_img = cv2.filter2D(frame, -1, kernel) + 128
        return emboss_img

@register_effect("sepia")
class SepiaEffect(Effect):
    """Apply sepia tone"""
    def process(self, frame):
        sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                [0.349, 0.686, 0.168],
                                [0.393, 0.769, 0.189]])
        sepia_img = cv2.transform(frame, sepia_kernel)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return sepia_img

@register_effect("negative")
class NegativeEffect(Effect):
    """Invert image colors"""
    def process(self, frame):
        return cv2.bitwise_not(frame)

@register_effect("pixelate")
class PixelateEffect(Effect):
    """Pixelate the image"""
    def process(self, frame):
        height, width = frame.shape[:2]
        w, h = (15, 15)

        # Resize down
        temp = cv2.resize(frame, (width // w, height // h), interpolation=cv2.INTER_LINEAR)

        # Resize back up using NEAREST to maintain pixelated look
        return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

# --- Advanced Effects ---

@register_effect("neon")
class NeonEffect(Effect):
    """Apply neon glow effect"""
    def process(self, frame):
        # Blur the image a bit
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        # Convert to HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # Increase saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
        # Create edge mask
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        # Combine
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_colored = edges_colored * np.array([0, 255, 255], dtype=np.uint8)  # Cyan edges
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = cv2.addWeighted(result, 0.7, edges_colored, 0.3, 0)
        return result

@register_effect("vaporwave")
class VaporwaveEffect(Effect):
    """Apply vaporwave aesthetic"""
    def process(self, frame):
        # Split channels
        b, g, r = cv2.split(frame)
        # Modify channels
        r = np.clip(r * 1.2, 0, 255).astype(np.uint8)
        b = np.clip(b * 1.8, 0, 255).astype(np.uint8)
        # Add channel shift effect
        shifted = cv2.merge([
            np.roll(b, 5, axis=1),
            g,
            np.roll(r, -5, axis=1)
        ])
        return shifted

@register_effect("glitch")
class GlitchEffect(Effect):
    """Apply digital glitch effect"""
    def process(self, frame):
        # Make a copy to work with
        result = frame.copy()

        # Random horizontal slices
        for i in range(10):
            y = random.randint(0, frame.shape[0]-10)
            h = random.randint(1, 10)
            if y + h >= frame.shape[0]:
                continue

            slice = result[y:y+h, :].copy()

            # Shift the slice
            shift = random.randint(-20, 20)
            if shift > 0 and shift < slice.shape[1]:
                slice = np.hstack([slice[:, shift:], slice[:, :shift]])
            elif shift < 0 and abs(shift) < slice.shape[1]:
                shift = abs(shift)
                slice = np.hstack([slice[:, -shift:], slice[:, :-shift]])

            # Random color distortion
            channel = random.randint(0, 2)
            slice[:, :, channel] = np.clip(slice[:, :, channel] * 1.5, 0, 255).astype(np.uint8)

            # Put back
            result[y:y+h, :] = slice

        return result

@register_effect("thermal")
class ThermalEffect(Effect):
    """Apply thermal camera effect"""
    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply false color map
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

@register_effect("mirror")
class MirrorEffect(Effect):
    """Mirror the image horizontally"""
    def process(self, frame):
        return cv2.flip(frame, 1)

# --- Wild Effects ---

@register_effect("cubism")
class CubismEffect(Effect):
    """Create cubism-inspired effect with geometric shapes"""
    def process(self, frame):
        # Create a blank output
        output = np.zeros_like(frame)
        height, width = frame.shape[:2]

        # Generate random polygons
        for _ in range(50):
            # Random region size
            size = random.randint(20, max(30, min(width, height) // 4))

            # Random center point
            cx = random.randint(0, width-1)
            cy = random.randint(0, height-1)

            # Get color from original image at this point (or nearby if out of bounds)
            sample_x = min(max(0, cx), width-1)
            sample_y = min(max(0, cy), height-1)
            color = frame[sample_y, sample_x].tolist()

            # Generate polygon vertices
            num_vertices = random.randint(3, 6)
            points = []

            for i in range(num_vertices):
                angle = 2 * math.pi * i / num_vertices + random.uniform(0, math.pi / 4)
                r = size * random.uniform(0.5, 1.0)
                x = int(cx + r * math.cos(angle))
                y = int(cy + r * math.sin(angle))
                points.append([x, y])

            # Convert to numpy array
            points = np.array(points, dtype=np.int32)

            # Draw filled polygon
            cv2.fillPoly(output, [points], color)

        return output



@register_effect("dream")
class DreamEffect(Effect):
    """Create a dreamy, ethereal effect"""
    def process(self, frame):
        # Apply soft blur
        blurred = cv2.GaussianBlur(frame, (21, 21), 0)

        # Adjust colors to be more vibrant
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Increase saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255).astype(np.uint8)

        # Adjust value for glow effect
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255).astype(np.uint8)

        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Add glow by blending with blurred bright areas
        brightness = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        _, bright_areas = cv2.threshold(brightness, 200, 255, cv2.THRESH_BINARY)

        # Dilate bright areas
        kernel = np.ones((15, 15), np.uint8)
        bright_areas = cv2.dilate(bright_areas, kernel, iterations=1)

        # Blur bright areas for glow
        bright_blur = cv2.GaussianBlur(bright_areas, (21, 21), 0)

        # Create glow layer
        glow = np.zeros_like(frame)
        glow[:, :, 0] = bright_blur  # Blue glow
        glow[:, :, 1] = bright_blur  # Green glow
        glow[:, :, 2] = bright_blur  # Red glow

        # Blend glow with enhanced image
        result = cv2.addWeighted(enhanced, 0.8, glow, 0.4, 0)

        return result


@register_effect("hologram")
class HologramEffect(Effect):
    """Create sci-fi hologram effect"""
    def process(self, frame):
        # Convert to grayscale first
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create blue tint
        blue = np.zeros_like(frame)
        blue[:, :, 0] = np.clip(gray * 1.5, 0, 255)  # Blue channel
        blue[:, :, 1] = gray // 2  # Green channel
        blue[:, :, 2] = gray // 4  # Red channel

        # Add scan lines
        height, width = blue.shape[:2]
        for y in range(0, height, 2):
            blue[y:y+1, :] = (blue[y:y+1, :] * 0.6).astype(np.uint8)

        # Add horizontal lines occasionally
        num_h_lines = random.randint(5, 15)
        for _ in range(num_h_lines):
            y = random.randint(0, height - 1)
            blue[y:y+1, :] = (blue[y:y+1, :] * 1.5).astype(np.uint8)

        # Add vertical displacement occasionally
        if random.random() < 0.1:
            shift_y = random.randint(0, height - 10)
            shift_height = random.randint(5, 10)
            shift_x = random.randint(-10, 10)
            if shift_y + shift_height < height:
                if shift_x > 0:
                    blue[shift_y:shift_y+shift_height, shift_x:] = blue[shift_y:shift_y+shift_height, :-shift_x]
                    blue[shift_y:shift_y+shift_height, :shift_x] = 0
                elif shift_x < 0:
                    shift_x = -shift_x
                    blue[shift_y:shift_y+shift_height, :-shift_x] = blue[shift_y:shift_y+shift_height, shift_x:]
                    blue[shift_y:shift_y+shift_height, -shift_x:] = 0

        # Add edge glow
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edges_blue = np.zeros_like(frame)
        edges_blue[:, :, 0] = edges  # Blue channel

        # Combine
        result = cv2.addWeighted(blue, 0.8, edges_blue, 0.5, 0)

        return result

@register_effect("painterly")
class PainterlyEffect(Effect):
    """Create an artistic, painterly effect"""
    def process(self, frame):
        # Create multiple layers with different blur levels
        layer1 = cv2.GaussianBlur(frame, (21, 21), 0)
        layer2 = cv2.GaussianBlur(frame, (11, 11), 0)
        layer3 = cv2.GaussianBlur(frame, (5, 5), 0)

        # Get edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        # Convert edges to 3 channels
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Create painterly result by layering
        result = layer1.copy()

        # Add detail where edges exist
        result = np.where(edges_bgr > 0, layer3, result)

        # Add medium detail in transition areas
        transition = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1) - edges
        transition_bgr = cv2.cvtColor(transition, cv2.COLOR_GRAY2BGR)
        result = np.where(transition_bgr > 0, layer2, result)

        # Enhance colors
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return result

@register_effect("cellular")
class CellularAutomatonEffect(Effect):
    """Apply cellular automaton rules to create evolving patterns"""
    def __init__(self):
        super().__init__()
        self.cells = None
        self.iteration = 0

    def process(self, frame):
        height, width = frame.shape[:2]

        # Initialize cells on first run using brightness from the frame
        if self.cells is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Binarize
            _, cells = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
            self.cells = cells

        # Every 5 frames, reinitialize with current frame to keep it responsive
        if self.iteration % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, new_cells = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
            # Blend with existing cells
            self.cells = (self.cells * 0.7 + new_cells * 0.3) > 0.5

        # Create padded version for neighborhood calculation
        padded = np.pad(self.cells, 1, mode='wrap')

        # Apply Conway's Game of Life rules
        new_cells = np.zeros_like(self.cells)

        # Count neighbors (vectorized approach for better performance)
        for i in range(3):
            for j in range(3):
                if i != 1 or j != 1:  # Skip center cell
                    new_cells += padded[i:i+height, j:j+width]

        # Apply rules
        born = (new_cells == 3) & (self.cells == 0)
        survive = ((new_cells == 2) | (new_cells == 3)) & (self.cells == 1)
        self.cells = born | survive

        # Convert cells to colors based on original frame
        result = np.zeros_like(frame)
        live_mask = np.repeat(self.cells[:, :, np.newaxis], 3, axis=2)

        # Get average color of live cells from original frame
        avg_color = np.mean(frame[self.cells == 1], axis=0) if np.sum(self.cells) > 0 else np.array([0, 0, 0])

        # Apply colors: live cells get original frame colors, dead cells get complementary colors
        dead_mask = ~live_mask
        result[live_mask] = frame[live_mask]
        result[dead_mask] = 255 - frame[dead_mask]

        self.iteration += 1
        return result

@register_effect("solarize")
class PsychedelicSolarizeEffect(Effect):
    """Create psychedelic solarization effect"""
    def __init__(self):
        super().__init__()
        self.time = 0

    def process(self, frame):
        # Create solarized effect (invert above threshold)
        result = frame.copy()

        # Shift threshold based on time
        threshold = (128 + 64 * math.sin(self.time * 0.05)) % 255

        # Apply to each channel with different thresholds
        for c in range(3):
            channel = result[:, :, c]
            mask = channel > (threshold + c * 20) % 255
            channel[mask] = 255 - channel[mask]

        # Increase color saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.time += 1
        return result

@register_effect("kaleidoscope")
class KaleidoscopeEffect(Effect):
   """Create kaleidoscope effect"""
   def process(self, frame):
       height, width = frame.shape[:2]
       cx, cy = width // 2, height // 2

       # Number of segments
       segments = 8

       # Create output image
       output = np.zeros_like(frame)

       # Calculate segment angle
       segment_angle = 2 * math.pi / segments

       # Extract a triangular segment from the original
       mask = np.zeros((height, width), dtype=np.uint8)
       points = np.array([
           [cx, cy],
           [width, 0],
           [0, 0]
       ], dtype=np.int32)
       cv2.fillPoly(mask, [points], 255)

       segment = frame.copy()
       segment[mask == 0] = 0

       # Rotate and copy segment to create kaleidoscope
       for i in range(segments):
           angle = i * segment_angle
           rotation_matrix = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle), 1.0)
           rotated = cv2.warpAffine(segment, rotation_matrix, (width, height))
           output = cv2.add(output, rotated)

       return output

@register_effect("rgb_scatter")
class RGBScatterEffect(Effect):
    """RGB channels scatter in different directions based on audio"""
    def __init__(self):
        super().__init__()
        self.time = 0

    def process(self, frame):
        # Split into RGB channels
        b, g, r = cv2.split(frame)

        # Calculate base shift amount
        shift_amount = 20  # Fixed shift amount

        # Create time-based movement
        t = self.time * 0.1

        # Each channel gets its own direction
        # Red channel (120° angle)
        angle_r = t + 2 * math.pi / 3
        shift_r_x = int(shift_amount * math.cos(angle_r))
        shift_r_y = int(shift_amount * math.sin(angle_r))

        # Green channel (240° angle)
        angle_g = t + 4 * math.pi / 3
        shift_g_x = int(shift_amount * math.cos(angle_g))
        shift_g_y = int(shift_amount * math.sin(angle_g))

        # Blue channel (0° angle)
        angle_b = t
        shift_b_x = int(shift_amount * math.cos(angle_b))
        shift_b_y = int(shift_amount * math.sin(angle_b))

        # Apply the shifts
        height, width = frame.shape[:2]

        # Create translation matrices
        r_matrix = np.float32([[1, 0, shift_r_x], [0, 1, shift_r_y]])
        g_matrix = np.float32([[1, 0, shift_g_x], [0, 1, shift_g_y]])
        b_matrix = np.float32([[1, 0, shift_b_x], [0, 1, shift_b_y]])

        # Apply translations
        r_shifted = cv2.warpAffine(r, r_matrix, (width, height))
        g_shifted = cv2.warpAffine(g, g_matrix, (width, height))
        b_shifted = cv2.warpAffine(b, b_matrix, (width, height))

        # Merge channels back
        result = cv2.merge([b_shifted, g_shifted, r_shifted])

        self.time += 1
        return result

@register_effect("rgb_directional")
class RGBDirectionalShiftEffect(Effect):
    """RGB shift with channels moving in specific directions"""
    def process(self, frame):
        # Split into RGB channels
        b, g, r = cv2.split(frame)

        # Fixed shift amount
        shift_amount = 25

        # Fixed directions as requested:
        # Red: North (0, -shift)
        shift_r_x = 0
        shift_r_y = -shift_amount

        # Green: Southwest (-shift*cos(30°), shift*sin(30°))
        shift_g_x = int(-shift_amount * 0.866)  # cos(30°) ≈ 0.866
        shift_g_y = int(shift_amount * 0.5)     # sin(30°) = 0.5

        # Blue: Southeast (shift*cos(30°), shift*sin(30°))
        shift_b_x = int(shift_amount * 0.866)
        shift_b_y = int(shift_amount * 0.5)

        # Apply the shifts
        height, width = frame.shape[:2]

        # Create translation matrices
        r_matrix = np.float32([[1, 0, shift_r_x], [0, 1, shift_r_y]])
        g_matrix = np.float32([[1, 0, shift_g_x], [0, 1, shift_g_y]])
        b_matrix = np.float32([[1, 0, shift_b_x], [0, 1, shift_b_y]])

        # Apply translations
        r_shifted = cv2.warpAffine(r, r_matrix, (width, height))
        g_shifted = cv2.warpAffine(g, g_matrix, (width, height))
        b_shifted = cv2.warpAffine(b, b_matrix, (width, height))

        # Merge channels back
        result = cv2.merge([b_shifted, g_shifted, r_shifted])

        return result

@register_effect("rgb_shift_extreme")
class RGBShiftExtremeEffect(Effect):
    """Extreme RGB channel shifting effect"""
    def __init__(self):
        super().__init__()
        self.time = 0

    def process(self, frame):
        # Split into RGB channels
        b, g, r = cv2.split(frame)

        # Fixed large shifts for visibility
        shift_amount = 30

        # Fixed shifts for more obvious effect
        shift_r_x = shift_amount
        shift_r_y = 0
        shift_b_x = -shift_amount
        shift_b_y = 0

        # Apply the shifts
        height, width = frame.shape[:2]

        # Create translation matrices
        r_matrix = np.float32([[1, 0, shift_r_x], [0, 1, shift_r_y]])
        b_matrix = np.float32([[1, 0, shift_b_x], [0, 1, shift_b_y]])

        # Apply translations
        r_shifted = cv2.warpAffine(r, r_matrix, (width, height))
        b_shifted = cv2.warpAffine(b, b_matrix, (width, height))

        # Merge channels back
        result = cv2.merge([b_shifted, g, r_shifted])

        self.time += 1
        return result

@register_effect("neontrails")
class NeonTrailsEffect(Effect):
    """Create neon-colored motion trails"""
    def __init__(self):
        super().__init__()
        self.prev_frames = []
        self.max_trails = 15

    def process(self, frame):
        # Add current frame to history
        self.prev_frames.append(frame.copy())
        if len(self.prev_frames) > self.max_trails:
            self.prev_frames.pop(0)

        # Start with black background
        result = np.zeros_like(frame)

        # Blend frames with different colors
        num_frames = len(self.prev_frames)
        for i, past_frame in enumerate(self.prev_frames):
            # Weight based on recency
            weight = (i + 1) / num_frames

            # Convert to HSV
            hsv = cv2.cvtColor(past_frame, cv2.COLOR_BGR2HSV)

            # Assign hue based on frame position
            hue = int((i * 180 / num_frames) % 180)
            hsv[:, :, 0] = hue

            # Increase saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)

            # Convert back to BGR
            colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Add to result with weight
            cv2.addWeighted(result, 1.0, colored, weight, 0, result)

        return result

@register_effect("graytrails")
class GrayTrailsEffect(Effect):
    """Create colorful neon trails that follow motion"""
    def __init__(self):
        super().__init__()
        self.prev_frame = None
        self.trails = None
        self.colors = None

    def process(self, frame):
        height, width = frame.shape[:2]

        # Initialize on first frame
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.trails = np.zeros((height, width, 3), dtype=np.float32)
            self.colors = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            return frame

        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference to detect motion
        diff = cv2.absdiff(gray, self.prev_frame)
        _, motion = cv2.threshold(diff, 25, 1, cv2.THRESH_BINARY)

        # Dilate motion mask
        kernel = np.ones((5, 5), np.uint8)
        motion = cv2.dilate(motion, kernel, iterations=1)

        # Update trails (decay existing, add new motion)
        self.trails *= 0.8  # Decay factor

        # Add new motion to trails with colors
        for c in range(3):
            self.trails[:, :, c] += motion * self.colors[:, :, c]

        # Occasionally rotate color palette for psychedelic effect
        if random.random() < 0.05:
            hsv = cv2.cvtColor(self.colors, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180
            self.colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Combine original frame with trails
        result = frame.copy()
        trails_uint8 = np.clip(self.trails, 0, 255).astype(np.uint8)
        mask = np.max(trails_uint8, axis=2) > 0
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        result[mask_3d] = cv2.addWeighted(result, 0.5, trails_uint8, 0.5, 0)[mask_3d]

        # Update previous frame
        self.prev_frame = gray

        return result