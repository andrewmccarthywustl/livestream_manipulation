import cv2
import numpy as np
import random
import math
import time

# Dictionary to store all registered effects
EFFECTS = {}

# Base Effect class
class Effect:
    def __init__(self, audio_manager=None):
        self.audio_manager = audio_manager

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
def get_effect(name, audio_manager=None):
    if name.lower() in EFFECTS:
        return EFFECTS[name.lower()](audio_manager)
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

@register_effect("fisheye")
class FisheyeEffect(Effect):
    """Apply fisheye lens distortion"""
    def process(self, frame):
        height, width = frame.shape[:2]

        # Calculate distortion center
        cx, cy = width / 2, height / 2

        # Prepare distortion maps
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)

        # Calculate fisheye distortion
        for y in range(height):
            for x in range(width):
                # Normalize coordinates
                nx = (x - cx) / cx
                ny = (y - cy) / cy
                nr = math.sqrt(nx * nx + ny * ny)

                if nr < 1.0:
                    # Apply fisheye formula
                    theta = nr ** 2.0  # Strength parameter
                    nx2 = theta * nx / nr if nr > 0 else 0
                    ny2 = theta * ny / nr if nr > 0 else 0

                    # Convert back to pixel coordinates
                    map_x[y, x] = cx + nx2 * cx
                    map_y[y, x] = cy + ny2 * cy
                else:
                    map_x[y, x] = 0
                    map_y[y, x] = 0

        # Remap the image
        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

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

@register_effect("starfield")
class StarfieldEffect(Effect):
    """Create a starfield/space warp effect"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.stars = None
        self.time = 0

    def process(self, frame):
        height, width = frame.shape[:2]

        # Initialize stars if not already done
        if self.stars is None:
            self.stars = []
            for _ in range(200):
                # Random 3D position
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = random.uniform(0, 1)
                self.stars.append([x, y, z])

        # Create output image
        output = np.zeros_like(frame)
        center_x, center_y = width // 2, height // 2

        # Update and draw stars
        for i, (x, y, z) in enumerate(self.stars):
            # Update z position (moving towards viewer)
            z -= 0.02

            # If star moved past viewer, reset it
            if z <= 0:
                x = random.uniform(-1, 1)
                y = random.uniform(-1, 1)
                z = 1
                self.stars[i] = [x, y, z]
            else:
                self.stars[i] = [x, y, z]

            # Project 3D position to 2D screen
            if z < 1:
                # Calculate projected position
                px = int(center_x + (x / z) * width)
                py = int(center_y + (y / z) * height)

                # Draw star if within frame
                if 0 <= px < width and 0 <= py < height:
                    # Star size and brightness based on z (closer = brighter and larger)
                    size = int(max(1, (1 - z) * 5))
                    brightness = int(255 * (1 - z))
                    color = [brightness, brightness, brightness]

                    # Draw star as circle
                    cv2.circle(output, (px, py), size, color, -1)

                    # Add glow for brighter stars
                    if brightness > 200:
                        cv2.circle(output, (px, py), size * 2, [b // 2 for b in color], -1)

        # Blend with original frame
        result = cv2.addWeighted(frame, 0.2, output, 1.0, 0)

        # Update time
        self.time += 1

        return result

@register_effect("matrix")
class MatrixEffect(Effect):
    """Create digital rain/matrix code effect"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.streams = None
        self.time = 0
        self.char_set = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ0123456789"

    def process(self, frame):
        height, width = frame.shape[:2]

        # Initialize streams if not already done
        if self.streams is None:
            self.streams = []
            stream_count = int(width * 0.05)  # 5% of width as streams

            for _ in range(stream_count):
                x = random.randint(0, width - 1)
                length = random.randint(5, 30)
                speed_factor = random.uniform(0.5, 1.5)
                head_pos = random.randint(-length, 0)
                self.streams.append({
                    'x': x,
                    'head_pos': head_pos,
                    'length': length,
                    'speed_factor': speed_factor,
                    'chars': [random.choice(self.char_set) for _ in range(length)]
                })

        # Create black canvas
        result = frame.copy()

        # Draw matrix streams
        for stream in self.streams:
            x = stream['x']
            head_pos = stream['head_pos']
            length = stream['length']

            # Update head position
            stream['head_pos'] += int(2 * stream['speed_factor'])

            # Reset if stream is off screen
            if head_pos - length > height:
                stream['head_pos'] = -length
                stream['x'] = random.randint(0, width - 1)
                stream['length'] = random.randint(5, 30)
                stream['chars'] = [random.choice(self.char_set) for _ in range(length)]

            # Draw characters
            for i in range(length):
                y = head_pos - i
                if 0 <= y < height and 0 <= x < width:
                    # Occasional character change
                    if random.random() < 0.05:
                        stream['chars'][i] = random.choice(self.char_set)

                    # Brightness decreases with distance from head
                    brightness = 255 - int(255 * i / length)
                    if i == 0:  # Head is brighter (white)
                        color = (255, 255, 255)
                    else:  # Trail is green
                        color = (0, brightness, 0)

                    # Draw a small rectangle for each character
                    rect_size = 4
                    y_start = max(0, y - rect_size // 2)
                    y_end = min(height, y + rect_size // 2)
                    x_start = max(0, x - rect_size // 2)
                    x_end = min(width, x + rect_size // 2)

                    if x_end > x_start and y_end > y_start:
                        result[y_start:y_end, x_start:x_end] = color

        self.time += 1

        return result

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

@register_effect("liquid")
class LiquidEffect(Effect):
    """Create a flowing liquid-like distortion"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0

    def process(self, frame):
        height, width = frame.shape[:2]

        # Create distortion maps
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)

        # Generate liquid-like distortion
        time_val = self.time * 0.05
        for y in range(height):
            for x in range(width):
                # Liquid wave pattern
                dx = 20 * math.sin(y / 30.0 + time_val) + 10 * math.sin(x / 60.0 - time_val)
                dy = 15 * math.cos(x / 20.0 + time_val) + 8 * math.cos(y / 40.0 - time_val * 0.5)

                # Ensure within bounds
                nx = min(max(0, x + dx), width - 1)
                ny = min(max(0, y + dy), height - 1)

                map_x[y, x] = nx
                map_y[y, x] = ny

        # Apply distortion
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # Update time
        self.time += 1

        return result

# --- Audio-Responsive Effects ---

@register_effect("pulse")
class PulseEffect(Effect):
    """Pulse the image based on audio volume"""
    def process(self, frame):
        if self.audio_manager is None:
            return frame

        # Get volume
        volume = self.audio_manager.audio_data["volume"]

        # Scale based on volume
        volume_norm = min(1.0, volume / 10000)
        scale = 1.0 + volume_norm * 0.2  # Scale between 1.0 and 1.2

        height, width = frame.shape[:2]
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Calculate offsets
        x_offset = (scaled_width - width) // 2
        y_offset = (scaled_height - height) // 2

        # Scale up
        scaled_frame = cv2.resize(frame, (scaled_width, scaled_height))

        # Crop center portion
        if x_offset > 0 and y_offset > 0:
            result = scaled_frame[y_offset:y_offset+height, x_offset:x_offset+width]
        else:
            result = frame

        return result

@register_effect("color_pulse")
class ColorPulseEffect(Effect):
    """Change colors based on audio intensity"""
    def process(self, frame):
        if self.audio_manager is None:
            return frame

        # Get volume and beat info
        volume = self.audio_manager.audio_data["volume"]
        beat_detected = self.audio_manager.audio_data["beat_detected"]

        # Normalize volume
        volume_norm = min(1.0, volume / 10000)

        # Convert to HSV to adjust saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Adjust saturation based on volume
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + volume_norm), 0, 255).astype(np.uint8)

        # If beat detected, shift hue
        if beat_detected:
            hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180

        # Convert back to BGR
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return result

@register_effect("beat_flash")
class BeatFlashEffect(Effect):
   """Flash the image on audio beats"""
   def process(self, frame):
       if self.audio_manager is None:
           return frame

       # Get beat info
       beat_detected = self.audio_manager.audio_data["beat_detected"]

       # Normal image
       result = frame.copy()

       # Flash on beat
       if beat_detected:
           # Increase brightness temporarily
           hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
           hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.5, 0, 255).astype(np.uint8)
           result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

       return result

@register_effect("audio_glitch")
class AudioGlitchEffect(Effect):
   """Apply glitch effects based on audio"""
   def process(self, frame):
       if self.audio_manager is None:
           return frame

       # Get audio data
       volume = self.audio_manager.audio_data["volume"]
       beat_detected = self.audio_manager.audio_data["beat_detected"]

       # Normalize volume
       volume_norm = min(1.0, volume / 10000)

       # Make a copy to work with
       result = frame.copy()

       # Number of glitch slices based on volume
       num_slices = int(5 + volume_norm * 15)

       # Random horizontal slices
       for i in range(num_slices):
           y = random.randint(0, frame.shape[0]-10)
           h = random.randint(1, 10)
           if y + h >= frame.shape[0]:
               continue

           slice = result[y:y+h, :].copy()

           # Shift based on volume
           shift = random.randint(-int(20 * volume_norm), int(20 * volume_norm))
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

       # Add extra effects on beat
       if beat_detected:
           # Add color channel shift on beat
           b, g, r = cv2.split(result)
           shift_amount = int(10 * volume_norm)
           result = cv2.merge([
               np.roll(b, random.randint(-shift_amount, shift_amount), axis=1),
               np.roll(g, random.randint(-shift_amount, shift_amount), axis=1),
               np.roll(r, random.randint(-shift_amount, shift_amount), axis=1)
           ])

       return result

@register_effect("equalizer")
class EqualizerEffect(Effect):
    """Display audio frequency spectrum as equalizer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothed_spectrum = None
        self.smoothing_factor = 0.3  # Adjust for desired smoothness

    def process(self, frame, alpha=0.5):
        if self.audio_manager is None:
            return frame

        spectrum = self.audio_manager.audio_data.get("spectrum")
        if spectrum is None:
            return frame

        height, width = frame.shape[:2]
        result = frame.copy()
        overlay = frame.copy()

        spectrum = spectrum[:100]

        if len(spectrum) > 0:
            spectrum = spectrum / (np.max(spectrum) if np.max(spectrum) > 0 else 1)
            spectrum = spectrum * height // 2

            # Exponential Smoothing
            if self.smoothed_spectrum is None:
                self.smoothed_spectrum = spectrum.copy()
            else:
                self.smoothed_spectrum = (
                    self.smoothing_factor * spectrum
                    + (1 - self.smoothing_factor) * self.smoothed_spectrum
                )

            smoothed_spectrum = self.smoothed_spectrum

            num_bands = min(20, len(smoothed_spectrum))
            band_width = width // num_bands

            for i in range(num_bands):
                idx = int(i * len(smoothed_spectrum) / num_bands)
                band_height = int(smoothed_spectrum[idx])

                hue = int(i * 180 / num_bands)
                color = cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
                )[0][0].tolist()

                x1 = i * band_width
                y1 = height - band_height
                x2 = (i + 1) * band_width
                y2 = height
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

            cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        return result

@register_effect("waveform")
class WaveformEffect(Effect):
   """Display audio waveform overlay"""
   def __init__(self, audio_manager=None):
       super().__init__(audio_manager)
       self.history = []
       self.history_length = 100

   def process(self, frame):
       if self.audio_manager is None:
           return frame

       # Get current volume
       volume = self.audio_manager.audio_data["volume"]

       # Update history
       self.history.append(volume)
       if len(self.history) > self.history_length:
           self.history.pop(0)

       # Get dimensions
       height, width = frame.shape[:2]

       # Create copy of frame
       result = frame.copy()

       # Draw waveform
       if len(self.history) > 1:
           # Normalize
           max_vol = max(self.history) if max(self.history) > 0 else 1
           norm_history = [v / max_vol for v in self.history]

           # Points for polyline
           points = []
           for i, vol in enumerate(norm_history):
               x = int(i * width / len(norm_history))
               y = int(height // 2 - vol * height // 4)
               points.append((x, y))

           # Draw polyline
           points_array = np.array(points, dtype=np.int32)
           cv2.polylines(result, [points_array], False, (0, 255, 0), 2)

           # Draw mirrored polyline (for symmetric waveform)
           mirror_points = [(p[0], height - p[1]) for p in points]
           mirror_array = np.array(mirror_points, dtype=np.int32)
           cv2.polylines(result, [mirror_array], False, (0, 255, 0), 2)

       return result

# Additional wild effect
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

# Another wild effect
@register_effect("fractal_noise")
class FractalNoiseEffect(Effect):
   """Apply dynamic fractal noise pattern"""
   def __init__(self, audio_manager=None):
       super().__init__(audio_manager)
       self.time = 0

   def process(self, frame):
       height, width = frame.shape[:2]

       # Create noise pattern
       noise = np.zeros((height, width), dtype=np.float32)

       # Generate Perlin-like noise (approximation)
       time_offset = self.time * 0.05
       for y in range(0, height, 4):  # Step by 4 for performance
           for x in range(0, width, 4):  # Step by 4 for performance
               # Multi-octave noise
               value = 0
               amplitude = 1.0
               frequency = 1.0
               for _ in range(3):  # 3 octaves for performance
                   nx = x / width * 30.0 * frequency
                   ny = y / height * 30.0 * frequency
                   nt = time_offset * frequency

                   # Simple hash function for pseudo-random noise
                   n = math.sin(nx * 12.9898 + ny * 78.233 + nt * 43.2364) * 43758.5453
                   value += (n - math.floor(n)) * amplitude

                   amplitude *= 0.5
                   frequency *= 2.0

               # Fill 4x4 block with same value for performance
               for dy in range(4):
                   for dx in range(4):
                       if y+dy < height and x+dx < width:
                           noise[y+dy, x+dx] = value

       # Normalize to 0-255
       noise = ((noise - np.min(noise)) / (np.max(noise) - np.min(noise) + 1e-8) * 255).astype(np.uint8)

       # Apply colormap
       noise_color = cv2.applyColorMap(noise, cv2.COLORMAP_PLASMA)

       # Blend with original image
       result = cv2.addWeighted(frame, 0.7, noise_color, 0.5, 0)

       # Update time
       self.time += 1

       return result


@register_effect("datamosh")
class DatamoshEffect(Effect):
    """Create digital datamoshing effect with motion frame corruption"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.prev_frame = None
        self.key_frame_counter = 0

    def process(self, frame):
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return frame

        # Only use key frames occasionally
        self.key_frame_counter += 1
        if self.key_frame_counter % 30 == 0:
            self.prev_frame = frame.copy()
            return frame

        # Create mosh effect
        result = frame.copy()

        # Randomly select regions to keep from previous frame
        height, width = frame.shape[:2]
        num_regions = random.randint(5, 15)

        for _ in range(num_regions):
            # Random region
            x = random.randint(0, width - 50)
            y = random.randint(0, height - 50)
            w = random.randint(30, min(width - x, 200))
            h = random.randint(30, min(height - y, 200))

            # Copy region from previous frame
            result[y:y+h, x:x+w] = self.prev_frame[y:y+h, x:x+w]

            # Add some motion blur
            if random.random() < 0.5:
                shift = random.randint(-20, 20)
                if shift > 0 and x + w + shift < width:
                    result[y:y+h, x+shift:x+w+shift] = self.prev_frame[y:y+h, x:x+w]
                elif shift < 0 and x + shift >= 0:
                    result[y:y+h, x+shift:x+w+shift] = self.prev_frame[y:y+h, x:x+w]

        # Save current frame
        self.prev_frame = frame.copy()

        return result

@register_effect("vhs")
class VHSEffect(Effect):
    """Create retro VHS tape distortion effect"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0

    def process(self, frame):
        # Add noise
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame_noisy = cv2.add(frame, noise)

        # Create VHS tracking lines
        height, width = frame.shape[:2]
        result = frame_noisy.copy()

        # Add horizontal noise lines
        num_lines = random.randint(3, 8)
        for _ in range(num_lines):
            y = random.randint(0, height - 1)
            h = random.randint(1, 5)
            line_type = random.randint(0, 2)

            if line_type == 0:  # White noise line
                result[y:y+h, :] = np.random.randint(150, 255, (h, width, 3), dtype=np.uint8)
            elif line_type == 1:  # Black line
                result[y:y+h, :] = 0
            else:  # Color shift line
                result[y:y+h, :, 0] = np.clip(result[y:y+h, :, 0] + random.randint(-50, 50), 0, 255)

        # Shift colors occasionally
        if self.time % 10 == 0:
            r, g, b = cv2.split(result)
            result = cv2.merge([
                np.roll(r, random.randint(-8, 8), axis=1),
                g,
                np.roll(b, random.randint(-8, 8), axis=1)
            ])

        # Reduce color depth to simulate VHS quality
        result = result // 16 * 16

        # Add tracking distortion
        if random.random() < 0.1:
            wave_height = random.randint(5, 15)
            for y in range(height):
                shift = int(wave_height * math.sin(y / 30.0 + self.time * 0.1))
                if abs(shift) > 0:
                    result[y, :] = np.roll(result[y, :], shift, axis=0)

        self.time += 1
        return result

@register_effect("timedisplace")
class TimeDisplacementEffect(Effect):
    """Create time displacement map effect"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.frames = []
        self.max_frames = 60

    def process(self, frame):
        # Store current frame
        self.frames.append(frame.copy())
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

        if len(self.frames) < 10:
            return frame

        # Create displacement map based on brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normalized = gray / 255.0

        result = np.zeros_like(frame)
        height, width = frame.shape[:2]

        # For each pixel, select a frame from history based on brightness
        for y in range(0, height, 2):  # Skip pixels for performance
            for x in range(0, width, 2):
                # Get displacement value (0-1)
                val = normalized[y, x]

                # Map to frame index
                frame_idx = min(len(self.frames) - 1, int(val * len(self.frames)))

                # Get color from historical frame
                if frame_idx >= 0 and frame_idx < len(self.frames):
                    # Apply to 2x2 block for performance
                    for dy in range(2):
                        for dx in range(2):
                            if y+dy < height and x+dx < width:
                                result[y+dy, x+dx] = self.frames[frame_idx][y+dy, x+dx]

        return result

@register_effect("chromakey")
class ChromaKeyMirrorEffect(Effect):
    """Psychedelic chroma key mirror effect"""
    def process(self, frame):
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mirrored version
        mirrored = cv2.flip(frame, 1)
        hsv_mirrored = cv2.cvtColor(mirrored, cv2.COLOR_BGR2HSV)

        # Shift hue in mirrored version
        hsv_mirrored[:, :, 0] = (hsv_mirrored[:, :, 0] + 90) % 180
        colored_mirror = cv2.cvtColor(hsv_mirrored, cv2.COLOR_HSV2BGR)

        # Create mask based on brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Apply mask to combine original and mirrored
        result = frame.copy()
        result[mask > 0] = colored_mirror[mask > 0]

        return result

@register_effect("neontrails")
class NeonTrailsEffect(Effect):
    """Create neon-colored motion trails"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
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

@register_effect("slice_shuffle")
class SliceShuffleEffect(Effect):
    """Cut the image into slices and shuffle them"""
    def process(self, frame):
        height, width = frame.shape[:2]

        # Define number of slices
        h_slices = random.randint(5, 15)
        v_slices = random.randint(5, 15)

        # Calculate slice dimensions
        slice_height = height // v_slices
        slice_width = width // h_slices

        # Create empty output
        result = np.zeros_like(frame)

        # Create list of slices and their positions
        slices = []
        for y in range(0, height - slice_height + 1, slice_height):
            for x in range(0, width - slice_width + 1, slice_width):
                slices.append((frame[y:y+slice_height, x:x+slice_width].copy(), y, x))

        # Shuffle slices
        random.shuffle(slices)

        # Place slices back
        i = 0
        for y in range(0, height - slice_height + 1, slice_height):
            for x in range(0, width - slice_width + 1, slice_width):
                if i < len(slices):
                    result[y:y+slice_height, x:x+slice_width] = slices[i][0]
                    i += 1

        return result

@register_effect("fractalize")
class FractalizeEffect(Effect):
    """Create recursive image-within-image fractal effect"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.depth = 3  # Recursion depth

    def process(self, frame):
        def recursive_paste(img, depth):
            if depth <= 0:
                return img

            h, w = img.shape[:2]

            # Create smaller version
            small = cv2.resize(img, (w//2, h//2))

            # Process smaller version recursively
            processed = recursive_paste(small, depth-1)

            # Resize back and paste in corners
            corner = cv2.resize(processed, (w//2, h//2))

            # Create output with original in center
            result = img.copy()

            # Paste in corners
            result[0:h//2, 0:w//2] = corner  # top-left
            result[0:h//2, w//2:w] = cv2.flip(corner, 1)  # top-right
            result[h//2:h, 0:w//2] = cv2.flip(corner, 0)  # bottom-left
            result[h//2:h, w//2:w] = cv2.flip(corner, -1)  # bottom-right

            return result

        return recursive_paste(frame, self.depth)

@register_effect("solarize")
class PsychedelicSolarizeEffect(Effect):
    """Create psychedelic solarization effect"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
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

@register_effect("scanlines")
class ScanlineGlitchEffect(Effect):
    """Create CRT scanline effect with glitches"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0

    def process(self, frame):
        height, width = frame.shape[:2]
        result = frame.copy()

        # Add scanlines
        for y in range(0, height, 2):
            result[y:y+1, :] = (result[y:y+1, :] * 0.5).astype(np.uint8)

        # Add vertical distortion
        if self.time % 30 < 5:  # Sometimes
            wave_x = int(10 * math.sin(self.time * 0.2))
            for y in range(height):
                offset = int(wave_x * math.sin(y / 20 + self.time * 0.1))
                if abs(offset) > 0:
                    result[y, :] = np.roll(result[y, :], offset, axis=0)

        # Random horizontal glitch lines
        if random.random() < 0.1:
            num_glitches = random.randint(1, 5)
            for _ in range(num_glitches):
                y = random.randint(0, height - 10)
                h = random.randint(5, 10)
                shift = random.randint(-50, 50)
                if abs(shift) > 0:
                    result[y:y+h, :] = np.roll(result[y:y+h, :], shift, axis=1)

        # Random color shift in a band
        if random.random() < 0.05:
            y = random.randint(0, height - 30)
            h = random.randint(10, 30)
            channel = random.randint(0, 2)
            value = random.randint(-50, 50)
            result[y:y+h, :, channel] = np.clip(result[y:y+h, :, channel] + value, 0, 255)

        self.time += 1
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

@register_effect("dna")
class DNAEffect(Effect):
    """Create a DNA-like helix pattern from the image"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0

    def process(self, frame):
        height, width = frame.shape[:2]
        result = np.zeros_like(frame)

        # Parameters for double helix
        amplitude = width // 6
        frequency = 2.0 * np.pi / height * 3  # 3 full waves
        phase = self.time * 0.05
        strand_width = 15

        # For each row in the image
        for y in range(height):
            # Calculate x positions for both strands of the helix
            x1 = int(width / 2 + amplitude * math.sin(frequency * y + phase))
            x2 = int(width / 2 + amplitude * math.sin(frequency * y + phase + np.pi))  # 180 degrees out of phase

            # Draw the strands
            for dx in range(-strand_width, strand_width + 1):
                # First strand
                x_pos1 = x1 + dx
                if 0 <= x_pos1 < width:
                    # Get color from original image
                    result[y, x_pos1] = frame[y, x_pos1]

                # Second strand
                x_pos2 = x2 + dx
                if 0 <= x_pos2 < width:
                    # Get color from original image
                    result[y, x_pos2] = frame[y, x_pos2]

            # Draw connecting "rungs" periodically
            if y % 20 < 2:
                # Draw line between x1 and x2
                start_x = min(x1, x2)
                end_x = max(x1, x2)
                for x in range(start_x, end_x + 1):
                    if 0 <= x < width:
                        result[y, x] = frame[y, x]

        self.time += 1
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

@register_effect("quantum")
class QuantumEffect(Effect):
    """Create a quantum probability wave collapse effect"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.frames = []
        self.max_frames = 10
        self.time = 0

    def process(self, frame):
        # Store frame history
        self.frames.append(frame.copy())
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

        if len(self.frames) < 3:
            return frame

        # Create quantum superposition effect
        height, width = frame.shape[:2]
        result = np.zeros_like(frame)

        # Create probability map (simplifed as noise)
        probability = np.random.random((height, width)) * 0.6 + 0.2  # Values between 0.2 and 0.8

        # Add wave pattern to probability
        for y in range(height):
            for x in range(width):
                wave = 0.2 * math.sin(x / 20.0 + self.time * 0.1) * math.cos(y / 20.0 + self.time * 0.1)
                probability[y, x] += wave

        # Clip probability to 0-1 range
        probability = np.clip(probability, 0, 1)

        # For each pixel, select from frames based on probability
        for y in range(height):
            for x in range(width):
                # Get probability for this pixel
                p = probability[y, x]

                # Select frame based on probability
                frame_idx = min(len(self.frames) - 1, int(p * len(self.frames)))

                # Get pixel from selected frame
                result[y, x] = self.frames[frame_idx][y, x]

        self.time += 1
        return result

@register_effect("oil_painting")
class OilPaintingEffect(Effect):
    """Create oil painting effect with dynamic texture"""
    def process(self, frame):
        # Parameters
        radius = 4
        intensity_levels = 10

        # Convert to grayscale for intensity
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create output image
        result = np.zeros_like(frame)
        height, width = frame.shape[:2]

        # Process with reduced resolution for performance
        step = 2

        for y in range(0, height, step):
            for x in range(0, width, step):
                # Define neighborhood
                y_min = max(0, y - radius)
                y_max = min(height, y + radius + 1)
                x_min = max(0, x - radius)
                x_max = min(width, x + radius + 1)

                # Get neighborhood pixels
                neighborhood = gray[y_min:y_max, x_min:x_max]

                # Quantize intensities
                intensities = (neighborhood * intensity_levels / 255).astype(np.int32)

                # Count occurrences of each intensity
                intensity_count = np.zeros(intensity_levels, dtype=np.int32)
                for i in range(intensity_levels):
                    intensity_count[i] = np.sum(intensities == i)

                # Find most common intensity
                max_intensity = np.argmax(intensity_count)

                # Map back to grayscale
                target_intensity = max_intensity * 255 / intensity_levels

                # Find pixels close to target intensity
                mask = np.abs(neighborhood - target_intensity) < 255 / intensity_levels
                if np.sum(mask) > 0:
                    # Get average color of matching pixels
                    region = frame[y_min:y_max, x_min:x_max]
                    mask_3d = np.stack([mask] * 3, axis=2)
                    avg_color = np.mean(region[mask_3d], axis=0)

                    # Apply to step x step region
                    for dy in range(step):
                        for dx in range(step):
                            if y+dy < height and x+dx < width:
                                result[y+dy, x+dx] = avg_color

        return result

@register_effect("wigglevision")
class WiggleVisionEffect(Effect):
    """Create a wiggly, wobbly vision effect"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0

    def process(self, frame):
        height, width = frame.shape[:2]

        # Create map for distortion
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)

        # Calculate distortion with time-based wobble
        time_val = self.time * 0.1

        for y in range(height):
            for x in range(width):
                # Multi-frequency wobble
                offset_x = 15 * math.sin(y / 30.0 + time_val) + 8 * math.sin(x / 60.0 + time_val * 1.5)
                offset_y = 10 * math.cos(x / 40.0 + time_val * 0.8) + 6 * math.sin(y / 50.0 + time_val * 1.2)

                # Apply offset
                map_x[y, x] = x + offset_x
                map_y[y, x] = y + offset_y

        # Apply distortion
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        # Add color fringing (chromatic aberration)
        b, g, r = cv2.split(result)

        # Shift red and blue channels slightly
        r_shifted = np.roll(r, 5, axis=1)
        b_shifted = np.roll(b, -5, axis=1)

        # Recombine channels
        result = cv2.merge([b_shifted, g, r_shifted])

        self.time += 1
        return result

@register_effect("cellular")
class CellularAutomatonEffect(Effect):
    """Apply cellular automaton rules to create evolving patterns"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
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

@register_effect("disintegrate")
class DisintegrateEffect(Effect):
   """Create a particle disintegration effect"""
   def __init__(self, audio_manager=None):
       super().__init__(audio_manager)
       self.particles = None
       self.time = 0

   def process(self, frame):
       height, width = frame.shape[:2]

       # Initialize particles if not already done
       if self.particles is None:
           # Sample points from the image
           num_particles = 5000
           self.particles = []

           # Generate random particles
           for _ in range(num_particles):
               x = random.randint(0, width - 1)
               y = random.randint(0, height - 1)
               color = frame[y, x].tolist()

               # Random velocity
               vx = random.uniform(-1, 1)
               vy = random.uniform(-1, 1)

               # Random lifetime
               lifetime = random.randint(20, 100)

               self.particles.append({
                   'x': x, 'y': y,
                   'orig_x': x, 'orig_y': y,
                   'vx': vx, 'vy': vy,
                   'color': color,
                   'lifetime': lifetime,
                   'age': 0
               })

       # Create output image
       result = np.zeros_like(frame)

       # Update particles and draw them
       alive_particles = []

       for particle in self.particles:
           # Update age
           particle['age'] += 1

           # Check if still alive
           if particle['age'] <= particle['lifetime']:
               # Calculate force back to original position (elastic)
               force_x = (particle['orig_x'] - particle['x']) * 0.001
               force_y = (particle['orig_y'] - particle['y']) * 0.001

               # Add random force
               force_x += random.uniform(-0.1, 0.1)
               force_y += random.uniform(-0.1, 0.1)

               # Add disintegration force based on time
               disintegration = min(1.0, self.time / 100.0)
               force_x += random.uniform(-0.5, 0.5) * disintegration
               force_y += random.uniform(-0.5, 0.5) * disintegration

               # Update velocity
               particle['vx'] += force_x
               particle['vy'] += force_y

               # Apply damping
               particle['vx'] *= 0.95
               particle['vy'] *= 0.95

               # Update position
               particle['x'] += particle['vx']
               particle['y'] += particle['vy']

               # Draw particle if within bounds
               x, y = int(particle['x']), int(particle['y'])
               if 0 <= x < width and 0 <= y < height:
                   result[y, x] = particle['color']

               # Keep particle
               alive_particles.append(particle)

       # Replace particles list with alive particles
       self.particles = alive_particles

       # Reset if all particles have died
       if len(self.particles) == 0:
           self.particles = None
           self.time = 0

       self.time += 1
       return result

@register_effect("hypnotic")
class HypnoticEffect(Effect):
   """Create mesmerizing hypnotic spiral patterns"""
   def __init__(self, audio_manager=None):
       super().__init__(audio_manager)
       self.time = 0

   def process(self, frame):
       height, width = frame.shape[:2]

       # Create spiral pattern
       center_x, center_y = width // 2, height // 2
       spiral = np.zeros((height, width), dtype=np.uint8)

       # Calculate spiral pattern
       for y in range(height):
           for x in range(width):
               # Distance from center
               dx = x - center_x
               dy = y - center_y
               distance = math.sqrt(dx*dx + dy*dy)

               # Angle
               angle = math.atan2(dy, dx)

               # Spiral function - combination of distance and angle
               spiral_value = (distance / 10 + angle * 5 / math.pi + self.time * 0.1) % 2

               # Create alternating bands
               spiral[y, x] = 255 if spiral_value < 1 else 0

       # Create hypnotic color effect
       spiral_bgr = np.zeros_like(frame)

       # Psychedelic color mapping
       for y in range(height):
           for x in range(width):
               # Create hue based on distance and angle
               dx = x - center_x
               dy = y - center_y
               distance = math.sqrt(dx*dx + dy*dy)
               angle = math.atan2(dy, dx)

               # Calculate hue (0-179 for OpenCV)
               hue = int((angle + math.pi) * 90 / math.pi + self.time * 2) % 180

               # Create HSV color
               hsv_color = np.array([hue, 255, spiral[y, x]], dtype=np.uint8)

               # Convert to BGR
               bgr_color = cv2.cvtColor(hsv_color.reshape(1, 1, 3), cv2.COLOR_HSV2BGR).reshape(3)

               # Assign to output
               spiral_bgr[y, x] = bgr_color

       # Blend with original image
       result = cv2.addWeighted(frame, 0.4, spiral_bgr, 0.6, 0)

       self.time += 1
       return result

@register_effect("impressionist")
class ImpressionistEffect(Effect):
   """Create an impressionist painting style effect"""
   def process(self, frame):
       # Create multiple brush stroke layers
       height, width = frame.shape[:2]
       canvas = np.zeros_like(frame)

       # Number of brush strokes
       num_strokes = 5000

       # Create random brush strokes
       for _ in range(num_strokes):
           # Random position
           x = random.randint(0, width - 1)
           y = random.randint(0, height - 1)

           # Random brush size
           size = random.randint(3, 10)

           # Get color from original image
           color = frame[min(y, height-1), min(x, width-1)].tolist()

           # Create brush stroke (simple circle)
           cv2.circle(canvas, (x, y), size, color, -1)

       # Blend with original image for details
       result = cv2.addWeighted(canvas, 0.8, frame, 0.2, 0)

       return result

@register_effect("symmetry")
class SymmetryEffect(Effect):
   """Create kaleidoscopic symmetry patterns"""
   def process(self, frame):
       height, width = frame.shape[:2]
       center_x, center_y = width // 2, height // 2

       # Choose number of reflections (between 4 and 12)
       num_reflections = random.randint(4, 12)

       # Create output canvas
       result = np.zeros_like(frame)

       # Calculate segment angle
       segment_angle = 2 * math.pi / num_reflections

       # For each pixel in the output
       for y in range(height):
           for x in range(width):
               # Calculate position relative to center
               dx = x - center_x
               dy = y - center_y

               # Calculate distance and angle
               distance = math.sqrt(dx*dx + dy*dy)
               angle = math.atan2(dy, dx)

               # Normalize angle to first segment
               normalized_angle = angle % segment_angle

               # Calculate source coordinates in the first segment
               src_x = int(center_x + distance * math.cos(normalized_angle))
               src_y = int(center_y + distance * math.sin(normalized_angle))

               # Ensure within bounds
               if 0 <= src_x < width and 0 <= src_y < height:
                   # Copy color from source
                   result[y, x] = frame[src_y, src_x]

       return result

@register_effect("warhol")
class WarholEffect(Effect):
   """Create a Warhol-inspired pop art effect"""
   def process(self, frame):
       height, width = frame.shape[:2]

       # Create four panels
       half_h, half_w = height // 2, width // 2

       # Convert to grayscale for consistent tone
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       # Create high contrast version
       _, high_contrast = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

       # Create four different color palettes
       palettes = [
           [(255, 0, 0), (255, 255, 0)],  # Red + Yellow
           [(0, 255, 255), (0, 0, 255)],  # Cyan + Blue
           [(255, 0, 255), (0, 255, 0)],  # Magenta + Green
           [(255, 165, 0), (128, 0, 128)]  # Orange + Purple
       ]

       # Create output canvas
       result = np.zeros_like(frame)

       # Fill each quadrant with different color palette
       for q in range(4):
           # Determine quadrant coordinates
           y_start = (q // 2) * half_h
           x_start = (q % 2) * half_w

           # Get palette
           bg_color, fg_color = palettes[q]

           # Apply colors based on threshold
           for y in range(half_h):
               for x in range(half_w):
                   if y + y_start < height and x + x_start < width:
                       src_y, src_x = y + y_start - half_h * (q // 2), x + x_start - half_w * (q % 2)
                       if 0 <= src_y < height and 0 <= src_x < width:
                           if high_contrast[src_y, src_x] > 127:
                               result[y + y_start, x + x_start] = fg_color
                           else:
                               result[y + y_start, x + x_start] = bg_color

       return result

@register_effect("chromadepth")
class ChromaDepthEffect(Effect):
   """Create depth perception using chromatic aberration"""
   def process(self, frame):
       # Convert to grayscale for depth
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       # Create depth map (simpler approach - just use intensity)
       depth = gray.copy()

       # Create output with color based on depth
       height, width = frame.shape[:2]
       result = np.zeros_like(frame)

       # Color map from red (near) to blue (far)
       for y in range(height):
           for x in range(width):
               depth_value = depth[y, x]

               # Map depth to hue (0-179 for OpenCV)
               # Red (0) = near, Blue (120) = far
               hue = 120 - int(depth_value * 120 / 255)

               # Create HSV color
               hsv_color = np.array([hue, 255, 255], dtype=np.uint8)

               # Convert to BGR
               bgr_color = cv2.cvtColor(hsv_color.reshape(1, 1, 3), cv2.COLOR_HSV2BGR).reshape(3)

               # Assign to output
               result[y, x] = bgr_color

       # Add chromatic aberration for enhanced depth effect
       # Split channels
       b, g, r = cv2.split(result)

       # Shift channels based on depth
       for y in range(height):
           for x in range(width):
               # Get depth value (0-255)
               d = depth[y, x]

               # Calculate shift amount (greater shift for near objects)
               shift = int((255 - d) / 64)  # 0-4 pixels

               # Shift red channel right (for pixels that have room to shift)
               if x + shift < width:
                   r[y, x + shift] = r[y, x]

               # Shift blue channel left
               if x - shift >= 0:
                   b[y, x - shift] = b[y, x]

       # Recombine channels
       result = cv2.merge([b, g, r])

       return result

@register_effect("reality_break")
class RealityBreakEffect(Effect):
    """Complete reality-shattering psychedelic breakdown"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.glitch = GlitchEffect(audio_manager)
        self.datamosh = DatamoshEffect(audio_manager)
        self.kaleidoscope = KaleidoscopeEffect(audio_manager)
        self.fractal = FractalNoiseEffect(audio_manager)
        self.time = 0

    def process(self, frame):
        # Apply effects in sequence with varying intensities
        result = frame.copy()

        # Apply fractal noise with time-varying blend
        fractal = self.fractal.process(result)
        alpha = (math.sin(self.time * 0.1) + 1) * 0.3
        result = cv2.addWeighted(result, 1-alpha, fractal, alpha, 0)

        # Apply datamoshing
        result = self.datamosh.process(result)

        # Apply aggressive glitch
        for _ in range(3):
            result = self.glitch.process(result)

        # Split into RGB channels and manipulate
        b, g, r = cv2.split(result)

        # Shift channels based on time
        shift_x = int(20 * math.sin(self.time * 0.2))
        shift_y = int(20 * math.cos(self.time * 0.2))

        b = np.roll(b, shift_x, axis=1)
        b = np.roll(b, shift_y, axis=0)
        r = np.roll(r, -shift_x, axis=1)
        r = np.roll(r, -shift_y, axis=0)

        # Recombine with channel mixture
        result = cv2.merge([
            b * 0.9 + g * 0.3,
            g * 0.7 + r * 0.4,
            r * 0.8 + b * 0.5
        ]).astype(np.uint8)

        # Apply kaleidoscope effect
        result = self.kaleidoscope.process(result)

        self.time += 1
        return result

@register_effect("nightmare")
class NightmareEffect(Effect):
    """Dark, unsettling visual distortion experience"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0
        self.prev_frames = []
        self.max_frames = 10

    def process(self, frame):
        # Store frame history
        self.prev_frames.append(frame.copy())
        if len(self.prev_frames) > self.max_frames:
            self.prev_frames.pop(0)

        height, width = frame.shape[:2]
        result = np.zeros_like(frame)

        # Create dark, high-contrast version
        dark = frame.copy()
        dark = cv2.convertScaleAbs(dark, alpha=1.5, beta=-50)

        # Add vignette
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        cv2.ellipse(mask, center, (width//3, height//3), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        mask_inv = 255 - mask

        # Apply vignette
        for c in range(3):
            result[:,:,c] = (dark[:,:,c] * (mask / 255.0) + dark[:,:,c] * 0.2 * (mask_inv / 255.0)).astype(np.uint8)

        # Create pulsing distortion
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)

        for y in range(height):
            for x in range(width):
                # Distance from center
                dx = x - width // 2
                dy = y - height // 2
                dist = math.sqrt(dx*dx + dy*dy)

                # Angle to center
                angle = math.atan2(dy, dx)

                # Distort more at edges
                distortion = dist / (width//2) * 30

                # Pulsing distortion
                pulse = math.sin(dist/20 - self.time * 0.1) * distortion

                # Calculate new coordinates
                new_x = x + pulse * math.cos(angle)
                new_y = y + pulse * math.sin(angle)

                map_x[y, x] = new_x
                map_y[y, x] = new_y

        # Apply distortion
        result = cv2.remap(result, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Add ghosting from previous frames
        if len(self.prev_frames) > 3:
            ghost_frame = self.prev_frames[-3]
            # Convert to grayscale and increase contrast
            ghost_gray = cv2.cvtColor(ghost_frame, cv2.COLOR_BGR2GRAY)
            _, ghost_mask = cv2.threshold(ghost_gray, 50, 255, cv2.THRESH_BINARY)
            ghost_mask = cv2.GaussianBlur(ghost_mask, (21, 21), 0)

            # Create ghost overlay
            ghost = cv2.cvtColor(ghost_mask, cv2.COLOR_GRAY2BGR)
            ghost = ghost * np.array([0.2, 0.0, 0.3], dtype=np.float32).reshape(1, 1, 3)

            # Add ghost to result
            result = cv2.add(result, ghost.astype(np.uint8))

        # Add random noise flashes
        if random.random() < 0.1:
            noise_area = np.random.randint(0, 2, (height, width, 3), dtype=np.uint8) * 255
            alpha = random.uniform(0.1, 0.3)
            result = cv2.addWeighted(result, 1-alpha, noise_area, alpha, 0)

        self.time += 1
        return result

@register_effect("acid_trip")
class AcidTripEffect(Effect):
    """Intensely colorful, flowing visual hallucination"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0

    def process(self, frame):
        height, width = frame.shape[:2]

        # Create flow field for distortion
        flow_x = np.zeros((height, width), np.float32)
        flow_y = np.zeros((height, width), np.float32)

        # Generate perlin-like noise flow field
        for y in range(0, height, 4):
            for x in range(0, width, 4):
                # Create flowing pattern
                angle = math.sin(x/30 + self.time*0.05) * math.cos(y/20 - self.time*0.07) * math.pi

                # Convert angle to flow vector
                fx = 30 * math.cos(angle)
                fy = 30 * math.sin(angle)

                # Fill 4x4 blocks
                for dy in range(4):
                    for dx in range(4):
                        if y+dy < height and x+dx < width:
                            flow_x[y+dy, x+dx] = fx
                            flow_y[y+dy, x+dx] = fy

        # Create distortion maps
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)

        for y in range(height):
            for x in range(width):
                map_x[y, x] = x + flow_x[y, x]
                map_y[y, x] = y + flow_y[y, x]

        # Apply distortion
        distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        # Enhance colors dramatically
        hsv = cv2.cvtColor(distorted, cv2.COLOR_BGR2HSV)

        # Cycle hues over time
        hsv[:, :, 0] = (hsv[:, :, 0] + int(self.time * 2)) % 180

        # Max out saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2, 0, 255).astype(np.uint8)

        # Enhanced version
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Add bloom/glow effect
        glow = cv2.GaussianBlur(enhanced, (21, 21), 0)
        result = cv2.addWeighted(enhanced, 0.7, glow, 0.5, 0)

        # Add fractals
        for i in range(5):
            # Create fractal pattern
            fractal = np.zeros((height, width), dtype=np.uint8)

            center_x = int(width/2 + width/4 * math.sin(self.time * 0.03 + i))
            center_y = int(height/2 + height/4 * math.cos(self.time * 0.05 + i))

            for y in range(height):
                for x in range(width):
                    dx = x - center_x
                    dy = y - center_y
                    dist = math.sqrt(dx*dx + dy*dy)
                    angle = math.atan2(dy, dx)

                    # Create spiral pattern
                    val = (dist/10 + angle*3/math.pi + self.time*0.1 + i) % 1
                    fractal[y, x] = 255 if val < 0.5 else 0

            # Convert to color
            hue = (self.time * 3 + i * 30) % 180
            fractal_color = np.zeros_like(frame)
            fractal_mask = fractal > 127

            # Create colored mask
            color_hsv = np.zeros((1, 1, 3), dtype=np.uint8)
            color_hsv[0, 0, 0] = hue
            color_hsv[0, 0, 1] = 255
            color_hsv[0, 0, 2] = 255
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)

            # Apply color to fractal pattern
            fractal_color[fractal_mask] = color_bgr[0, 0]

            # Blend with result
            alpha = 0.1 + 0.1 * math.sin(self.time * 0.1 + i)
            result = cv2.addWeighted(result, 1-alpha, fractal_color, alpha, 0)

        self.time += 1
        return result

@register_effect("dimension_rift")
class DimensionRiftEffect(Effect):
    """Creates a tear between dimensions with chaotic visuals"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0
        self.mirror = MirrorEffect(audio_manager)
        self.liquid = LiquidEffect(audio_manager)
        self.matrix = MatrixEffect(audio_manager)

    def process(self, frame):
        height, width = frame.shape[:2]

        # Create dimension rift mask
        rift_mask = np.zeros((height, width), dtype=np.uint8)

        # Animate rift position
        center_x = width // 2 + int(width // 4 * math.sin(self.time * 0.05))
        center_y = height // 2 + int(height // 4 * math.cos(self.time * 0.07))

        # Create curved rift
        points = []
        num_points = 20
        for i in range(num_points):
            t = i / (num_points - 1)
            x = int(center_x + width * 0.4 * math.sin(t * math.pi * 2 + self.time * 0.1))
            y = int(center_y + height * 0.4 * math.cos(t * math.pi * 2 + self.time * 0.13))
            points.append((x, y))

        # Draw rift
        points = np.array(points, dtype=np.int32)
        rift_thickness = 50 + int(20 * math.sin(self.time * 0.2))
        cv2.polylines(rift_mask, [points], False, 255, rift_thickness)

        # Blur rift
        rift_mask = cv2.GaussianBlur(rift_mask, (51, 51), 0)

        # Process different dimensions
        dimension1 = self.mirror.process(frame)
        dimension2 = self.liquid.process(frame)

        # Create chaotic matrix for the rift itself
        rift_content = self.matrix.process(frame)

        # Combine dimensions based on mask
        result = np.zeros_like(frame)

        for y in range(height):
            for x in range(width):
                mask_value = rift_mask[y, x] / 255.0

                # Rift area
                if mask_value > 0.7:
                    result[y, x] = rift_content[y, x]
                # Blend between dimensions
                elif mask_value > 0:
                    result[y, x] = (dimension1[y, x] * (1 - mask_value) +
                                  dimension2[y, x] * mask_value).astype(np.uint8)
                # Outside rift
                else:
                    result[y, x] = dimension1[y, x]

        # Add energy particles around rift
        for _ in range(100):
            # Random angle around the rift
            angle = random.uniform(0, math.pi * 2)
            distance = random.uniform(rift_thickness * 0.5, rift_thickness * 0.9)

            # Find position on the rift line
            t = random.uniform(0, 1)
            i = min(int(t * (len(points) - 1)), len(points) - 2)
            rift_x = int(points[i][0] * (1-t) + points[i+1][0] * t)
            rift_y = int(points[i][1] * (1-t) + points[i+1][1] * t)

            # Position particle
            x = int(rift_x + distance * math.cos(angle))
            y = int(rift_y + distance * math.sin(angle))

            # Draw particle if within bounds
            if 0 <= x < width and 0 <= y < height:
                # Random color (energy colors: blue, cyan, purple)
                color_choice = random.randint(0, 2)
                if color_choice == 0:
                    color = (255, 0, 0)  # Blue
                elif color_choice == 1:
                    color = (255, 255, 0)  # Cyan
                else:
                    color = (255, 0, 255)  # Purple

                # Draw particle
                size = random.randint(1, 3)
                cv2.circle(result, (x, y), size, color, -1)

        self.time += 1
        return result

@register_effect("chaos_theory")
class ChaosTheoryEffect(Effect):
    """Cascading butterfly effect of visual chaos"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0
        self.prev_frames = []
        self.max_frames = 20
        self.reset_counter = 0

    def process(self, frame):
        height, width = frame.shape[:2]

        # Store frame history
        self.prev_frames.append(frame.copy())
        if len(self.prev_frames) > self.max_frames:
            self.prev_frames.pop(0)

        # Create butterfly shape mask that evolves over time
        butterfly = np.zeros((height, width), dtype=np.uint8)

        # Animate butterfly position
        center_x = width // 2
        center_y = height // 2

        # Draw evolving butterfly
        t = self.time * 0.05
        for r in range(1, 100):
            for theta in np.linspace(0, 2 * np.pi, 100):
                # Butterfly curve formula (simplified)
                r_mod = r * (math.exp(math.cos(theta)) - 2*math.cos(4*theta) + math.sin(theta/12)**5)

                # Add time variation
                r_mod *= 0.5 + 0.2 * math.sin(t + r/20)

                # Calculate x, y
                x = int(center_x + r_mod * math.cos(theta + t))
                y = int(center_y + r_mod * math.sin(theta + t))

                # Draw if within bounds
                if 0 <= x < width and 0 <= y < height:
                    butterfly[y, x] = 255

        # Dilate and blur the butterfly
        butterfly = cv2.dilate(butterfly, np.ones((3, 3), np.uint8), iterations=2)
        butterfly = cv2.GaussianBlur(butterfly, (7, 7), 0)

        # Create output canvas
        result = np.zeros_like(frame)

        # Create butterfly effect using frame history
        if len(self.prev_frames) > 5:
            for y in range(0, height, 2):  # Skip pixels for performance
                for x in range(0, width, 2):
                    # Get butterfly mask value (0-255)
                    mask_value = butterfly[y, x]

                    if mask_value > 0:
                        # Calculate which historical frame to use based on position
                        # This creates chaotic dependency on position
                        chaos_value = (x + y + int(50 * math.sin(x/20 + y/20 + t))) % len(self.prev_frames)
                        frame_idx = chaos_value

                        # Apply to 2x2 block for performance
                        for dy in range(2):
                            for dx in range(2):
                                if y+dy < height and x+dx < width:
                                    result[y+dy, x+dx] = self.prev_frames[frame_idx][y+dy, x+dx]
                    else:
                        # Outside butterfly
                        for dy in range(2):
                            for dx in range(2):
                                if y+dy < height and x+dx < width:
                                    result[y+dy, x+dx] = frame[y+dy, x+dx]
        else:
            result = frame.copy()

        # Add fractal noise overlay
        noise = np.zeros((height, width), dtype=np.float32)
        scale = 30.0

        for y in range(0, height, 4):
            for x in range(0, width, 4):
                # Multi-octave noise
                value = 0
                amplitude = 1.0
                frequency = 1.0

                for _ in range(3):
                    nx = x / width * scale * frequency
                    ny = y / height * scale * frequency
                    nt = t * frequency

                    # Simple hash for pseudo-random noise
                    n = math.sin(nx * 12.9898 + ny * 78.233 + nt * 43.2364) * 43758.5453
                    value += (n - math.floor(n)) * amplitude

                    amplitude *= 0.5
                    frequency *= 2.0

                # Fill 4x4 block
                for dy in range(4):
                    for dx in range(4):
                        if y+dy < height and x+dx < width:
                            noise[y+dy, x+dx] = value

        # Normalize noise
        noise = (noise * 50).astype(np.uint8)

        # Apply noise to result
        for c in range(3):
            result[:,:,c] = np.clip(result[:,:,c].astype(np.int32) + noise - 25, 0, 255).astype(np.uint8)

        # Occasionally reset the system (butterfly effect reset)
        self.reset_counter += 1
        if self.reset_counter > 100:
            if random.random() < 0.02:
                self.prev_frames = [frame.copy()]
                self.reset_counter = 0

        self.time += 1
        return result


@register_effect("art_nouveau_lines")
class ArtNouveauLinesEffect(Effect):
    """Create flowing, organic line patterns in Art Nouveau style"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0

    def process(self, frame):
        height, width = frame.shape[:2]

        # Create flowing curve patterns
        curves = np.zeros((height, width), dtype=np.uint8)

        # Generate organic, flowing curves
        num_curves = 15
        for i in range(num_curves):
            # Create control points for cubic Bezier curves
            points = []
            num_segments = 5

            # First point
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            points.append((start_x, start_y))

            # Create flowing sequence of points
            for j in range(num_segments):
                # Direction with slight variation
                angle = math.pi * 2 * (j + i) / num_segments + self.time * 0.01
                length = random.randint(50, 150)

                # Next point
                next_x = int(points[-1][0] + length * math.cos(angle))
                next_y = int(points[-1][1] + length * math.sin(angle))
                points.append((next_x, next_y))

            # Convert to numpy array
            points = np.array(points, dtype=np.int32)

            # Draw curve with varying thickness
            thickness = random.randint(1, 3)
            cv2.polylines(curves, [points], False, 255, thickness)

        # Add swirls and ornamental details
        num_swirls = 8
        for _ in range(num_swirls):
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)

            # Create spiral swirl
            for radius in range(5, 50, 2):
                for angle in np.linspace(0, 6 * np.pi, 150):
                    x = int(center_x + radius * math.cos(angle) * angle/10)
                    y = int(center_y + radius * math.sin(angle) * angle/10)

                    if 0 <= x < width and 0 <= y < height:
                        curves[y, x] = 255

        # Dilate and blur the curves
        curves = cv2.dilate(curves, np.ones((2, 2), np.uint8), iterations=1)
        curves = cv2.GaussianBlur(curves, (3, 3), 0)

        # Create a colored version with Art Nouveau palette
        # Convert to 3-channel
        curves_bgr = cv2.cvtColor(curves, cv2.COLOR_GRAY2BGR)

        # Art Nouveau gold color
        gold = np.array([32, 165, 218], dtype=np.uint8)  # BGR format
        curves_colored = curves_bgr * gold / 255

        # Blend with original image
        result = cv2.addWeighted(frame, 0.7, curves_colored.astype(np.uint8), 0.8, 0)

        self.time += 1
        return result

@register_effect("mucha_portrait")
class MuchaPortraitEffect(Effect):
    """Transform portrait in Alphonse Mucha style with ornamental borders"""
    def process(self, frame):
        height, width = frame.shape[:2]

        # Create ornamental border
        border_width = width // 6
        inner_width = width - 2 * border_width
        inner_height = height - 2 * border_width

        # Create mask for inner image
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (border_width, border_width),
                     (width - border_width, height - border_width), 255, -1)

        # Create stylized border
        border = np.zeros((height, width, 3), dtype=np.uint8)

        # Fill border with muted background color
        border_color = np.array([210, 180, 140], dtype=np.uint8)  # Tan color
        border[:, :] = border_color

        # Add ornamental patterns to border
        # Top border
        for x in range(border_width, width - border_width, 20):
            # Draw floral pattern
            cv2.circle(border, (x, border_width // 2), 8, (70, 100, 110), -1)
            for angle in range(0, 360, 45):
                end_x = int(x + 12 * math.cos(math.radians(angle)))
                end_y = int(border_width // 2 + 12 * math.sin(math.radians(angle)))
                cv2.line(border, (x, border_width // 2), (end_x, end_y), (50, 80, 90), 2)

        # Bottom border (mirror of top)
        for x in range(border_width, width - border_width, 20):
            cv2.circle(border, (x, height - border_width // 2), 8, (70, 100, 110), -1)
            for angle in range(0, 360, 45):
                end_x = int(x + 12 * math.cos(math.radians(angle)))
                end_y = int(height - border_width // 2 + 12 * math.sin(math.radians(angle)))
                cv2.line(border, (x, height - border_width // 2), (end_x, end_y), (50, 80, 90), 2)

        # Left border
        for y in range(border_width, height - border_width, 20):
            cv2.circle(border, (border_width // 2, y), 8, (70, 100, 110), -1)
            for angle in range(0, 360, 45):
                end_x = int(border_width // 2 + 12 * math.cos(math.radians(angle)))
                end_y = int(y + 12 * math.sin(math.radians(angle)))
                cv2.line(border, (border_width // 2, y), (end_x, end_y), (50, 80, 90), 2)

        # Right border (mirror of left)
        for y in range(border_width, height - border_width, 20):
            cv2.circle(border, (width - border_width // 2, y), 8, (70, 100, 110), -1)
            for angle in range(0, 360, 45):
                end_x = int(width - border_width // 2 + 12 * math.cos(math.radians(angle)))
                end_y = int(y + 12 * math.sin(math.radians(angle)))
                cv2.line(border, (width - border_width // 2, y), (end_x, end_y), (50, 80, 90), 2)

        # Stylize the inner image
        inner = frame.copy()

        # Apply poster effect (reduce colors)
        inner = inner // 32 * 32

        # Enhance edges in Art Nouveau style
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        edges_colored = np.zeros_like(inner)
        edges_colored[edges > 0] = [0, 0, 0]  # Black outline

        # Overlay edges
        inner = cv2.subtract(inner, edges_colored)

        # Combine border and inner image
        result = border.copy()
        mask_3d = np.stack([mask] * 3, axis=2) > 0
        result[mask_3d] = inner[mask_3d]

        return result

@register_effect("stained_glass")
class StainedGlassEffect(Effect):
    """Create Art Nouveau stained glass effect with flowing lead lines"""
    def process(self, frame):
        height, width = frame.shape[:2]

        # Step 1: Create regions for the stained glass
        # Start with edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Dilate edges to create thicker lead lines
        lead_lines = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        # Add flowing curves for Art Nouveau style
        num_curves = 15
        for _ in range(num_curves):
            # Create flowing curve
            points = []
            num_points = 10

            # Start point
            start_x = random.randint(0, width)
            start_y = random.randint(0, height)
            points.append((start_x, start_y))

            # Create flowing curve
            for i in range(1, num_points):
                # Previous point
                prev_x, prev_y = points[i-1]

                # Random angle with continuity
                if i == 1:
                    angle = random.uniform(0, 2 * np.pi)
                else:
                    prev_angle = math.atan2(points[i-1][1] - points[i-2][1],
                                          points[i-1][0] - points[i-2][0])
                    angle = prev_angle + random.uniform(-np.pi/4, np.pi/4)

                # Distance
                distance = random.randint(20, 50)

                # New point
                new_x = int(prev_x + distance * math.cos(angle))
                new_y = int(prev_y + distance * math.sin(angle))

                # Ensure within bounds
                new_x = max(0, min(width-1, new_x))
                new_y = max(0, min(height-1, new_y))

                points.append((new_x, new_y))

            # Convert to numpy array
            points = np.array(points, dtype=np.int32)

            # Draw curve
            cv2.polylines(lead_lines, [points], False, 255, 2)

        # Step 2: Create mosaic-like regions
        regions = np.zeros_like(gray)
        num_seeds = 50
        seeds = []

        # Create random seed points
        for _ in range(num_seeds):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            seeds.append((x, y))
            regions[y, x] = random.randint(1, num_seeds)

        # Grow regions (simplified watershed-like approach)
        for _ in range(5):  # Number of iterations
            new_regions = regions.copy()
            for y in range(1, height-1):
                for x in range(1, width-1):
                    if lead_lines[y, x] == 0:  # Not on a lead line
                        # Check neighbors
                        neighbors = [
                            regions[y-1, x],
                            regions[y+1, x],
                            regions[y, x-1],
                            regions[y, x+1]
                        ]
                        neighbors = [n for n in neighbors if n > 0]
                        if neighbors:
                            new_regions[y, x] = random.choice(neighbors)
            regions = new_regions

        # Step 3: Color each region with Art Nouveau palette
        result = np.zeros_like(frame)

        # Art Nouveau palette (BGR format)
        palette = [
            (190, 190, 190),  # Silver
            (32, 165, 218),   # Gold
            (98, 122, 157),   # Muted blue
            (71, 112, 132),   # Teal
            (87, 65, 47),     # Brown
            (34, 87, 104),    # Red-brown
            (45, 82, 160),    # Maroon
            (164, 178, 127),  # Pale green
            (156, 156, 220),  # Pale gold
            (107, 142, 35)    # Olive
        ]

        # Assign colors to regions
        for region_id in range(1, num_seeds + 1):
            mask = regions == region_id
            if np.any(mask):
                # Get average color from original image
                avg_color = np.mean(frame[mask], axis=0).astype(np.uint8)

                # Find closest palette color
                min_dist = float('inf')
                best_color = palette[0]
                for color in palette:
                    dist = np.sum((avg_color - np.array(color))**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_color = color

                # Apply color
                result[mask] = best_color

        # Step 4: Add lead lines in black
        result[lead_lines > 0] = [0, 0, 0]

        # Step 5: Add some translucency and glow
        # Blend with original image for some detail
        result = cv2.addWeighted(result, 0.8, frame, 0.2, 0)

        # Add subtle glow
        glow = cv2.GaussianBlur(result, (9, 9), 0)
        result = cv2.addWeighted(result, 0.85, glow, 0.15, 0)

        return result

@register_effect("floral_frame")
class FloralFrameEffect(Effect):
    """Apply decorative Art Nouveau floral frame"""
    def process(self, frame):
        height, width = frame.shape[:2]

        # Create output canvas
        result = frame.copy()

        # Create mask for frame area
        frame_width = width // 10
        frame_mask = np.zeros((height, width), dtype=np.uint8)

        # Draw outer rectangle (border)
        cv2.rectangle(frame_mask, (0, 0), (width, height), 255, -1)
        # Draw inner rectangle (hole)
        cv2.rectangle(frame_mask, (frame_width, frame_width),
                     (width - frame_width, height - frame_width), 0, -1)

        # Create ornamental frame
        frame_img = np.zeros_like(frame)

        # Base color - cream
        frame_img[frame_mask > 0] = [200, 225, 235]  # Light cream (BGR)

        # Create floral patterns
        # Corners
        corners = [
            (frame_width // 2, frame_width // 2),  # Top-left
            (width - frame_width // 2, frame_width // 2),  # Top-right
            (frame_width // 2, height - frame_width // 2),  # Bottom-left
            (width - frame_width // 2, height - frame_width // 2)  # Bottom-right
        ]

        for corner in corners:
            center_x, center_y = corner

            # Draw flower
            cv2.circle(frame_img, corner, frame_width // 2, (100, 180, 210), -1)  # Gold center

            # Draw petals
            num_petals = 6
            petal_length = frame_width * 0.8
            for i in range(num_petals):
                angle = 2 * np.pi * i / num_petals
                end_x = int(center_x + petal_length * math.cos(angle))
                end_y = int(center_y + petal_length * math.sin(angle))

                # Create control points for curved petal
                ctrl1_x = int(center_x + petal_length * 0.5 * math.cos(angle - 0.3))
                ctrl1_y = int(center_y + petal_length * 0.5 * math.sin(angle - 0.3))
                ctrl2_x = int(center_x + petal_length * 0.8 * math.cos(angle + 0.3))
                ctrl2_y = int(center_y + petal_length * 0.8 * math.sin(angle + 0.3))

                # Draw curved petal
                points = np.array([[center_x, center_y],
                                 [ctrl1_x, ctrl1_y],
                                 [ctrl2_x, ctrl2_y],
                                 [end_x, end_y]], dtype=np.int32)
                cv2.fillPoly(frame_img, [points], (70, 100, 140))  # Muted blue-green

        # Add flowing vine patterns along the frame edges
        # Top edge
        for x in range(frame_width * 2, width - frame_width * 2, frame_width):
            # Draw vine
            points = []
            for t in np.linspace(0, 1, 20):
                # Sinusoidal vine
                curve_x = x + frame_width * math.sin(t * 4 * np.pi)
                curve_y = frame_width // 2 + frame_width // 3 * math.sin(t * 6 * np.pi)
                points.append((int(curve_x), int(curve_y)))

            # Convert to numpy array
            points = np.array(points, dtype=np.int32)

            # Draw vine
            cv2.polylines(frame_img, [points], False, (36, 75, 48), 2)  # Dark green

            # Add leaves/flowers
            for i in range(3, len(points) - 3, 5):
                if random.random() < 0.7:  # 70% chance of a decoration
                    x, y = points[i]
                    if random.random() < 0.5:  # Leaf
                        # Create leaf shape
                        leaf_size = random.randint(5, 10)
                        leaf_points = []
                        for angle in np.linspace(0, 2 * np.pi, 8):
                            lx = x + leaf_size * math.cos(angle)
                            ly = y + leaf_size * math.sin(angle) * 0.6  # Elliptical shape
                            leaf_points.append((int(lx), int(ly)))

                        # Convert to numpy array
                        leaf_points = np.array(leaf_points, dtype=np.int32)

                        # Draw leaf
                        cv2.fillPoly(frame_img, [leaf_points], (36, 100, 70))  # Olive green
                    else:  # Flower
                        cv2.circle(frame_img, (x, y), 4, (100, 180, 210), -1)  # Gold center
                        # Draw petals
                        for angle in range(0, 360, 60):
                            petal_x = int(x + 6 * math.cos(math.radians(angle)))
                            petal_y = int(y + 6 * math.sin(math.radians(angle)))
                            cv2.circle(frame_img, (petal_x, petal_y), 3, (70, 100, 140), -1)

        # Bottom edge (mirror of top)
        for x in range(frame_width * 2, width - frame_width * 2, frame_width):
            points = []
            for t in np.linspace(0, 1, 20):
                curve_x = x + frame_width * math.sin(t * 4 * np.pi)
                curve_y = height - frame_width // 2 - frame_width // 3 * math.sin(t * 6 * np.pi)
                points.append((int(curve_x), int(curve_y)))

            points = np.array(points, dtype=np.int32)
            cv2.polylines(frame_img, [points], False, (36, 75, 48), 2)

            for i in range(3, len(points) - 3, 5):
                if random.random() < 0.7:
                    x, y = points[i]
                    if random.random() < 0.5:
                        leaf_size = random.randint(5, 10)
                        leaf_points = []
                        for angle in np.linspace(0, 2 * np.pi, 8):
                            lx = x + leaf_size * math.cos(angle)
                            ly = y + leaf_size * math.sin(angle) * 0.6
                            leaf_points.append((int(lx), int(ly)))

                        leaf_points = np.array(leaf_points, dtype=np.int32)
                        cv2.fillPoly(frame_img, [leaf_points], (36, 100, 70))
                    else:
                        cv2.circle(frame_img, (x, y), 4, (100, 180, 210), -1)
                        for angle in range(0, 360, 60):
                            petal_x = int(x + 6 * math.cos(math.radians(angle)))
                            petal_y = int(y + 6 * math.sin(math.radians(angle)))
                            cv2.circle(frame_img, (petal_x, petal_y), 3, (70, 100, 140), -1)

        # Left edge
        for y in range(frame_width * 2, height - frame_width * 2, frame_width):
            points = []
            for t in np.linspace(0, 1, 20):
                curve_x = frame_width // 2 + frame_width // 3 * math.sin(t * 6 * np.pi)
                curve_y = y + frame_width * math.sin(t * 4 * np.pi)
                points.append((int(curve_x), int(curve_y)))

            points = np.array(points, dtype=np.int32)
            cv2.polylines(frame_img, [points], False, (36, 75, 48), 2)

            for i in range(3, len(points) - 3, 5):
                if random.random() < 0.7:
                    x, y = points[i]
                    if random.random() < 0.5:
                        leaf_size = random.randint(5, 10)
                        leaf_points = []
                        for angle in np.linspace(0, 2 * np.pi, 8):
                            lx = x + leaf_size * math.cos(angle) * 0.6
                            ly = y + leaf_size * math.sin(angle)
                            leaf_points.append((int(lx), int(ly)))

                        leaf_points = np.array(leaf_points, dtype=np.int32)
                        cv2.fillPoly(frame_img, [leaf_points], (36, 100, 70))
                    else:
                        cv2.circle(frame_img, (x, y), 4, (100, 180, 210), -1)
                        for angle in range(0, 360, 60):
                            petal_x = int(x + 6 * math.cos(math.radians(angle)))
                            petal_y = int(y + 6 * math.sin(math.radians(angle)))
                            cv2.circle(frame_img, (petal_x, petal_y), 3, (70, 100, 140), -1)

        # Right edge (mirror of left)
        for y in range(frame_width * 2, height - frame_width * 2, frame_width):
            points = []
            for t in np.linspace(0, 1, 20):
                curve_x = width - frame_width // 2 - frame_width // 3 * math.sin(t * 6 * np.pi)
                curve_y = y + frame_width * math.sin(t * 4 * np.pi)
                points.append((int(curve_x), int(curve_y)))

            points = np.array(points, dtype=np.int32)
            cv2.polylines(frame_img, [points], False, (36, 75, 48), 2)

            for i in range(3, len(points) - 3, 5):
                if random.random() < 0.7:
                    x, y = points[i]
                    if random.random() < 0.5:
                        leaf_size = random.randint(5, 10)
                        leaf_points = []
                        for angle in np.linspace(0, 2 * np.pi, 8):
                            lx = x + leaf_size * math.cos(angle) * 0.6
                            ly = y + leaf_size * math.sin(angle)
                            leaf_points.append((int(lx), int(ly)))

                        leaf_points = np.array(leaf_points, dtype=np.int32)
                        cv2.fillPoly(frame_img, [leaf_points], (36, 100, 70))
                    else:
                        cv2.circle(frame_img, (x, y), 4, (100, 180, 210), -1)
                        for angle in range(0, 360, 60):
                            petal_x = int(x + 6 * math.cos(math.radians(angle)))
                            petal_y = int(y + 6 * math.sin(math.radians(angle)))
                            cv2.circle(frame_img, (petal_x, petal_y), 3, (70, 100, 140), -1)

        # Apply frame to result
        frame_mask_3d = np.stack([frame_mask] * 3, axis=2) > 0
        result[frame_mask_3d] = frame_img[frame_mask_3d]

        return result
@register_effect("concert_visuals")
class ConcertVisualsEffect(Effect):
    """High-performance concert-style visual effects"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0
        # Pre-generate lookup tables for better performance
        self.sin_lut = [math.sin(i/100.0) for i in range(628)]  # 0 to 2π
        self.cos_lut = [math.cos(i/100.0) for i in range(628)]  # 0 to 2π
        # Pre-allocate arrays
        self.temp_frame = None
        self.last_frame = None
        # Color palettes (bright, high-contrast concert colors)
        self.palettes = [
            [(255, 0, 0), (0, 0, 255)],      # Red-Blue
            [(0, 255, 0), (255, 0, 255)],    # Green-Magenta
            [(255, 255, 0), (0, 255, 255)],  # Yellow-Cyan
            [(255, 0, 255), (255, 255, 0)],  # Magenta-Yellow
            [(0, 0, 255), (255, 165, 0)]     # Blue-Orange
        ]
        self.current_palette = 0
        self.palette_change_time = 0

    def sin(self, x):
        # Fast sine lookup (much faster than math.sin)
        idx = int((x % (2 * math.pi)) * 100) % 628
        return self.sin_lut[idx]

    def cos(self, x):
        # Fast cosine lookup
        idx = int((x % (2 * math.pi)) * 100) % 628
        return self.cos_lut[idx]

    def process(self, frame):
        # Initialize buffers if needed
        height, width = frame.shape[:2]
        if self.temp_frame is None or self.temp_frame.shape[:2] != (height, width):
            self.temp_frame = np.zeros_like(frame)
            self.last_frame = np.zeros_like(frame)

        # Get audio data if available
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Change palette on beat or every few seconds
        if beat or (self.time - self.palette_change_time > 30):
            self.current_palette = (self.current_palette + 1) % len(self.palettes)
            self.palette_change_time = self.time

        # Get current color palette
        color1, color2 = self.palettes[self.current_palette]

        # Clear temp frame (faster than creating new array)
        self.temp_frame.fill(0)

        # Calculate time-based movement
        t = self.time * 0.1

        # Draw laser beams (optimize by using vectorized operations)
        num_lasers = 5 + int(volume * 10)
        laser_colors = np.array([color1, color2])

        # Create laser positions (more efficient to pre-calculate)
        laser_angles = [(i / num_lasers * math.pi * 2 + t) for i in range(num_lasers)]
        laser_origins = [(
            int(width/2 + width/3 * self.sin(angle)),
            int(height/2 + height/3 * self.cos(angle))
        ) for angle in laser_angles]

        # Draw lasers (using efficient line drawing)
        for i, (x, y) in enumerate(laser_origins):
            # Target is center with some movement
            target_x = width//2 + int(width/8 * self.sin(t * 1.5))
            target_y = height//2 + int(height/8 * self.cos(t * 1.7))

            # Select color
            color = laser_colors[i % 2]

            # Draw line
            cv2.line(self.temp_frame, (x, y), (target_x, target_y), color,
                    thickness=1 + int(volume * 3))

        # Create circular audio visualizer
        if self.audio_manager and self.audio_manager.audio_data["spectrum"] is not None:
            spectrum = self.audio_manager.audio_data["spectrum"][:64]  # Use fewer bands for performance
            if len(spectrum) > 0:
                # Normalize
                spectrum = spectrum / (np.max(spectrum) if np.max(spectrum) > 0 else 1)

                # Draw visualizer (optimized to skip pixels)
                center_x, center_y = width // 2, height // 2
                radius = min(width, height) // 4
                step = 4  # Skip pixels for speed

                for i, mag in enumerate(spectrum[::step]):
                    # Calculate angle
                    angle = i * step * 2 * math.pi / len(spectrum)

                    # Calculate line end points
                    r1 = radius
                    r2 = radius + int(radius * mag * 0.8)

                    x1 = center_x + int(r1 * self.cos(angle))
                    y1 = center_y + int(r1 * self.sin(angle))
                    x2 = center_x + int(r2 * self.cos(angle))
                    y2 = center_y + int(r2 * self.sin(angle))

                    # Alternate colors
                    color = laser_colors[i % 2]

                    # Draw line
                    cv2.line(self.temp_frame, (x1, y1), (x2, y2), color, 2)

        # Add starburst on beat
        if beat:
            cv2.circle(self.temp_frame, (width//2, height//2),
                     int(min(width, height) * 0.4), (255, 255, 255), -1)

        # Add motion blur (optimized)
        alpha = 0.7
        cv2.addWeighted(self.temp_frame, 1.0, self.last_frame, alpha, 0, self.temp_frame)
        self.last_frame = self.temp_frame.copy()

        # Overlay onto original frame
        # Use vectorized operations instead of pixel-by-pixel processing
        # Subtract from 255 for "screen" blend mode effect
        inverted_src = 255 - frame
        inverted_overlay = 255 - self.temp_frame
        result = 255 - cv2.multiply(inverted_src, inverted_overlay, scale=1/255)

        self.time += 1
        return result

@register_effect("laser_storm")
class LaserStormEffect(Effect):
    """High-performance laser light show effect"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0
        # Pre-compute trig tables
        self.sin_lut = [math.sin(i/100.0) for i in range(628)]
        self.cos_lut = [math.cos(i/100.0) for i in range(628)]
        # Laser beams
        self.lasers = []
        self.max_lasers = 20

    def sin(self, x):
        idx = int((x % (2 * math.pi)) * 100) % 628
        return self.sin_lut[idx]

    def cos(self, x):
        idx = int((x % (2 * math.pi)) * 100) % 628
        return self.cos_lut[idx]

    def process(self, frame):
        height, width = frame.shape[:2]

        # Get audio data if available
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Create black overlay
        result = np.zeros_like(frame)

        # Update time
        t = self.time * 0.1

        # Add lasers on beat or randomly
        if beat or (random.random() < 0.05):
            # Random laser properties
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.02, 0.1)
            color = (
                random.randint(150, 255),
                random.randint(150, 255),
                random.randint(150, 255)
            )

            # Add laser
            self.lasers.append({
                'angle': angle,
                'speed': speed,
                'width': 1 + int(volume * 3),
                'color': color,
                'age': 0,
                'lifetime': random.randint(20, 50)
            })

        # Update and draw lasers
        new_lasers = []
        for laser in self.lasers:
            # Update age
            laser['age'] += 1

            # Keep if still alive
            if laser['age'] < laser['lifetime']:
                # Update angle slightly for movement
                laser['angle'] += laser['speed']

                # Calculate start and end points
                cx, cy = width // 2, height // 2
                length = max(width, height)

                # Calculate direction vector
                dx = self.cos(laser['angle'])
                dy = self.sin(laser['angle'])

                # Start at center
                x1, y1 = cx, cy

                # End at edge of screen
                x2 = int(cx + dx * length)
                y2 = int(cy + dy * length)

                # Draw laser beam
                cv2.line(result, (x1, y1), (x2, y2), laser['color'],
                        thickness=laser['width'])

                # Keep laser
                new_lasers.append(laser)

        # Replace lasers list
        self.lasers = new_lasers[:self.max_lasers]  # Limit for performance

        # Add circular glow at center
        glow_radius = int(min(width, height) * 0.1 * (0.8 + 0.2 * self.sin(t)))
        glow_color = (0, 0, 255) if self.time % 60 < 30 else (255, 0, 0)  # Alternate blue/red
        cv2.circle(result, (width//2, height//2), glow_radius, glow_color, -1)

        # Add glow effect (optimized with smaller kernel)
        glow = cv2.GaussianBlur(result, (15, 15), 0)
        result = cv2.addWeighted(result, 0.7, glow, 0.5, 0)

        # Add flash on beat
        if beat:
            flash = np.ones_like(frame) * 255
            alpha = volume * 0.5
            result = cv2.addWeighted(result, 1-alpha, flash, alpha, 0)

        # Blend with original (optimized screen blend)
        inverted_src = 255 - frame
        inverted_overlay = 255 - result
        final = 255 - cv2.multiply(inverted_src, inverted_overlay, scale=1/255)

        self.time += 1
        return final

@register_effect("liquid_color")
class LiquidColorEffect(Effect):
    """Fluid, flowing color distortions that respond to audio levels"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0
        # Pre-compute trig tables
        self.sin_lut = [math.sin(i/100.0) for i in range(628)]
        self.cos_lut = [math.cos(i/100.0) for i in range(628)]
        # Pre-allocate buffers
        self.flow_x = None
        self.flow_y = None

    def sin(self, x):
        idx = int((x % (2 * math.pi)) * 100) % 628
        return self.sin_lut[idx]

    def cos(self, x):
        idx = int((x % (2 * math.pi)) * 100) % 628
        return self.cos_lut[idx]

    def process(self, frame):
        height, width = frame.shape[:2]

        # Initialize flow field buffers if needed
        if self.flow_x is None or self.flow_x.shape != (height, width):
            self.flow_x = np.zeros((height, width), np.float32)
            self.flow_y = np.zeros((height, width), np.float32)

        # Get audio data
        volume = 0.5
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)

        # Time variables
        t = self.time * 0.05

        # Update flow field (optimized with step size)
        step = 8  # Process every 8th pixel for performance
        for y in range(0, height, step):
            for x in range(0, width, step):
                # Create flowing pattern based on noise
                angle = self.sin(x/20 + t) * self.cos(y/20 - t*0.7) * math.pi

                # Increase flow strength with volume
                strength = 10 + volume * 20

                # Calculate flow vectors
                fx = strength * self.cos(angle)
                fy = strength * self.sin(angle)

                # Fill 8x8 blocks for performance
                for dy in range(step):
                    for dx in range(step):
                        if y+dy < height and x+dx < width:
                            self.flow_x[y+dy, x+dx] = fx
                            self.flow_y[y+dy, x+dx] = fy

        # Create distortion maps
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)

        # Fill coordinate maps
        for y in range(height):
            for x in range(width):
                map_x[y, x] = x + self.flow_x[y, x]
                map_y[y, x] = y + self.flow_y[y, x]

        # Apply distortion
        distorted = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

        # Enhance colors based on volume
        hsv = cv2.cvtColor(distorted, cv2.COLOR_BGR2HSV)

        # Shift hue over time
        hsv[:, :, 0] = (hsv[:, :, 0] + int(t * 5)) % 180

        # Boost saturation with volume
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + volume), 0, 255).astype(np.uint8)

        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        self.time += 1
        return result

@register_effect("fractal_zoom")
class FractalZoomEffect(Effect):
    """Infinitely detailed patterns that slowly evolve and pulse with the music (stable version)"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0
        self.zoom = 1.0
        # Pre-generate color palette for performance
        self.palette = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            hue = i % 180
            self.palette[i, 0] = [hue, 255, 255]
        self.palette = cv2.cvtColor(self.palette, cv2.COLOR_HSV2BGR)

    def process(self, frame):
        height, width = frame.shape[:2]

        # Get audio data
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Adjust zoom with beat
        if beat:
            self.zoom *= 0.9  # Zoom in on beat
        else:
            self.zoom *= 1.01  # Slowly zoom out

        # Keep zoom in reasonable range
        self.zoom = max(0.1, min(2.0, self.zoom))

        # Create fractal image (simplified version for stability)
        result = np.zeros((height, width, 3), dtype=np.uint8)

        # Calculate center coordinates
        center_x = width / 2
        center_y = height / 2

        # Julia set parameters
        t = self.time * 0.01
        c_real = 0.7885 * math.cos(t)
        c_imag = 0.7885 * math.sin(t)

        # Maximum iterations
        max_iter = 30

        # Step size for better performance
        step = 3

        # Process each pixel (with steps for performance)
        for y in range(0, height, step):
            for x in range(0, width, step):
                # Map pixel to complex plane
                scale = 1.5 * self.zoom
                z_real = (x - center_x) / (width / 2) * scale
                z_imag = (y - center_y) / (height / 2) * scale

                # Julia set iteration
                i = 0
                while i < max_iter and (z_real*z_real + z_imag*z_imag) < 4.0:
                    # z = z^2 + c
                    temp = z_real*z_real - z_imag*z_imag + c_real
                    z_imag = 2 * z_real * z_imag + c_imag
                    z_real = temp
                    i += 1

                # Map iteration count to color
                color_idx = (i * 8 + self.time) % 256
                color = self.palette[color_idx, 0].tolist()

                # Fill step×step block
                for dy in range(step):
                    for dx in range(step):
                        if y+dy < height and x+dx < width:
                            result[y+dy, x+dx] = color

        # Blend with original image
        alpha = 0.7
        result = cv2.addWeighted(frame, 1-alpha, result, alpha, 0)

        self.time += 1
        return result

@register_effect("audio_particles")
class AudioParticlesEffect(Effect):
    """Colorful particles that burst and flow based on sound intensity"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.particles = []
        self.time = 0

    def process(self, frame):
        height, width = frame.shape[:2]

        # Get audio data
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Create black background
        result = np.zeros_like(frame)

        # Create new particles on beat or based on volume
        if beat or random.random() < volume * 0.3:
            # Burst of particles
            num_new = int(20 + volume * 30)
            for _ in range(num_new):
                # Random position (near center)
                x = width // 2 + random.randint(-width//4, width//4)
                y = height // 2 + random.randint(-height//4, height//4)

                # Random velocity (outward from center)
                angle = random.uniform(0, math.pi * 2)
                speed = random.uniform(2, 6 + volume * 4)
                vx = speed * math.cos(angle)
                vy = speed * math.sin(angle)

                # Random color (bright colors)
                hue = random.randint(0, 180)
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()

                # Random size and lifetime
                size = random.randint(2, 4 + int(volume * 3))
                lifetime = random.randint(20, 50)

                self.particles.append({
                    'x': x, 'y': y,
                    'vx': vx, 'vy': vy,
                    'color': color,
                    'size': size,
                    'lifetime': lifetime,
                    'age': 0
                })

        # Maximum particles (limit for performance)
        max_particles = 300

        # Update and draw particles
        updated_particles = []

        for particle in self.particles:
            # Update age
            particle['age'] += 1

            # Skip if too old
            if particle['age'] > particle['lifetime']:
                continue

            # Update position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']

            # Apply gravity/attraction toward center
            cx, cy = width // 2, height // 2
            dx = cx - particle['x']
            dy = cy - particle['y']
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > 0:
                # Strength of gravity decreases with distance
                strength = 0.05 * (1 + volume)
                particle['vx'] += dx / dist * strength
                particle['vy'] += dy / dist * strength

            # Apply velocity damping
            particle['vx'] *= 0.98
            particle['vy'] *= 0.98

            # Draw particle if within bounds
            x, y = int(particle['x']), int(particle['y'])
            if 0 <= x < width and 0 <= y < height:
                # Get fade based on age
                fade = 1.0 - particle['age'] / particle['lifetime']

                # Apply color with fade
                color = [int(c * fade) for c in particle['color']]
                size = max(1, int(particle['size'] * fade))

                # Draw circle
                cv2.circle(result, (x, y), size, color, -1)

            # Keep particle
            updated_particles.append(particle)

            # Limit number of particles for performance
            if len(updated_particles) >= max_particles:
                break

        # Update particles list
        self.particles = updated_particles

        # Add glow effect (optimized with smaller kernel)
        glow_size = 5 + int(volume * 10)
        if glow_size % 2 == 0:
            glow_size += 1  # Ensure odd kernel size
        glow = cv2.GaussianBlur(result, (glow_size, glow_size), 0)
        result = cv2.addWeighted(result, 0.7, glow, 0.5, 0)

        # Blend with original image
        inverted_src = 255 - frame
        inverted_overlay = 255 - result
        final = 255 - cv2.multiply(inverted_src, inverted_overlay, scale=1/255)

        self.time += 1
        return final

@register_effect("wavefront")
class WavefrontEffect(Effect):
    """Ripple effects that emanate from performers or beat drops"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.waves = []
        self.time = 0

    def process(self, frame):
        height, width = frame.shape[:2]

        # Get audio data
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Create distortion maps
        map_x = np.zeros((height, width), np.float32)
        map_y = np.zeros((height, width), np.float32)

        # Generate identity maps
        for y in range(height):
            map_x[y, :] = np.arange(width)
        for x in range(width):
            map_y[:, x] = np.arange(height)

        # Create new wave on beat
        if beat:
            self.waves.append({
                'x': width // 2,
                'y': height // 2,
                'radius': 10,
                'max_radius': max(width, height),
                'amplitude': 5 + volume * 15,
                'speed': 5 + volume * 5,
                'thickness': 3 + volume * 5
            })

        # Update and apply waves
        new_waves = []

        for wave in self.waves:
            # Update radius
            wave['radius'] += wave['speed']

            # Keep if still within bounds
            if wave['radius'] < wave['max_radius']:
                # Calculate wave effect
                for y in range(0, height, 2):  # Step by 2 for performance
                    for x in range(0, width, 2):
                        # Distance from wave center
                        dx = x - wave['x']
                        dy = y - wave['y']
                        distance = math.sqrt(dx*dx + dy*dy)

                        # Check if within wave band
                        wave_min = wave['radius'] - wave['thickness']
                        wave_max = wave['radius'] + wave['thickness']

                        if wave_min <= distance <= wave_max:
                            # Calculate displacement
                            factor = 1 - (distance - wave_min) / (wave_max - wave_min)
                            factor = factor * factor  # Square for smoother falloff

                            # Amplitude decreases with radius
                            amplitude = wave['amplitude'] * (1 - wave['radius'] / wave['max_radius'])

                            # Direction vectors
                            if distance > 0:
                                dir_x = dx / distance
                                dir_y = dy / distance
                            else:
                                dir_x, dir_y = 0, 0

                            # Apply displacement to 2x2 block
                            for dy in range(2):
                                for dx in range(2):
                                    if y+dy < height and x+dx < width:
                                        # Add displacement to maps
                                        map_x[y+dy, x+dx] += dir_x * amplitude * factor
                                        map_y[y+dy, x+dx] += dir_y * amplitude * factor

                # Keep wave
                new_waves.append(wave)

        # Update waves list
        self.waves = new_waves

        # Apply distortion
        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Add colorful wave visualization
        for wave in self.waves:
            # Draw circle for each wave
            radius = int(wave['radius'])
            color_hue = (self.time * 2) % 180
            color = cv2.cvtColor(np.uint8([[[color_hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            thickness = max(1, int(wave['thickness'] * 0.5))
            cv2.circle(result, (wave['x'], wave['y']), radius, color, thickness)

        self.time += 1
        return result

@register_effect("neon_tracers")
class NeonTracersEffect(Effect):
    """Leaving colorful trails behind moving objects (fixed version)"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.prev_frames = []
        self.max_history = 15
        self.time = 0

    def process(self, frame):
        # Add frame to history
        self.prev_frames.append(frame.copy())
        if len(self.prev_frames) > self.max_history:
            self.prev_frames.pop(0)

        # Need at least 2 frames
        if len(self.prev_frames) < 2:
            return frame

        # Get audio data
        volume = 0.5
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)

        # Create motion detection
        prev = cv2.cvtColor(self.prev_frames[-2], cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(prev, curr)

        # Threshold to get significant motion
        threshold = 15 + int(volume * 10)
        _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Dilate to connect nearby motion
        kernel_size = 3 + int(volume * 4)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

        # Create color trails
        result = frame.copy()

        # Draw trails from previous frames
        for i, past_frame in enumerate(self.prev_frames):
            # Skip current frame
            if i == len(self.prev_frames) - 1:
                continue

            # Calculate weight based on recency
            weight = (i + 1) / len(self.prev_frames)

            # Color shift based on position in history
            hsv = cv2.cvtColor(past_frame, cv2.COLOR_BGR2HSV)

            # Shift hue based on time and position
            shift = (self.time + i * 10) % 180
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180

            # Increase saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)

            # Convert back to BGR
            colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Get motion mask for this frame
            if i > 0:
                prev_frame = cv2.cvtColor(self.prev_frames[i-1], cv2.COLOR_BGR2GRAY)
                curr_frame = cv2.cvtColor(past_frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(prev_frame, curr_frame)
                _, frame_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
                frame_mask = cv2.dilate(frame_mask, kernel, iterations=1)
            else:
                frame_mask = motion_mask

            # Apply mask properly (avoid boolean indexing error)
            # Convert mask to 3 channel
            frame_mask_3d = cv2.cvtColor(frame_mask, cv2.COLOR_GRAY2BGR)
            # Use weighted blending with mask
            alpha = weight * 0.5
            mask_normalized = frame_mask_3d / 255.0
            inv_mask = 1.0 - mask_normalized

            # Blend using the formula: result = img1 * (1-alpha*mask) + img2 * (alpha*mask)
            result = (result * inv_mask + colored * mask_normalized * alpha).astype(np.uint8)

        # Create glow by blurring motion areas
        motion_mask_3d = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        glow_img = (result * (motion_mask_3d / 255.0)).astype(np.uint8)
        glow = cv2.GaussianBlur(glow_img, (15, 15), 0)

        # Add glow (use direct calculation rather than boolean indexing)
        result = cv2.addWeighted(result, 1.0, glow, 0.5, 0)

        self.time += 1
        return result
    """Leaving colorful trails behind moving objects"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.prev_frames = []
        self.max_history = 15
        self.time = 0

    def process(self, frame):
        # Add frame to history
        self.prev_frames.append(frame.copy())
        if len(self.prev_frames) > self.max_history:
            self.prev_frames.pop(0)

        # Need at least 2 frames
        if len(self.prev_frames) < 2:
            return frame

        # Get audio data
        volume = 0.5
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)

        # Create motion detection
        prev = cv2.cvtColor(self.prev_frames[-2], cv2.COLOR_BGR2GRAY)
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff = cv2.absdiff(prev, curr)

        # Threshold to get significant motion
        threshold = 15 + int(volume * 10)
        _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        # Dilate to connect nearby motion
        kernel_size = 3 + int(volume * 4)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

        # Create color trails
        result = frame.copy()

        # Draw trails from previous frames
        for i, past_frame in enumerate(self.prev_frames):
            # Skip current frame
            if i == len(self.prev_frames) - 1:
                continue

            # Calculate weight based on recency
            weight = (i + 1) / len(self.prev_frames)

            # Color shift based on position in history
            hsv = cv2.cvtColor(past_frame, cv2.COLOR_BGR2HSV)

            # Shift hue based on time and position
            shift = (self.time + i * 10) % 180
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180

            # Increase saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)

            # Convert back to BGR
            colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # Get motion mask for this frame
            if i > 0:
                prev_frame = cv2.cvtColor(self.prev_frames[i-1], cv2.COLOR_BGR2GRAY)
                curr_frame = cv2.cvtColor(past_frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(prev_frame, curr_frame)
                _, frame_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
                frame_mask = cv2.dilate(frame_mask, kernel, iterations=1)
            else:
                frame_mask = motion_mask

            # Apply mask
            frame_mask_3d = np.stack([frame_mask] * 3, axis=2) > 0

            # Blend with result (only in motion areas)
            alpha = weight * 0.5
            result[frame_mask_3d] = cv2.addWeighted(
                result[frame_mask_3d], 1-alpha,
                colored[frame_mask_3d], alpha, 0
            )

        # Add final neon glow effect
        motion_mask_3d = np.stack([motion_mask] * 3, axis=2) > 0

        # Create glow by blurring motion areas
        glow_img = np.zeros_like(result)
        glow_img[motion_mask_3d] = result[motion_mask_3d]
        glow = cv2.GaussianBlur(glow_img, (15, 15), 0)

        # Add glow
        result = cv2.addWeighted(result, 1.0, glow, 0.5, 0)

        self.time += 1
        return result

@register_effect("geometric_overlay")
class GeometricOverlayEffect(Effect):
    """Sacred geometry patterns that float and transform with the music (fixed version)"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0
        # Pre-compute trig tables
        self.sin_lut = [math.sin(i/100.0) for i in range(628)]
        self.cos_lut = [math.cos(i/100.0) for i in range(628)]
        # Initialize patterns
        self.patterns = []
        for _ in range(3):  # Multiple overlapping patterns
            self.patterns.append({
                'type': random.choice(['circle', 'hexagon', 'pentagon', 'star']),
                'scale': random.uniform(0.5, 1.5),
                'rotation': random.uniform(0, math.pi * 2),
                'speed': random.uniform(0.01, 0.03),
                'center_x': random.uniform(0.3, 0.7),
                'center_y': random.uniform(0.3, 0.7),
                'color_shift': random.uniform(0, 1)
            })

    def sin(self, x):
        idx = int((x % (2 * math.pi)) * 100) % 628
        return self.sin_lut[idx]

    def cos(self, x):
        idx = int((x % (2 * math.pi)) * 100) % 628
        return self.cos_lut[idx]

    def draw_pattern(self, canvas, pattern_type, center_x, center_y, scale, rotation, color):
        height, width = canvas.shape[:2]
        cx, cy = int(width * center_x), int(height * center_y)
        size = int(min(width, height) * 0.3 * scale)

        if pattern_type == 'circle':
            # Draw circle with inner patterns
            cv2.circle(canvas, (cx, cy), size, color, 2)
            cv2.circle(canvas, (cx, cy), int(size * 0.7), color, 1)
            cv2.circle(canvas, (cx, cy), int(size * 0.4), color, 1)

            # Add cross lines
            num_lines = 6
            for i in range(num_lines):
                angle = rotation + i * math.pi / (num_lines // 2)
                x1 = int(cx + size * self.cos(angle))
                y1 = int(cy + size * self.sin(angle))
                x2 = int(cx - size * self.cos(angle))
                y2 = int(cy - size * self.sin(angle))
                cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

        elif pattern_type == 'hexagon':
            # Draw hexagon
            points = []
            for i in range(6):
                angle = rotation + i * math.pi / 3
                x = int(cx + size * self.cos(angle))
                y = int(cy + size * self.sin(angle))
                points.append((x, y))

            # Convert to numpy array
            points = np.array(points, dtype=np.int32)

            # Draw hexagon
            cv2.polylines(canvas, [points], True, color, 2)

            # Draw inner hexagon
            inner_points = []
            for i in range(6):
                angle = rotation + i * math.pi / 3
                x = int(cx + size * 0.6 * self.cos(angle))
                y = int(cy + size * 0.6 * self.sin(angle))
                inner_points.append((x, y))

                # Connect to outer points
                cv2.line(canvas, points[i], inner_points[i], color, 1)

            # Convert to numpy array
            inner_points = np.array(inner_points, dtype=np.int32)

            # Draw inner hexagon
            cv2.polylines(canvas, [inner_points], True, color, 1)

        elif pattern_type == 'pentagon':
            # Draw pentagon
            points = []
            for i in range(5):
                angle = rotation + i * 2 * math.pi / 5
                x = int(cx + size * self.cos(angle))
                y = int(cy + size * self.sin(angle))
                points.append((x, y))

            # Convert to numpy array
            points = np.array(points, dtype=np.int32)

            # Draw pentagon
            cv2.polylines(canvas, [points], True, color, 2)

            # Draw inner pentagon
            inner_points = []
            for i in range(5):
                angle = rotation + i * 2 * math.pi / 5
                x = int(cx + size * 0.6 * self.cos(angle))
                y = int(cy + size * 0.6 * self.sin(angle))
                inner_points.append((x, y))

            # Convert to numpy array
            inner_points = np.array(inner_points, dtype=np.int32)

            # Draw inner pentagon
            cv2.polylines(canvas, [inner_points], True, color, 1)

            # Draw star by connecting alternate points
            for i in range(5):
                cv2.line(canvas, points[i], points[(i+2) % 5], color, 1)

        elif pattern_type == 'star':
            # Draw star
            outer_points = []
            inner_points = []

            for i in range(5):
                # Outer point
                angle_outer = rotation + i * 2 * math.pi / 5
                x_outer = int(cx + size * self.cos(angle_outer))
                y_outer = int(cy + size * self.sin(angle_outer))
                outer_points.append((x_outer, y_outer))

                # Inner point
                angle_inner = rotation + (i + 0.5) * 2 * math.pi / 5
                x_inner = int(cx + size * 0.4 * self.cos(angle_inner))
                y_inner = int(cy + size * 0.4 * self.sin(angle_inner))
                inner_points.append((x_inner, y_inner))

            # Draw star by alternating outer and inner points
            star_points = []
            for i in range(5):
                star_points.append(outer_points[i])
                star_points.append(inner_points[i])

            # Convert to numpy array
            star_points = np.array(star_points, dtype=np.int32)

            # Draw star
            cv2.polylines(canvas, [star_points], True, color, 2)

            # Draw inner pentagon
            cv2.polylines(canvas, [np.array(inner_points, dtype=np.int32)], True, color, 1)

    def process(self, frame):
        height, width = frame.shape[:2]

        # Get audio data
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Create overlay canvas
        overlay = np.zeros_like(frame)

        # Update and draw patterns
        for pattern in self.patterns:
            # Update rotation
            pattern['rotation'] += pattern['speed'] * (1 + volume)

            # Change pattern on beat with probability
            if beat and random.random() < 0.3:
                pattern['type'] = random.choice(['circle', 'hexagon', 'pentagon', 'star'])

            # Calculate color with time-varying hue
            hue = int((self.time * 0.5 + pattern['color_shift'] * 180) % 180)
            sat = 220 + int(volume * 35)
            val = 180 + int(volume * 75)
            color_hsv = np.uint8([[[hue, sat, val]]])
            color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()

            # Draw pattern
            self.draw_pattern(
                overlay, pattern['type'],
                pattern['center_x'], pattern['center_y'],
                pattern['scale'] * (0.8 + 0.3 * volume),
                pattern['rotation'], color
            )

        # Add glow effect
        glow = cv2.GaussianBlur(overlay, (9, 9), 0)
        overlay = cv2.addWeighted(overlay, 0.7, glow, 0.5, 0)

        # Blend with original frame using addWeighted instead of boolean indexing
        alpha = 0.6 + 0.2 * volume
        result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)

        self.time += 1
        return result

    """Sacred geometry patterns that float and transform with the music"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0
        # Pre-compute trig tables
        self.sin_lut = [math.sin(i/100.0) for i in range(628)]
        self.cos_lut = [math.cos(i/100.0) for i in range(628)]
        # Initialize patterns
        self.patterns = []
        for _ in range(3):  # Multiple overlapping patterns
            self.patterns.append({
                'type': random.choice(['circle', 'hexagon', 'pentagon', 'star']),
                'scale': random.uniform(0.5, 1.5),
                'rotation': random.uniform(0, math.pi * 2),
                'speed': random.uniform(0.01, 0.03),
                'center_x': random.uniform(0.3, 0.7),
                'center_y': random.uniform(0.3, 0.7),
                'color_shift': random.uniform(0, 1)
            })

    def sin(self, x):
        idx = int((x % (2 * math.pi)) * 100) % 628
        return self.sin_lut[idx]

    def cos(self, x):
        idx = int((x % (2 * math.pi)) * 100) % 628
        return self.cos_lut[idx]

    def draw_pattern(self, canvas, pattern_type, center_x, center_y, scale, rotation, color):
        height, width = canvas.shape[:2]
        cx, cy = int(width * center_x), int(height * center_y)
        size = int(min(width, height) * 0.3 * scale)

        if pattern_type == 'circle':
            # Draw circle with inner patterns
            cv2.circle(canvas, (cx, cy), size, color, 2)
            cv2.circle(canvas, (cx, cy), int(size * 0.7), color, 1)
            cv2.circle(canvas, (cx, cy), int(size * 0.4), color, 1)

            # Add cross lines
            for i in range(6):
                angle = rotation + i * math.pi / 3
                x1 = int(cx + size * self.cos(angle))
                y1 = int(cy + size * self.sin(angle))
                x2 = int(cx - size * self.cos(angle))
                y2 = int(cy - size * self.sin(angle))
                cv2.line(canvas, (x1, y1), (x2, y2), color, 1)

        elif pattern_type == 'hexagon':
            # Draw hexagon
            points = []
            for i in range(6):
                angle = rotation + i * math.pi / 3
                x = int(cx + size * self.cos(angle))
                y = int(cy + size * self.sin(angle))
                points.append((x, y))

            # Convert to numpy array
            points = np.array(points, dtype=np.int32)

            # Draw hexagon
            cv2.polylines(canvas, [points], True, color, 2)

            # Draw inner hexagon
            inner_points = []
            for i in range(6):
                angle = rotation + i * math.pi / 3
                x = int(cx + size * 0.6 * self.cos(angle))
                y = int(cy + size * 0.6 * self.sin(angle))
                inner_points.append((x, y))

                # Connect to outer points
                cv2.line(canvas, points[i], inner_points[i], color, 1)

            # Convert to numpy array
            inner_points = np.array(inner_points, dtype=np.int32)

            # Draw inner hexagon
            cv2.polylines(canvas, [inner_points], True, color, 1)

        elif pattern_type == 'pentagon':
            # Draw pentagon
            points = []
            for i in range(5):
                angle = rotation + i * 2 * math.pi / 5
                x = int(cx + size * self.cos(angle))
                y = int(cy + size * self.sin(angle))
                points.append((x, y))

            # Convert to numpy array
            points = np.array(points, dtype=np.int32)

            # Draw pentagon
            cv2.polylines(canvas, [points], True, color, 2)

            # Draw inner pentagon
            inner_points = []
            for i in range(5):
                angle = rotation + i * 2 * math.pi / 5
                x = int(cx + size * 0.6 * self.cos(angle))
                y = int(cy + size * 0.6 * self.sin(angle))
                inner_points.append((x, y))

            # Convert to numpy array
            inner_points = np.array(inner_points, dtype=np.int32)

            # Draw inner pentagon
            cv2.polylines(canvas, [inner_points], True, color, 1)

            # Draw star by connecting alternate points
            for i in range(5):
                cv2.line(canvas, points[i], points[(i+2) % 5], color, 1)

        elif pattern_type == 'star':
            # Draw star
            outer_points = []
            inner_points = []

            for i in range(5):
                # Outer point
                angle_outer = rotation + i * 2 * math.pi / 5
                x_outer = int(cx + size * self.cos(angle_outer))
                y_outer = int(cy + size * self.sin(angle_outer))
                outer_points.append((x_outer, y_outer))

                # Inner point
                angle_inner = rotation + (i + 0.5) * 2 * math.pi / 5
                x_inner = int(cx + size * 0.4 * self.cos(angle_inner))
                y_inner = int(cy + size * 0.4 * self.sin(angle_inner))
                inner_points.append((x_inner, y_inner))

            # Draw star by alternating outer and inner points
            star_points = []
            for i in range(5):
                star_points.append(outer_points[i])
                star_points.append(inner_points[i])

            # Convert to numpy array
            star_points = np.array(star_points, dtype=np.int32)

            # Draw star
            cv2.polylines(canvas, [star_points], True, color, 2)

            # Draw inner pentagon
            cv2.polylines(canvas, [np.array(inner_points, dtype=np.int32)], True, color, 1)

    def process(self, frame):
        height, width = frame.shape[:2]

        # Get audio data
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Create overlay canvas
        overlay = np.zeros_like(frame)

        # Update and draw patterns
        for pattern in self.patterns:
            # Update rotation
            pattern['rotation'] += pattern['speed'] * (1 + volume)

            # Change pattern on beat with probability
            if beat and random.random() < 0.3:
                pattern['type'] = random.choice(['circle', 'hexagon', 'pentagon', 'star'])

            # Calculate color with time-varying hue
            hue = int((self.time * 0.5 + pattern['color_shift'] * 180) % 180)
            sat = 220 + int(volume * 35)
            val = 180 + int(volume * 75)
            color_hsv = np.uint8([[[hue, sat, val]]])
            color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()

            # Draw pattern
            self.draw_pattern(
                overlay, pattern['type'],
                pattern['center_x'], pattern['center_y'],
                pattern['scale'] * (0.8 + 0.3 * volume),
                pattern['rotation'], color
            )

        # Add glow effect
        glow = cv2.GaussianBlur(overlay, (9, 9), 0)
        overlay = cv2.addWeighted(overlay, 0.7, glow, 0.5, 0)

        # Blend with original frame
        alpha = 0.6 + 0.2 * volume
        result = cv2.addWeighted(frame, 1-alpha, overlay, alpha, 0)

        self.time += 1
        return result



@register_effect("color_cycling")
class ColorCyclingEffect(Effect):
   """Gradual shifts through vibrant color spectrums"""
   def __init__(self, audio_manager=None):
       super().__init__(audio_manager)
       self.time = 0

   def process(self, frame):
       # Get audio data
       volume = 0.5
       beat = False
       if self.audio_manager:
           volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
           beat = self.audio_manager.audio_data["beat_detected"]

       # Convert to HSV for easy color manipulation
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

       # Get dimensions
       height, width = hsv.shape[:2]

       # Calculate color shift amount
       # Normal shift plus extra on beats
       shift_amount = int(self.time * 0.5)
       if beat:
           shift_amount += 10

       # Apply different shifts to different regions for more interesting effect
       for y in range(0, height, 2):  # Process every other pixel for performance
           # Calculate y-dependent hue shift
           y_factor = y / height
           shift = (shift_amount + int(30 * y_factor)) % 180

           # Apply shift to entire row
           hsv[y:y+2, :, 0] = (hsv[y:y+2, :, 0] + shift) % 180

       # Boost saturation based on volume
       hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + volume * 0.5), 0, 255).astype(np.uint8)

       # Convert back to BGR
       result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

       self.time += 1
       return result

@register_effect("anatomical_silhouette")
class AnatomicalSilhouetteEffect(Effect):
    """Create an anatomical drawing effect with high-contrast colors and film grain"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0
        self.noise_texture = None

    def process(self, frame):
        height, width = frame.shape[:2]

        # Create noise texture if not already done
        if self.noise_texture is None or self.noise_texture.shape[:2] != (height, width):
            self.noise_texture = np.random.randint(0, 25, (height, width), dtype=np.uint8)

        # Step 1: Create high-contrast silhouette
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, silhouette = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

        # Step 2: Find edges for interior linework
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        # Step 3: Create background and foreground colors
        # Background: mustard yellow
        bg_color = np.ones_like(frame) * np.array([35, 142, 147], dtype=np.uint8)  # BGR

        # Foreground: pink with red lines
        fg_color = np.ones_like(frame) * np.array([180, 180, 240], dtype=np.uint8)  # Light pink
        line_color = np.ones_like(frame) * np.array([60, 60, 220], dtype=np.uint8)  # Red lines

        # Step 4: Combine elements
        result = bg_color.copy()

        # Apply silhouette (pink areas)
        silhouette_mask = silhouette > 0
        result[silhouette_mask] = fg_color[silhouette_mask]

        # Apply edges (red lines) within silhouette
        edge_mask = (edges > 0) & silhouette_mask
        result[edge_mask] = line_color[edge_mask]

        # Step 5: Add film grain/noise
        # Create dynamic noise (changes slightly over time)
        dynamic_noise = np.random.randint(0, 25, (height, width), dtype=np.uint8)
        noise = (self.noise_texture * 0.5 + dynamic_noise * 0.5).astype(np.uint8)

        # Apply noise to result
        for c in range(3):
            result[:, :, c] = cv2.add(result[:, :, c], noise)
            result[:, :, c] = cv2.subtract(result[:, :, c], noise // 2)

        # Step 6: Add black outline to silhouette
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(silhouette, kernel, iterations=1)
        outline = dilated - silhouette
        result[outline > 0] = [0, 0, 0]  # Black outline

        self.time += 1
        return result






@register_effect("rgb_shift_audio")
class RGBShiftAudioEffect(Effect):
    """RGB channel shifting effect that reacts to audio intensity"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0

    def process(self, frame):
        # Get audio data
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Split into RGB channels
        b, g, r = cv2.split(frame)

        # Calculate shift amount based on audio volume
        base_shift = 3  # Minimum shift
        max_shift = 15  # Maximum shift

        # Calculate current shift (more shift on louder audio)
        shift_amount = int(base_shift + volume * max_shift)

        # Add extra shift on beat
        if beat:
            shift_amount += 5

        # Create time-based shifting pattern
        t = self.time * 0.1
        shift_r_x = int(shift_amount * math.cos(t))
        shift_r_y = int(shift_amount * 0.5 * math.sin(t))
        shift_b_x = int(shift_amount * math.cos(t + math.pi))
        shift_b_y = int(shift_amount * 0.5 * math.sin(t + math.pi))

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





@register_effect("rgb_shift_extreme")
class RGBShiftExtremeEffect(Effect):
    """Extreme RGB channel shifting effect that reacts to audio intensity"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0

    def process(self, frame):
        # Get audio data
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Split into RGB channels
        b, g, r = cv2.split(frame)

        # Calculate shift amount based on audio volume
        # Make the shifts much larger for visibility
        base_shift = 10  # Increased minimum shift
        max_shift = 30  # Increased maximum shift

        # Calculate current shift (more shift on louder audio)
        shift_amount = int(base_shift + volume * max_shift)

        # Add extra shift on beat
        if beat:
            shift_amount += 15

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





@register_effect("rgb_scatter")
class RGBScatterEffect(Effect):
    """RGB channels scatter in different directions based on audio"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)
        self.time = 0

    def process(self, frame):
        # Get audio data
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Split into RGB channels
        b, g, r = cv2.split(frame)

        # Calculate base shift amount based on audio volume
        base_shift = 10
        max_shift = 40
        shift_amount = int(base_shift + volume * max_shift)

        # Add extra shift on beat
        if beat:
            shift_amount *= 1.5

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
    """RGB shift with channels moving in specific directions based on audio volume"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)

    def process(self, frame):
        # Get audio volume
        volume = 0.5
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)

        # Split into RGB channels
        b, g, r = cv2.split(frame)

        # Calculate shift amount based solely on audio volume
        max_shift = 50  # Maximum possible shift at highest volume
        shift_amount = int(volume * max_shift)

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



@register_effect("rgb_debug")
class RGBDebugEffect(Effect):
    """RGB shift with visual debug information for audio levels"""
    def __init__(self, audio_manager=None):
        super().__init__(audio_manager)

    def process(self, frame):
        height, width = frame.shape[:2]

        # Get audio volume
        volume = 0.5
        beat = False
        if self.audio_manager:
            volume = min(1.0, self.audio_manager.audio_data["volume"] / 10000)
            beat = self.audio_manager.audio_data["beat_detected"]

        # Split into RGB channels
        b, g, r = cv2.split(frame)

        # Calculate shift amount based solely on audio volume
        max_shift = 50  # Maximum possible shift at highest volume
        shift_amount = int(volume * max_shift)

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

        # Apply translations
        r_matrix = np.float32([[1, 0, shift_r_x], [0, 1, shift_r_y]])
        g_matrix = np.float32([[1, 0, shift_g_x], [0, 1, shift_g_y]])
        b_matrix = np.float32([[1, 0, shift_b_x], [0, 1, shift_b_y]])

        r_shifted = cv2.warpAffine(r, r_matrix, (width, height))
        g_shifted = cv2.warpAffine(g, g_matrix, (width, height))
        b_shifted = cv2.warpAffine(b, b_matrix, (width, height))

        # Merge channels back
        result = cv2.merge([b_shifted, g_shifted, r_shifted])

        # Add debug information
        # Draw volume meter
        meter_width = 200
        meter_height = 30
        meter_x = 20
        meter_y = height - 50

        # Background for meter
        cv2.rectangle(result, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (0, 0, 0), -1)

        # Fill based on volume
        fill_width = int(meter_width * volume)
        cv2.rectangle(result, (meter_x, meter_y), (meter_x + fill_width, meter_y + meter_height), (0, 255, 0), -1)

        # Add text
        cv2.putText(result, f"Volume: {volume:.3f}", (meter_x, meter_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(result, f"Shift: {shift_amount} px", (meter_x + meter_width + 20, meter_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Indicate beat detection
        if beat:
            cv2.putText(result, "BEAT", (width - 100, meter_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return result

