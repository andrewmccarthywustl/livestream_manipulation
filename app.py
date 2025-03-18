import cv2
import numpy as np
import argparse
import sys
import threading
import os
import subprocess
from datetime import datetime
from audio_manager import AudioManager
from effects import get_all_effects, get_effect

def list_cameras():
    """List all available camera devices with their indices."""
    index = 0
    devices = []

    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break

        # Get camera name if possible (not always available on all platforms)
        name = f"Camera {index}"

        # Get resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Store camera info
        devices.append({
            'index': index,
            'name': name,
            'resolution': f"{width}x{height}"
        })

        cap.release()
        index += 1

    return devices

def list_audio_devices():
    """List all available audio input devices with their indices."""
    import pyaudio

    p = pyaudio.PyAudio()
    info = []

    print("\nAvailable Audio Input Devices:")
    print("-" * 50)

    # Iterate through all audio devices
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)

        # Check if this is an input device (has input channels)
        if dev_info['maxInputChannels'] > 0:
            default = ""
            if dev_info.get('defaultSampleRate'):
                default = f", {int(dev_info['defaultSampleRate'])}Hz"

            print(f"Index {i}: {dev_info['name']} (Channels: {dev_info['maxInputChannels']}{default})")

            # Store device info
            info.append({
                'index': i,
                'name': dev_info['name'],
                'channels': dev_info['maxInputChannels'],
                'sample_rate': int(dev_info.get('defaultSampleRate', 0))
            })

    # Identify the default input device
    try:
        default_idx = p.get_default_input_device_info()['index']
        print(f"\nDefault Input Device: Index {default_idx}")
    except IOError:
        print("\nNo default input device available")

    p.terminate()
    return info

def test_audio_manager(input_device_index=None):
    """
    Run a standalone test for the AudioManager with a visual display
    showing the audio level and beat detection.
    """
    print("Testing AudioManager...")
    print("This will display a simple visualization of audio levels.")
    print("Press 'q' to quit the test.")

    # Create and start audio manager
    audio_manager = AudioManager(input_device_index=input_device_index)
    audio_manager.start()

    # Create a black canvas for visualization
    width, height = 800, 400
    window_name = 'Audio Manager Test'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    try:
        while True:
            # Create black canvas
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # Get current audio data
            volume = audio_manager.audio_data["volume"]
            peak_volume = audio_manager.audio_data["peak_volume"]
            beat_detected = audio_manager.audio_data["beat_detected"]
            spectrum = audio_manager.audio_data["spectrum"]

            # Normalize volumes for display
            norm_volume = min(1.0, volume / 10000)
            norm_peak = min(1.0, peak_volume / 10000)

            # Draw volume meter
            meter_height = 50
            meter_y = 50
            meter_width = width - 100

            # Background for meter
            cv2.rectangle(canvas, (50, meter_y), (50 + meter_width, meter_y + meter_height), (50, 50, 50), -1)

            # Current volume (green)
            vol_width = int(meter_width * norm_volume)
            cv2.rectangle(canvas, (50, meter_y), (50 + vol_width, meter_y + meter_height), (0, 255, 0), -1)

            # Peak volume (blue line)
            peak_x = 50 + int(meter_width * norm_peak)
            cv2.line(canvas, (peak_x, meter_y), (peak_x, meter_y + meter_height), (255, 0, 0), 2)

            # Draw beat indicator
            if beat_detected:
                # Flash the entire screen briefly when a beat is detected
                canvas[:] = (0, 0, 100)  # Subtle blue flash
                # Add big "BEAT" text
                cv2.putText(canvas, "BEAT", (width//2 - 100, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)

            # Draw spectrum if available
            if spectrum is not None and len(spectrum) > 0:
                # Normalize spectrum
                norm_spectrum = spectrum / (np.max(spectrum) if np.max(spectrum) > 0 else 1)

                # Display spectrum as bars
                num_bars = min(64, len(norm_spectrum))
                bar_width = (width - 100) // num_bars
                bar_y = 200

                for i in range(num_bars):
                    bar_height = int(150 * norm_spectrum[i])
                    bar_x = 50 + i * bar_width

                    # Calculate color based on frequency (rainbow effect)
                    hue = int(i * 180 / num_bars)
                    color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()

                    cv2.rectangle(canvas, (bar_x, bar_y),
                                 (bar_x + bar_width - 2, bar_y - bar_height),
                                 color, -1)

            # Add text info
            cv2.putText(canvas, f"Volume: {norm_volume:.3f}", (50, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(canvas, f"Peak: {norm_peak:.3f}", (300, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(canvas, "Press 'q' to quit", (width - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Display the visualization
            cv2.imshow(window_name, canvas)

            # Check for key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Clean up
        audio_manager.stop()
        cv2.destroyAllWindows()
        print("Audio test complete.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Webcam effects application")
    parser.add_argument("-e", "--effects", type=str, default="original",
                      help="Effects to apply, comma-separated (e.g., 'grayscale,neon,glitch')")
    parser.add_argument("-a", "--audio", action="store_true",
                      help="Enable audio-responsive effects")
    parser.add_argument("-l", "--list", action="store_true",
                      help="List all available effects")
    parser.add_argument("-r", "--record", action="store_true",
                      help="Record the output window using macOS screen recording")
    parser.add_argument("-o", "--output", type=str, default="",
                      help="Output filename for recording (default: 'output_YYYYMMDD_HHMMSS.mp4')")
    parser.add_argument("-c", "--camera", type=int, default=0,
                      help="Camera index to use (default: 0)")
    parser.add_argument("--list-cameras", action="store_true",
                      help="List all available camera devices")
    parser.add_argument("--list-audio", action="store_true",
                      help="List all available audio input devices")
    parser.add_argument("--audio-device", type=int,
                      help="Index of audio input device to use")
    parser.add_argument("--test-audio", action="store_true",
                      help="Run a test for the AudioManager with visual feedback")
    args = parser.parse_args()

    # List cameras if requested
    if args.list_cameras:
        devices = list_cameras()
        if devices:
            print("Available camera devices:")
            for device in devices:
                print(f"  {device['index']}: {device['name']} ({device['resolution']})")
        else:
            print("No camera devices found.")
        sys.exit(0)

    # List audio devices if requested
    if args.list_audio:
        list_audio_devices()
        sys.exit(0)

    # Test audio if requested
    if args.test_audio:
        test_audio_manager(input_device_index=args.audio_device)
        sys.exit(0)

    # List effects if requested
    if args.list:
        print("Available effects:")
        effects = get_all_effects()
        for name, effect_class in sorted(effects.items()):
            description = effect_class.__doc__ or "No description"
            print(f"  - {name}: {description}")
        sys.exit(0)

    # Initialize audio manager if needed
    audio_manager = None
    if args.audio:
        audio_manager = AudioManager(input_device_index=args.audio_device)
        audio_manager.start()

    # Parse effect names
    effect_names = [name.strip().lower() for name in args.effects.split(',')]

    # Get effect instances
    effects = []
    for name in effect_names:
        effect = get_effect(name, audio_manager)
        if effect:
            effects.append(effect)
        else:
            print(f"Warning: Effect '{name}' not found. Skipping.")

    # If no valid effects, use original
    if not effects:
        print("No valid effects specified. Using 'original'.")
        effects = [get_effect("original")]

    # Display active effects
    print("Active effects: " + " → ".join([effect.__class__.__name__.replace("Effect", "") for effect in effects]))

    # Access webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {args.camera}.")
        sys.exit(1)

    # Get window dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Start screen recording if requested
    if args.record:
        # Generate output filename
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"output_{timestamp}.mp4"
        else:
            output_filename = args.output
            if not output_filename.lower().endswith('.mp4'):
                output_filename += '.mp4'

        # Full path to output
        output_path = os.path.abspath(output_filename)

        # Create AppleScript to start screen recording
        # This script will record just the window titled "Webcam Effects"
        applescript = f'''
        tell application "QuickTime Player"
            set newMovie to new screen recording
            tell application "System Events"
                keystroke "w" using {{command down, shift down}}  -- Switch to window recording mode
                delay 1
                click at {{400, 400}}  -- Click inside the window (adjust coordinates if needed)
                delay 0.5
                keystroke return  -- Start recording
            end tell
        end tell
        '''

        try:
            print("Starting window recording. Please click on the 'Webcam Effects' window when prompted.")
            subprocess.Popen(['osascript', '-e', applescript])
            print(f"Recording will be saved when you quit the application.")
            print("Note: You'll need to manually save the recording when prompted after quitting.")
        except Exception as e:
            print(f"Error starting screen recording: {e}")

    # Create window with consistent name for recording
    window_name = 'Webcam Effects'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply all effects
        result = frame.copy()
        for effect in effects:
            result = effect.process(result)

        # Display the result
        cv2.imshow(window_name, result)

        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    # Stop audio processing
    if audio_manager:
        audio_manager.stop()

    # Remind about screen recording
    if args.record:
        print("If you started screen recording, please save the recording when prompted.")
        # AppleScript to stop recording
        stop_script = '''
        tell application "QuickTime Player"
            stop document 1
            activate
        end tell
        '''
        try:
            subprocess.run(['osascript', '-e', stop_script])
        except Exception as e:
            print(f"Error stopping recording: {e}")
            print("Please manually stop the QuickTime recording.")

if __name__ == "__main__":
    main()