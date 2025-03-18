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
    parser.add_argument("-c", "--camera", type=int, default=1,
                      help="Camera index to use (default: 1)")
    args = parser.parse_args()

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
        audio_manager = AudioManager()
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