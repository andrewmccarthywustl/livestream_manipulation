import cv2
import numpy as np
import argparse
import sys
import threading
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
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

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
        cv2.imshow('Webcam Effects', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Stop audio processing
    if audio_manager:
        audio_manager.stop()

if __name__ == "__main__":
    main()