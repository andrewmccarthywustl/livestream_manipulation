import cv2
import numpy as np
import argparse
import sys
import os
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

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Webcam effects application")
    parser.add_argument("-e", "--effects", type=str, default="original",
                      help="Effects to apply, comma-separated (e.g., 'grayscale,neon,glitch')")
    parser.add_argument("-l", "--list", action="store_true",
                      help="List all available effects")
    parser.add_argument("-c", "--camera", type=int, default=0,
                      help="Camera index to use (default: 0)")
    parser.add_argument("--list-cameras", action="store_true",
                      help="List all available camera devices")
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

    # List effects if requested
    if args.list:
        print("Available effects:")
        effects = get_all_effects()
        for name, effect_class in sorted(effects.items()):
            description = effect_class.__doc__ or "No description"
            print(f"  - {name}: {description}")
        sys.exit(0)

    # Parse effect names
    effect_names = [name.strip().lower() for name in args.effects.split(',')]

    # Get effect instances
    effects = []
    for name in effect_names:
        effect = get_effect(name)
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

    # Create window
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

if __name__ == "__main__":
    main()