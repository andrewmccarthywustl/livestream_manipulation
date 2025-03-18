import numpy as np
import pyaudio
import threading
import time

class AudioManager:
    def __init__(self, input_device_index=None):
        self.running = False
        self.thread = None
        self.input_device_index = input_device_index
        self.audio_data = {
            "volume": 0,
            "peak_volume": 0,
            "spectrum": None,
            "beat_detected": False,
            "history": [],
            "beat_history": [],
            "last_beat_time": 0
        }

    def start(self):
        """Start audio processing thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._audio_thread, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop audio processing thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

    def _audio_thread(self):
        """Audio processing thread function"""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        p = pyaudio.PyAudio()
        stream = None
        try:
            stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=self.input_device_index,
                        frames_per_buffer=CHUNK)

            # Volume history for beat detection
            volume_history = []

            while self.running:
                # Read audio data
                data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

                # Calculate volume (RMS)
                volume = np.sqrt(max(0, np.mean(data**2)))

                # Update volume history
                volume_history.append(volume)
                if len(volume_history) > 30:  # Keep last 30 chunks
                    volume_history.pop(0)

                # Calculate peak volume
                peak_volume = max(volume_history) if volume_history else volume

                # Calculate frequency spectrum
                spectrum = np.abs(np.fft.rfft(data))

                # Beat detection
                beat_detected = False
                current_time = time.time()

                # Only detect beats if enough time has passed
                if current_time - self.audio_data["last_beat_time"] > 0.1:  # At most 10 beats per second
                    # Beat if current volume is above threshold and significantly higher than average
                    avg_volume = sum(volume_history) / len(volume_history) if volume_history else 0
                    if volume > 1000 and volume > 1.5 * avg_volume:
                        beat_detected = True
                        self.audio_data["last_beat_time"] = current_time

                # Update audio data
                self.audio_data["volume"] = volume
                self.audio_data["peak_volume"] = peak_volume
                self.audio_data["spectrum"] = spectrum
                self.audio_data["beat_detected"] = beat_detected

                # Update history
                self.audio_data["history"].append((volume, spectrum))
                if len(self.audio_data["history"]) > 100:
                    self.audio_data["history"].pop(0)

                # Update beat history
                if beat_detected:
                    self.audio_data["beat_history"].append(current_time)
                    # Keep only recent beats
                    while self.audio_data["beat_history"] and self.audio_data["beat_history"][0] < current_time - 5:
                        self.audio_data["beat_history"].pop(0)

        except Exception as e:
            print(f"Audio processing error: {e}")
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            p.terminate()