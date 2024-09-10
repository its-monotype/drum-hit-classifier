import time
import wave

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudiowpatch as pyaudio


def capture_audio(filename, duration=6.0, chunk_size=1024):
    with pyaudio.PyAudio() as p:
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("Looks like WASAPI is not available on the system. Exiting...")
            return

        # Get default WASAPI speakers
        default_speakers = p.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        if not default_speakers["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                print("Default loopback output device not found.")
                print("Run `python -m pyaudiowpatch` to check available devices.")
                print("Exiting...")
                return

        print(
            f"Recording from: ({default_speakers['index']}){default_speakers['name']}"
        )

        wave_file = wave.open(filename, "wb")
        wave_file.setnchannels(default_speakers["maxInputChannels"])
        wave_file.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(int(default_speakers["defaultSampleRate"]))

        def callback(in_data, frame_count, time_info, status):
            wave_file.writeframes(in_data)
            return (in_data, pyaudio.paContinue)

        with p.open(
            format=pyaudio.paInt16,
            channels=default_speakers["maxInputChannels"],
            rate=int(default_speakers["defaultSampleRate"]),
            frames_per_buffer=chunk_size,
            input=True,
            input_device_index=default_speakers["index"],
            stream_callback=callback,
        ):
            print(
                f"Recording... The next {duration} seconds will be written to {filename}"
            )
            time.sleep(duration)

        wave_file.close()
        print("Recording finished.")


def classify_chunk(chunk, rate):
    fft_result = np.fft.fft(chunk, n=4096)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / rate)
    magnitude = np.abs(fft_result)

    # Focus on relevant frequency ranges for drum hits
    min_freq, max_freq = 800, 5000
    filtered_indices = (frequencies > min_freq) & (frequencies < max_freq)
    filtered_frequencies = frequencies[filtered_indices]
    filtered_magnitude = magnitude[filtered_indices]

    if len(filtered_frequencies) > 0:
        weighted_avg_freq = np.sum(filtered_frequencies * filtered_magnitude) / np.sum(
            filtered_magnitude
        )
    else:
        weighted_avg_freq = 0

    return "п" if weighted_avg_freq > 2000 else "л"


def classify_audio(audio_file):
    y, sr = librosa.load(audio_file)
    y = y / np.max(np.abs(y))

    silence_threshold = 0.1

    non_silent_intervals = librosa.effects.split(y, top_db=silence_threshold * 100)

    results = []
    timings = []
    for start, end in non_silent_intervals:
        chunk = y[start:end]
        result = classify_chunk(chunk, sr)
        results.append(result)
        timings.append((start / sr, end / sr))

    for result in results:
        print(result)

    plt.figure(figsize=(14, 6))
    plt.plot(np.arange(len(y)) / sr, y, label="Waveform")

    plt.grid(True, which="both", axis="both", linestyle="--", linewidth=0.7)
    plt.xticks(np.arange(0, len(y) / sr, step=0.25))

    for (start, end), classification in zip(timings, results):
        color = "red" if classification == "п" else "green"
        plt.axvspan(start, end, color=color, alpha=0.3, label=classification)

    plt.title("Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")

    plt.show()


def main():
    filename = "loopback_record.wav"
    capture_audio(filename)
    classify_audio(filename)


if __name__ == "__main__":
    main()
