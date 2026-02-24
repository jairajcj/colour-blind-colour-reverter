# Color Blindness Correction System

A real-time camera application that helps colorblind individuals see colors more accurately using advanced Daltonization algorithms. Supports local webcams and IP/network cameras.

## Overview

This software uses your camera as an "eye" and displays corrected colors on the screen, allowing colorblind individuals to perceive colors they normally cannot distinguish. The system supports three types of color blindness:

- **Protanopia** (Red-blind): Missing L-cones, difficulty distinguishing red from green
- **Deuteranopia** (Green-blind): Missing M-cones, difficulty distinguishing red from green
- **Tritanopia** (Blue-blind): Missing S-cones, difficulty distinguishing blue from yellow

## How It Works

The application uses an **optimized Daltonization algorithm** with precomputed single-pass matrix transformation to achieve maximum performance. Instead of the traditional multi-step process (RGB→LMS → simulate → error correct → LMS→RGB), this implementation precomputes a single 3×3 combined matrix at startup:

```
M_combined = M_rgb2lms × (I + (I - M_sim) × M_err_corr) × M_lms2rgb
```

This reduces the per-frame workload to a single `cv2.transform()` call — a highly optimized C++ function, enabling true real-time processing:
1. Precomputes the combined matrix once based on the selected deficiency.
2. Applies the matrix via hardware-accelerated transformation.
3. Clips and converts results back to displayable uint8.

This achieves **100+ FPS at 640×480** and **40+ FPS at 720p**, ensuring smooth real-time performance even at HD resolutions.

## Installation

### Prerequisites
- Python 3.7 or higher
- A working webcam (local or IP camera)

### Setup

```bash
pip install -r requirements.txt
```

## Usage

### Local Webcam (Default)

```bash
python colorblind_correction.py
```

### IP Camera / External Network Camera

```bash
# Android IP Webcam app
python colorblind_correction.py --ip http://192.168.1.100:8080/video

# RTSP security camera
python colorblind_correction.py --ip rtsp://user:pass@192.168.1.5:554/stream1

# MJPEG stream
python colorblind_correction.py --ip http://192.168.1.100/mjpg/video.mjpg
```

### Multiple Local Cameras

```bash
python colorblind_correction.py --camera 1
```

### Custom Resolution

```bash
python colorblind_correction.py --width 1280 --height 720
```

### Controls

| Key | Action |
|-----|--------|
| **N** | Normal mode (no correction) |
| **P** | Protanopia correction (red-blind) |
| **D** | Deuteranopia correction (green-blind) |
| **T** | Tritanopia correction (blue-blind) |
| **Q** | Quit application |

## Performance Benchmarks

Tested on multiple resolution tiers to confirm real-time viability:

| Resolution | Hardware Tier | Avg Time (ms) | FPS | Status |
|:-----------|:--------------|:--------------|:----|:-------|
| **640×480** | SD / Mobile | 8.42 ms | **118.7 FPS** | ✅ Ultra Fast |
| **1280×720** | HD / Tablet | 24.73 ms | **40.4 FPS** | ✅ Smooth real-time |
| **1920×1080** | Full HD Desktop | 64.48 ms | **15.5 FPS** | ⚠️ Process Intensive |

Run benchmarks on your own hardware:
```bash
python performance_benchmark.py
```

## Project Structure

```
.
├── colorblind_correction.py          # Core correction engine + camera app
├── performance_benchmark.py          # Hardware tier benchmarking tool
├── check_cameras.py                 # Utility to check available cameras
├── demo_static_image.py              # Static image demo
├── run_once_test.py                  # Quick validation test (non-GUI)
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## Troubleshooting & Support

### IP Camera Integration
- **Auto-reconnects**: System attempts 5 retries on connection drop.
- **Latency**: Buffer size is optimized (size=1) for minimal network lag.
- **Formats**: Supports `http://` (Android IP Webcam), `rtsp://`, and `mjpg`.

### Common Issues
- **Camera Not Opening**: Verify no other app is using the camera; check index with `--camera 1`.
- **Low FPS**: Reduce resolution using `--width 320 --height 240`; ensure graphics drivers are up to date.

### Scientific References

- Brettel, H., Viénot, F., & Mollon, J. D. (1997). Computerized simulation of color appearance for dichromats. *Journal of the Optical Society of America A*, 14(10), 2647-2655.
- Fidaner, I. B., Aydin, T. O., & Çapın, T. K. (2005). *Adaptive Image Recoloring for Red-Green Dichromats*.

## License

This project is open source and available for educational and personal use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.


