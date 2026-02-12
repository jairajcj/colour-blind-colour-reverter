# Color Blindness Correction System

A real-time camera application that helps colorblind individuals see colors more accurately using advanced Daltonization algorithms. Supports local webcams and IP/network cameras.

## Overview

This software uses your camera as an "eye" and displays corrected colors on the screen, allowing colorblind individuals to perceive colors they normally cannot distinguish. The system supports three types of color blindness:

- **Protanopia** (Red-blind): Missing L-cones, difficulty distinguishing red from green
- **Deuteranopia** (Green-blind): Missing M-cones, difficulty distinguishing red from green
- **Tritanopia** (Blue-blind): Missing S-cones, difficulty distinguishing blue from yellow

## Real-Life Correction Results

The system corrects colors across real-world scenarios — nature, fruits, traffic signals — not just test charts.

### Before & After Comparison Grid

![Master Comparison Grid](reallife_comparisons/master_reallife_grid.png)

### Ishihara Plate Study

![Ishihara Comparison](master_ishihara_comparison.png)

## How It Works

The application uses an **optimized Daltonization algorithm** with precomputed single-pass matrix transformation:

1. Precomputes a combined RGB→LMS→Correction→RGB matrix at startup
2. Applies the single matrix via `cv2.transform()` for hardware-accelerated speed
3. Clips and converts back to displayable uint8

This achieves **100+ FPS at 640×480** and **40+ FPS at 720p** — true real-time performance.

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

| Resolution | Quality Tier | Avg Time (ms) | FPS | Status |
|:-----------|:-------------|:---------------|:----|:-------|
| **640×480** | SD / Mobile | 8.42 ms | **118.7 FPS** | ✅ Ultra Fast |
| **1280×720** | HD / Tablet | 24.73 ms | **40.4 FPS** | ✅ Smooth Real-time |
| **1920×1080** | Full HD Desktop | 64.48 ms | **15.5 FPS** | ⚠️ Process Intensive |

Run benchmarks on your own hardware:
```bash
python performance_benchmark.py
```

## Ishihara Study Framework

A quantitative evaluation tool using Ishihara plates:

```bash
python ishihara_study.py
```

- Records participant accuracy with/without correction
- Exports results to `all_study_results.csv` for statistical analysis
- Supports toggling correction on/off during testing

## Project Structure

```
opencv/
├── colorblind_correction.py          # Core correction engine + camera app
├── performance_benchmark.py          # Hardware tier benchmarking tool
├── ishihara_study.py                 # Quantitative Ishihara plate study
├── generate_reallife_comparisons.py  # Real-life scene comparison generator
├── generate_comparisons.py           # Ishihara plate comparison generator
├── create_master_comparison.py       # Master grid generator
├── demo_static_image.py              # Static image demo
├── run_once_test.py                  # Quick validation test
├── requirements.txt                  # Dependencies
├── ishihara_plates/                  # Generated Ishihara test plates
├── comparison_results/               # Ishihara before/after images
├── reallife_comparisons/             # Real-life scene before/after images
└── README.md                         # This file
```

## Technical Details

### Optimized Single-Pass Daltonization

Instead of the traditional 4-step process (RGB→LMS, simulate, error correct, LMS→RGB), this implementation precomputes a single 3×3 combined matrix:

```
M_combined = M_rgb2lms × (I + (I - M_sim) × M_err_corr) × M_lms2rgb
```

This reduces the per-frame work to a single `cv2.transform()` call — a highly optimized C++ function.

### IP Camera Support

- Auto-reconnects up to 5 times on connection drop
- Buffer size set to 1 for minimal latency on network streams
- Supports HTTP, RTSP, and MJPEG protocols

### Color Space Transformation

Based on scientifically accurate matrices:
- **RGB → LMS**: Converts standard RGB to Long, Medium, Short wavelength cone responses
- **Simulation Matrices**: Based on research by Brettel, Viénot, and Mollon (1997)

## Troubleshooting

### Camera Not Opening
- Check if another application is using the camera
- Try `--camera 1` or `--camera 2` for multiple cameras
- For IP cameras, verify the URL and network connectivity

### IP Camera Issues
- Ensure phone/camera is on the same Wi-Fi network
- Common formats: `http://<IP>:8080/video` (Android IP Webcam)
- Check firewall settings

### Low FPS
- Reduce resolution: `--width 320 --height 240`
- Close other applications
- Update graphics drivers

## Scientific References

- Brettel, H., Viénot, F., & Mollon, J. D. (1997). Computerized simulation of color appearance for dichromats. *Journal of the Optical Society of America A*, 14(10), 2647-2655.
- Fidaner, I. B., Aydin, T. O., & Çapın, T. K. (2005). *Adaptive Image Recoloring for Red-Green Dichromats*.

## License

This project is open source and available for educational and personal use.

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
