# Fast Image Resizer

A high-performance batch image resizing tool using PyTorch with GPU acceleration. Optimized for medical imaging (supports TIFF/PNG) but works with any images.


## Features

- ⚡ ​**​GPU-accelerated​**​ resizing (10x faster than CPU methods)
- 🖼️ Supports ​**​TIFF and PNG​**​ formats (common in medical imaging)
- 🔄 ​**​Batch processing​**​ for optimal performance
- 📏 ​**​Configurable target size​**​ (default: 512x512)
- 🧵 ​**​Multi-threaded​**​ loading with configurable workers
- ✅ ​**​Preserves aspect ratio​**​ with bilinear interpolation

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/PolskieStronk/fast-image-resizer.git
   cd fast-image-resizer
