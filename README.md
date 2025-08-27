# Stable Diffusion v1.5 Image Generator 🎨

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/%20Diffusers-0.30+-yellow.svg)](https://huggingface.co/docs/diffusers/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-red.svg)](https://gradio.app/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

## 📊 Project Overview

This project implements a **high-performance AI image generation system** using Stable Diffusion v1.5 with advanced speed optimizations. Generate stunning, high-quality images from text prompts with **lightning-fast inference** and **professional-grade results**, featuring both programmatic interface and user-friendly web UI powered by Gradio.

### 🎯 Key Objectives

✔ **Ultra-Fast Generation**: Optimized SD 1.5 pipeline with xformers and model compilation  
✔ **Multiple Generation Modes**: Quick (15 steps), Quality (25 steps), and Portrait-specialized  
✔ **Memory Efficient**: Smart GPU memory management and CPU offloading  
✔ **Web Interface**: Interactive Gradio UI for non-technical users  
✔ **Batch Processing**: Generate multiple images simultaneously  
✔ **Smart File Management**: Automatic saving with timestamp and prompt-based naming  
✔ **Professional Output**: 512x512 high-quality JPEG images with 95% quality  
✔ **Cross-Platform**: Compatible with CUDA GPU acceleration and CPU fallback  

## 🚀 Performance Features

### Speed Optimizations
- **xformers Memory Efficient Attention**: 40-50% faster inference
- **Model CPU Offload**: Reduced VRAM usage by 2-3GB
- **PyTorch 2.0 Compilation**: Additional 15-20% speed boost
- **FP16 Precision**: 50% memory reduction with minimal quality loss
- **DPM Solver**: High-quality results with fewer inference steps

### Generation Times
- **Quick Mode** (15 steps): ~3-5 seconds on RTX 3080
- **Quality Mode** (25 steps): ~5-8 seconds on RTX 3080  
- **Portrait Mode** (20 steps): ~4-6 seconds on RTX 3080

## 🗂️ Project Structure

```
SD15-Image-Generator/
│
├── 📁 notebooks/
│   ├── Image_Generator.ipynb          # Complete Jupyter notebook
│   └── Image_Generator.py             # Standalone Python script
│
├── 📁 sd15_generated_images/          # Auto-generated output folder
│   ├── a_cat_sitting_in_a_garden_20250826_144146.jpg
│   ├── a_majestic_dragon_flying__20250826_152123.jpg
│   ├── ancient_Egyptian_pyramids_20250826_145155.jpg
│   └── portrait_of_a_wise_old_ma_20250826_145217.jpg
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
└── LICENSE                           # MIT License
```

## 🔍 Key Features & Functionality

### Generation Modes
1. **Quick Generate**: 15 steps for rapid prototyping and testing
2. **Quality Generate**: 25 steps for final high-quality results
3. **Portrait Generate**: Optimized for human faces with enhanced prompts

### Advanced Features
- **Negative Prompting**: Exclude unwanted elements from images
- **Guidance Scale Control**: Balance creativity vs prompt adherence
- **Seed Management**: Reproducible results for consistent outputs
- **Batch Processing**: Generate multiple variations simultaneously
- **Memory Optimization**: Automatic cleanup and efficient resource usage

### Web Interface Features
- **Intuitive UI**: Clean, modern design with responsive layout
- **Real-time Preview**: Immediate display of generated images
- **Parameter Control**: Adjust quality, quantity, and prompts
- **Download Support**: Direct image download from browser
- **Mobile Friendly**: Responsive design for all devices

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📞 Contact

**Ahmed Maher Abd Rabbo**
- 💼 [LinkedIn](https://www.linkedin.com/in/ahmed-maherr/)
- 📊 [Kaggle](https://kaggle.com/ahmedmaherabdrabbo)
- 📧 Email: ahmedbnmaher1@gmail.com
- 💻 [GitHub](https://github.com/AhmedMaherAbdRabbo)


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.