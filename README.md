# JawTrack

JawTrack is a real-time jaw motion analysis system that uses computer vision to track and analyze jaw movements. Built with MediaPipe and OpenCV, it provides quantitative measurements for jaw motion assessment.

## Features

- Real-time jaw motion tracking
- Video-based analysis
- Quantitative measurements:
  - Jaw opening distance
  - Lateral deviation
  - Movement patterns
- Data visualization
- Assessment reports
- CSV data export

## Requirements

- Python 3.10+
- OpenCV
- MediaPipe
- Gradio
- NumPy
- Pandas
- Matplotlib

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/jawtrack.git
cd jawtrack
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:

```bash
python app.py
```

2. Open your web browser and navigate to:

```
http://localhost:7860
```

3. Upload a video or use webcam for real-time analysis

## Development Setup

1. Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

2. Run tests:

```bash
pytest tests/
```

## Project Structure

```
jawtrack/
├── README.md
├── requirements.txt
├── setup.py
├── jawtrack/
│   ├── core/
│   ├── analysis/
│   └── ui/
├── tests/
└── examples/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Your Name - Initial work

## Acknowledgments

- MediaPipe team for face mesh implementation
- OpenCV community
- Gradio team for the web interface framework
# jawtrack
