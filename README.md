
---

# README for Computer Assignment - PETE 332

## Overview
This project is a computer assignment for the course PETE 332: Petroleum Production Engineering II at the Middle East Technical University. The assignment involves simulating and analyzing petroleum production engineering concepts using Python.

## Authors
- Mert Yücel
- Arda Çimen
- Ege Filizli
- Date: 05/06/2025

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Requirements
To run this code, you need the following Python packages:
- OpenCV
- NumPy
- Matplotlib

You can install the required packages using pip:
```bash
pip install opencv-python numpy matplotlib
```

## Code Description
The code simulates the pressure profiles in a petroleum production system. It includes the following key components:

### Input Data
- **Common Input Data**: Parameters such as reservoir pressure, perforation depth, tubing length, target oil production rate, and fluid properties.
- **Gas Injection Input Data**: Parameters related to gas injection, including valve depth and flow rate.
- **Production Tubing Input Data**: Parameters for the production tubing, including tubing dimensions and temperature.

### Functions
- **plot_section**: Plots pressure profiles, average fluid density, and dissolved gas in oil against depth.
- **dRHOv**: Calculates the change in fluid density.
- **f_2F**: Computes the friction factor.
- **k_Factor**: Calculates the K factor for pressure drop calculations.
- **RHO_av**: Computes average fluid density and related properties.
- **Z_Factor**: Calculates the compressibility factor.
- **p_drop_comp**: Computes pressure drop along the tubing.

### Execution
1. The program starts by displaying an image related to the assignment.
2. It prompts the user to enter their student ID, which influences the target oil production rate and gas injection flow rate.
3. The program calculates pressure profiles and plots the results.

## Usage
To run the program, execute the following command in your terminal:
```bash
python main.py
```
Follow the on-screen prompts to enter your student ID.

## Output
The program generates plots showing:
- Pressure profiles from the bottomhole to the wellhead.
- Average fluid density versus depth.
- Dissolved gas in oil versus depth.

## Contact
mert.yucel@metu.edu.tr
ege.filizli@metu.edu.tr
arda.cimen@metu.edu.tr
meyucel@tpao.gov.tr
