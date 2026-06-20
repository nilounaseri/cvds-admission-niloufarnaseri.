# cvds-admission-niloufarnaseri

This repository contains my submission for the intake of the Master Computer
Vision & Data Science (2026–2027) at NHL Stenden.

Applicant: Niloufar Naseri

It contains the two pieces of programming evidence requested in the admission
form. Both files are in the root of this repository:

Part 2a — Personal code example

File: convolution.py

A 2D image convolution implemented from scratch in Python with NumPy, used for
edge detection (Sobel) and Gaussian blur. The script is organised into small,
single-purpose functions (pad_with_zeros, convolve, gaussian_kernel) and
follows the PEP 8 style guide. It uses no built-in convolution routine, the
sliding-window operation is written by hand to show the underlying algorithm,
which is also the basis of convolutional neural networks.

Running the file produces a three-panel figure (original image, detected edges,
blurred image), saved as convolution_demo.png.



Part 2b — Solutions to the admission assignments

File: debugging-correction.ipynb

My solutions to the four debugging exercises from the admission assignments. The
notebook is saved in an executed state, so every cell shows its output. For
each exercise it identifies the bug, explains why it occurs, and applies the
fix.
