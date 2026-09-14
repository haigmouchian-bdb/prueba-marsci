# mmm_demo

Toy Marketing Mix Model (MMM) pipeline used as a training example for Git 101.
It is **not** a real client project — inputs are synthetic data.

## What it does

Reads weekly sales and media spend (TV, digital, OOH), applies a geometric
adstock transform to each channel, fits a linear regression to estimate each
channel's contribution to sales, and writes the results and the trained model
to disk.

## Setup

conda env create -f environment.yml
conda activate mmm-demo
