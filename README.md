# homo



Title: High-Order Matching for One-Step Shortcut Diffusion Models. 

Authors: Bo Chen, Chengyue Gong, Xiaoyu Li, Yingyu Liang, Zhizhou Sha, Zhenmei Shi, Zhao Song, Mingda Wan

Arxiv Link: https://arxiv.org/pdf/2502.00688

## 📁 Repository Structure

Each subfolder corresponds to a specific toy target distribution:
- `2round_spiral`, `3round_spiral`: Two or three-rotation spiral distributions
- `4mode`, `5mode`, `8mode`: Multi-modal Gaussian mixtures with 4, 5, or 8 clusters
- `circle`, `irr_circle`: Regular and irregular circular patterns
- `dotpluscircle`: A combined dot and circle pattern
- `spiral`: Classic spiral pattern
- `figures`: Folder to save generated visualizations

Each folder contains the following types of `.py` files:
- Files name with `1`: Implement **First Order** terms
- Files name with `2`: Implement **Second Order** terms
- Files name with `3`: Implement **Self-Consistency** terms

Additionally, for `2round_spiral`, `3round_spiral`, and `dotpluscircle`, each script is further categorized:
- Prefix `old`: Uses the **original trajectory setting**
- Prefix `new`: Uses the **updated trajectory setting**

## ⚙️ Getting Started

1. **Clone the repo:**

```bash
git clone https://github.com/magiclinux/homo.git
cd homo
````

2. **Set up the environment using conda:**

```bash
conda create -n homo python=3.12
conda activate homo
pip install -r requirements.txt 
```
``torch'' may need to be installed from [the website](https://pytorch.org/get-started/locally/).

3. **Run an example script:**

```bash
cd 4mode
python run_4_123.py
```

4. **Check generated plots:**

All figures are saved to the `figures/` directory or shown interactively.
