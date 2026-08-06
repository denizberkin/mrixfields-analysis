# Mean Absolute Error (MAE) per prediction

$$
\text{MAE} = \frac{1}{|M|} \sum_{(i,j) \in M} |I_{\text{pred}}(i,j) - I_{\text{src}}(i,j)|
$$

where $M$ is the brain mask $(I_{\text{src}} > 10^{-6})$ at slice $z$, and $(i,j)$ are spatial coordinates within the mask.

# High-Pass Detail RMS (HP-RMS) per prediction

$$
\begin{aligned}
I_{\text{smooth}} &= G_{\sigma=2} * I_{\text{pred}} \\
I_{\text{hp}} &= (I_{\text{pred}} - I_{\text{smooth}}) \cdot M \\
\text{HP-RMS} &= \sqrt{\frac{1}{|M|} \sum_{(i,j) \in M} I_{\text{hp}}(i,j)^2}
\end{aligned}
$$

where $G_{\sigma=2}$ is a 2D Gaussian blur with $\sigma = 2$ pixels.
