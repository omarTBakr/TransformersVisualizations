# Draft LinkedIn post

Attach `linkedin-carousel.pdf` as a document post, or `walkthrough.mp4` as a video.

---

"An image is worth 16×16 words." But what actually happens to those words? 🔍

The second model in my open-source Transformers Visualizations series is the Vision Transformer (ViT). It's a real PyTorch forward pass on synthetic images, shown one tensor operation at a time:

➡️ The image cut into P×P patches, each flattened to P²·C pixel values
➡️ The patch projection E as a matrix multiplication: click any cell to see its dot product, checked against the model (a Conv2d with kernel = stride = P is exactly that matmul)
➡️ The [class] token, position embeddings, pre-norm LayerNorm, the residual adds and GELU, written out in LaTeX with the real values
➡️ Take [CLS] → head → softmax over the classes
➡️ What this code adds beyond the paper, such as an extra Linear and sinusoidal positions, is flagged on the diagram and in the code
➡️ Three sizes: Tiny 6×6 (every value fits on screen), Small 12×12 RGB and ViT-Base/16

Nothing is hand-drawn: the page is generated from a traced forward pass (forward hooks + sys.settrace).

🔗 github.com/omarTBakr/TransformersVisualizations

#MachineLearning #DeepLearning #ComputerVision #VisionTransformer #PyTorch #AIEducation
