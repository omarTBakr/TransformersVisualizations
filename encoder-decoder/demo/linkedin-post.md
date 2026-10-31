# Draft LinkedIn post

Attach `linkedin-carousel.pdf` as a document post, or `feature-tour.mp4` as a video.

---

How many people who've read "Attention Is All You Need" have actually seen the numbers move? 🔍

I built an interactive, open-source walkthrough of a real PyTorch encoder–decoder Transformer that shows every tensor operation, one step at a time:

➡️ Matrix multiplications broken down cell by cell: click any result to see its row × column and the dot product, checked against the model
➡️ Softmax, LayerNorm and positional encoding written out in LaTeX with the real values
➡️ ReLU zeros, the row max, −∞ masks and padding flagged where they happen
➡️ The exact line of code for every stage, plus implementation details the paper doesn't mention
➡️ Three model sizes: Tiny 3×3 (every value fits on screen), Small 9×9 (3 heads, 2 + 2 layers) and the full config

Nothing is hand-drawn: the page is generated from a traced forward pass (forward hooks + sys.settrace).

This is the first of a series. Decoder-only (GPT-style) and encoder-only (BERT-style) are next.

🔗 github.com/omarTBakr/TransformersVisualizations

#MachineLearning #DeepLearning #Transformers #PyTorch #NLP #AIEducation
