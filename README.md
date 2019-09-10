Anomaly Detection using CVAE
=====

Example of Anomaly Detection using CVAE [<a href="https://github.com/YeongHyeon/CVAE">Related repository</a>].

## Architecture
<div align="center">
  <img src="./figures/vae.png" width="400">  
  <p>Simplified VAE architecture.</p>
</div>

## Problem Definition
<div align="center">
  <img src="./figures/definition.png" width="600">  
  <p>'Class-1' is defined as normal and the others are defined as abnormal.</p>
</div>

## Results

### Training
<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by CVAE.</p>
</div>

<div align="center">
  <img src="./figures/latent_tr.png" width="300"><img src="./figures/latent_walk.png" width="250">
  <p>Latent vector space of training set, and reconstruction result of latent space walking.</p>
</div>

### Test
<div align="center">
  <img src="./figures/latent_te.png" width="350"><img src="./figures/test-box.png" width="400">    
  <p>Left figure shows latent vector space of test set. Right figure shows box plot with encoding loss of test procedure.</p>
</div>

## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  

## Reference
[1] Kingma, D. P., & Welling, M. (2013). <a href="https://arxiv.org/abs/1312.6114">Auto-encoding variational bayes</a>. arXiv preprint arXiv:1312.6114.  
[2] <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback Leibler divergence</a>. Wikipedia
