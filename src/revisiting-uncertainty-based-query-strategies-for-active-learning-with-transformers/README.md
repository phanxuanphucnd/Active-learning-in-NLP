# Revisiting Uncertainty-based Query Strategies for Active Learning with Transformers

Accepted at ACL 2022 Findings.

## Summary

This repository contains the code to reproduce the paper 
"[Revisiting Uncertainty-based Query Strategies for Active Learning with Transformers](https://webis.de/publications.html#schroeder_2022a)" (C. Schröder, A. Niekler and M. Potthast. 2022).

Large parts of the query strategies have been migrated into the [small-text library](https://github.com/webis-de/small-text).
Don't forget to check this out too if you're interested in active learning components *for both experiments and applications*.

## Installation

The installation is described in the [installation instructions](INSTALL.md). It is straightforward, except that 
you may need to make adjustments to install a Pytorch version that is compatible with your CUDA version.

# Usage

How to run this experiment (including configuration and inspecting obtained results)
is described in the [usage instructions](USAGE.md).

## Citation

As long as the proceedings are not available yet, you can cite the preprint:

```
@Article{schroeder:2022,
  author = {Christopher Schr{\"o}der and Andreas Niekler and Martin Potthast},
  journal = {arXiv preprint arXiv:2107.05687},
  title = {Revisiting Uncertainty-based Query Strategies for Active Learning with Transformers},
  url = {https://arxiv.org/abs/2107.05687},
  year = {2022}
}
```

## Acknowledgments

This research was partly funded by the Development Bank of Saxony (SAB) under project number 100335729.
