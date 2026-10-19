# ISBI 2021 Paper Submission Templates
Word and Latex template documents for [ISBI 2021](https://biomedicalimaging.org/2021/) submissions can be found here.

# Usage
Please refer to the [IEEE Author Center](https://ieeeauthorcenter.ieee.org/publish-with-ieee/author-education-resources/guidelines-and-policies/policy-posting-your-article/) for usage guidelines on the templates. Specific intructions for ISBI 2021 submission can be found on the [conference website](https://biomedicalimaging.org/2021/full-paper-instructions/).

# Licence
This material is subject to the LaTeX Project Public License. See http://www.ctan.org/license/lppl1.3 for the details of that license.


---
name: Proj2Proj Self-Supervised LDCT Reconstruction
assignee: Mehmet Ozan Unal
github: mozanunal
status: published
---

# Proj2Proj

Self-supervised low-dose CT reconstruction trained directly in the projection
(sinogram) domain. LDCT projections are partitioned into complementary subsets
and the network learns a mapping between them, so reconstruction needs no
paired normal-dose data.

Two publications came out of this line of work: the ISBI 2021 conference paper
that introduced the self-supervised training strategy, and the PeerJ Computer
Science 2024 journal article that extended it into a complete, validated
pipeline.

## Artifacts

- Journal manuscript (2024): [`peerj/`](peerj/)
- Conference manuscript (2021): [`isbi/`](isbi/)

## Publications

### PeerJ Computer Science 2024

Proj2Proj: Self-Supervised Low-Dose CT Reconstruction.
Mehmet Ozan Unal, Metin Ertas, Isa Yildirim.
[peerj.com/articles/cs-1849](https://peerj.com/articles/cs-1849/)

```bibtex
@article{unal2024proj2proj,
  title={Proj2Proj: self-supervised low-dose CT reconstruction},
  author={Unal, Mehmet Ozan and Ertas, Metin and Yildirim, Isa},
  journal={PeerJ Computer Science},
  volume={10},
  pages={e1849},
  year={2024},
  publisher={PeerJ Inc.}
}
```

### IEEE ISBI 2021

Self-supervised training for low-dose CT reconstruction.
Mehmet Ozan Unal, Metin Ertas, Isa Yildirim.
[IEEE Xplore](https://ieeexplore.ieee.org/document/9434028)

```bibtex
@inproceedings{unal2021self,
  title={Self-supervised training for low-dose ct reconstruction},
  author={Unal, Mehmet Ozan and Ertas, Metin and Yildirim, Isa},
  booktitle={2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
  pages={69--72},
  year={2021},
  organization={IEEE}
}
```