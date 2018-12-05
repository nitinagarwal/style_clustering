# Style Clustering

A 3D mesh can be represented by Images, Points, Voxels or Mesh. In this code we are trying to compare between
these representations and analyze the tradeoffs between them. We can compare them on various tasks.

Tasks:
1. Capturing details: 
Question: Which representation would be best for capturing details?
Dataset Used: [Co-Locating Style-Defining Elements on 3D Shapes](https://dl.acm.org/citation.cfm?id=3092817#)


## Getting Started

To run the demo, download the sample data
```
bash download_dataset.sh
```

Compile mex files
```
mex solveLaplace.cpp
```

Please read the run_me.m file.

## Citation
If you use the code/data, please cite the following paper:

```
@article{agarwal2017mouse,
  author = {Nitin Agarwal, Xiangmin Xu, Gopi Meenakshisundaram},
  title = {Geometry Processing of Conventionally Produced Mouse Brain Slice Images},
  journal = {arXiv:1712.09684},
  year = {2017}
}
```

## License

Our code is released under MIT License (see License file for details).

## Contact

Please contact [Nitin Agarwal](http://www.ics.uci.edu/~agarwal/) if you have any questions or comments.
