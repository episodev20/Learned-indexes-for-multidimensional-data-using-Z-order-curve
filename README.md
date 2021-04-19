# Generating Synthetic 2D-Datasets

## Uniform Distribution

Execute the following command to create 10M uniformly distributed points. Pipe
the result to a file!

```
ruby generate_uniform.rb 10000000 > ds_uniform.csv
```


## Gaussian Distribution

Execute the following command to create 10M gaussian distributed points. Pipe
the result to a file!

```
ruby generate_gaussian.rb 10000000 > ds_gaussian.csv
```


## Point Range

The x and y values of the generated 2d points have the following range
[0,2**16-1]. When interleaved with the z-order curve this generates unsigned 32
bit integers.


## Generated Datasets

The already generated datasets have the following *unique* number of points:

ds_uniform.csv:  9988215
ds_gaussian.csv: 9908107
