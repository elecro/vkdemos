# Single file Vulkan example(s)

## vktriangle.cpp

Draws a triangle and saves it as a ppm image (with minimal "helper" methods).

For dependencies and more details see the `vktriangle.cpp` file.

Usage:

```sh
$ g++ vktriangle.cpp -o triangle -lvulkan -lshaderc_shared -std=c++11
$ ./triangle
```
