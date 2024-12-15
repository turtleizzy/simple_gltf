# Simple GLTF

## Overview

Simple GLTF is a Python library that provides a user-friendly interface for parsing and accessing data within GLTF (GL Transmission Format) files. It leverages the `pygltflib` library to handle the low-level details of GLTF parsing, while offering a higher-level API for easy data manipulation and access.

## Features

- **Pythonic Interface**: Provides a clean and intuitive interface for working with GLTF data.
- **Data Access**: Easily access various components of a GLTF file such as scenes, nodes, meshes, primitives, materials, etc.

## Installation

You can install Simple GLTF using pip:

```bash
pip install simple-gltf
```

Alternatively, you can clone the repository and install it manually:

```bash
git clone https://github.com/turtleizzy/simple_gltf.git
cd simple_gltf
pip install .
```

## Usage

Here's a basic example of how to use Simple GLTF to parse a GLTF file and access its data:

```python
import pygltflib
from simple_gltf import RootInterface

# Load a GLTF file
gltf = pygltflib.GLTF2().load('path_to_your_file.glb')

# Create a RootInterface object
root = RootInterface(gltf)

# Access scenes
scenes = root.scenes

# Access the default scene
default_scene = root.scene

# Access nodes in the default scene
nodes = default_scene.nodes

# Access a specific node's mesh
mesh = nodes[0].mesh

# Access primitives in the mesh
primitives = mesh.primitives

# Access position data of the first primitive
positions = primitives[0].position

print(positions)
```

## API Reference

### Classes

- **RootInterface**

  - `scenes`: List of `SceneInterface` objects.
  - `scene`: Default `SceneInterface` object.

- **SceneInterface**

  - `nodes`: List of `NodeInterface` objects.

- **NodeInterface**

  - `children`: List of child `NodeInterface` objects.
  - `mesh`: `MeshInterface` object for the attached mesh.
  - `matrix`: Transformation matrix of the node.
  - `name`: Name of the node.

- **MeshInterface**

  - `primitives`: List of `PrimitiveInterface` objects.
  - `name`: Name of the mesh.

- **PrimitiveInterface**
  - `position`: Numpy array of vertex positions.
  - `normal`: Numpy array of vertex normals.
  - `tangent`: Numpy array of vertex tangents.
  - `color`: List of numpy arrays containing vertex colors.
  - `texcoords`: List of numpy arrays containing texture coordinates.
  - `material`: Material object associated with the primitive.
  - `mode`: Rendering mode of the primitive.

## Contributing

Contributions to Simple GLTF are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [pygltflib](https://github.com/KhronosGroup/pygltflib): A Python library for reading and writing GLTF files.
- [Khronos Group](https://www.khronos.org/gltf/): Maintainers of the GLTF specification.

Feel free to reach out if you have any questions or need further assistance!
