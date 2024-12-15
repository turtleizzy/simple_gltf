""" A pythonic interface for parsing gltf files based on pygltflib. """

from typing import List, Union, Optional
from functools import cached_property

import numpy as np
import pygltflib

from pygltflib import GLTF2, Accessor, Scene, Node, Mesh, Primitive, Material

ACCESSOR_COMPONENT_TYPE_MAPPING = {
    pygltflib.BYTE: np.int8,
    pygltflib.UNSIGNED_BYTE: np.uint8,
    pygltflib.SHORT: np.int16,
    pygltflib.UNSIGNED_SHORT: np.uint16,
    pygltflib.UNSIGNED_INT: np.uint32,
    pygltflib.FLOAT: np.float32,
}

ACCESSOR_COMPONENT_COUNT_MAPPING = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

PRIMITIVE_MODE_MAPPING = {
    pygltflib.POINTS: "POINTS",
    pygltflib.LINES: "LINES",
    pygltflib.LINE_LOOP: "LINE_LOOP",
    pygltflib.LINE_STRIP: "LINE_STRIP",
    pygltflib.TRIANGLES: "TRIANGLES",
    pygltflib.TRIANGLE_STRIP: "TRIANGLE_STRIP",
    pygltflib.TRIANGLE_FAN: "TRIANGLE_FAN",
}


def load_data_from_accessor(gltf: GLTF2, accessor: Accessor) -> np.ndarray:
    """
    Load data from a given accessor in the GLTF file.

    :param gltf: Input GLTF2 object.
    :param accessor: The Accessor object from which to load data.
    :return: A numpy array containing the loaded data.
    :raises NotImplementedError: If the accessor is sparse, which is not currently supported.
    """
    if hasattr(accessor, "sparse") and accessor.sparse is not None:
        raise NotImplementedError("Sparse accessor is not supported yet")
    buffer_view = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[buffer_view.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    res_data = np.frombuffer(
        data,
        dtype=ACCESSOR_COMPONENT_TYPE_MAPPING[accessor.componentType],
        count=accessor.count * ACCESSOR_COMPONENT_COUNT_MAPPING[accessor.type],
        offset=buffer_view.byteOffset + accessor.byteOffset,
    )
    if accessor.type.startswith("VEC"):
        dim = int(accessor.type[3:])
        res_data = res_data.reshape((-1, dim))
    if accessor.type.startswith("MAT"):
        dim = int(accessor.type[3:])
        res_data = res_data.reshape((-1, dim, dim))
    return res_data


class PrimitiveInterface:
    """
    Interface for accessing data of a single primitive in a GLTF mesh.

    :param gltf: Input GLTF2 object.
    :param primitive: The Primitive object to interface with.
    """

    def __init__(self, gltf: GLTF2, primitive: Primitive):
        self._gltf = gltf
        self._primitive: Primitive = primitive

    def _load_data_from_accessor(self, accessor_idx: Optional[int]) -> Optional[np.ndarray]:
        """
        Load data from an accessor index.

        :param accessor_idx: Index of the accessor to load data from.
        :return: Numpy array containing the loaded data, or None if no accessor index is provided.
        """
        if accessor_idx is not None:
            accessor = self._gltf.accessors[accessor_idx]
            return load_data_from_accessor(self._gltf, accessor)
        return None

    @cached_property
    def position(self) -> Optional[np.ndarray]:
        """Numpy array of vertex positions, or None if not available."""
        return self._load_data_from_accessor(self._primitive.attributes.POSITION)

    @cached_property
    def normal(self) -> Optional[np.ndarray]:
        """Numpy array of vertex normals, or None if not available."""
        return self._load_data_from_accessor(self._primitive.attributes.NORMAL)

    @cached_property
    def tangent(self) -> Optional[np.ndarray]:
        """Numpy array of vertex tangents, or None if not available."""
        return self._load_data_from_accessor(self._primitive.attributes.TANGENT)

    @cached_property
    def color(self) -> List[np.ndarray]:
        """List of numpy arrays containing vertex colors, or an empty list if none are available."""
        valid_color_idx = [
            getattr(self._primitive.attributes, k) for k in dir(self._primitive.attributes) if k.startswith("COLOR_")
        ]
        valid_color_idx = [i for i in valid_color_idx if i is not None]
        return [self._load_data_from_accessor(i) for i in valid_color_idx]

    @cached_property
    def texcoords(self) -> List[np.ndarray]:
        """List of numpy arrays containing texture coordinates, or an empty list if none are available."""
        valid_texcoord_idx = [
            getattr(self._primitive.attributes, k) for k in dir(self._primitive.attributes) if k.startswith("TEXCOORD_")
        ]
        valid_texcoord_idx = [i for i in valid_texcoord_idx if i is not None]
        return [self._load_data_from_accessor(i) for i in valid_texcoord_idx]

    @cached_property
    def material(self) -> Optional[Material]:
        """Material object associated with this primitive, or None if not available."""
        if self._primitive.material is not None:
            return self._gltf.materials[self._primitive.material]
        return None

    @cached_property
    def mode(self) -> str:
        """Rendering mode of the primitive (e.g., TRIANGLES, LINES)."""
        return PRIMITIVE_MODE_MAPPING[self._primitive.mode]

    def __repr__(self):
        """String representation of the primitive, including its mode and available data."""
        reprstr = "Primitive("
        datas = ["mode: " + self.mode]
        if self.position is not None:
            datas.append("position: " + str(self.position.shape))
        if self.normal is not None:
            datas.append("normal: " + str(self.normal.shape))
        if self.tangent is not None:
            datas.append("tangent: " + str(self.tangent.shape))
        if len(self.color) > 0:
            datas.append("color: [" + ",".join([str(i.shape) for i in self.color]) + "]")
        if len(self.texcoords) > 0:
            datas.append("texcoords: [" + ",".join([str(i.shape) for i in self.texcoords]) + "]")
        if self.material is not None:
            datas.append("material: " + str(self.material))
        return reprstr + ", ".join(datas) + ")"


class MeshInterface:
    """
    Interface for accessing data of a single mesh in a GLTF model.

    :param gltf: Input GLTF2 object.
    :param mesh: The Mesh object or index of the mesh to interface with.
    """

    def __init__(self, gltf: GLTF2, mesh: Union[int, Mesh]):
        self._gltf = gltf
        if isinstance(mesh, Mesh):
            self._mesh: Mesh = mesh
        else:
            self._mesh: Mesh = self._gltf.meshes[mesh]

    @cached_property
    def primitives(self) -> List[PrimitiveInterface]:
        """List of Primitive objects."""
        return [PrimitiveInterface(self._gltf, i) for i in self._mesh.primitives]

    @cached_property
    def name(self) -> str:
        """Name of the mesh."""
        return self._mesh.name

    def __repr__(self) -> str:
        """String representation of the mesh, including its name."""
        return f"Mesh['{self.name}']"


class NodeInterface:
    """
    Interface for accessing data of a single node in a GLTF scene.

    :param gltf: Input GLTF2 object.
    :param node: The Node object or index of the node to interface with.
    """

    def __init__(self, gltf: GLTF2, node: Union[int, Node]):
        self._gltf = gltf
        if isinstance(node, Node):
            self._node: Node = node
        else:
            self._node: Node = self._gltf.nodes[node]

    @cached_property
    def children(self) -> List["NodeInterface"]:
        """Children nodes of this node."""
        return [NodeInterface(self._gltf, i) for i in self._node.children]

    @cached_property
    def mesh(self) -> Optional["MeshInterface"]:
        """MeshInterface object for the mesh attached to this node, or None if no mesh is attached."""
        if self._node.mesh is None:
            return None
        return MeshInterface(self._gltf, self._node.mesh)

    @cached_property
    def matrix(self) -> Optional[np.ndarray]:
        """Transformation matrix of the node, or None if no matrix is specified."""
        if self._node.matrix is None:
            return None
        return np.array(self._node.matrix).reshape([4, 4])

    @cached_property
    def name(self) -> str:
        """Name of the node."""
        return self._node.name

    def __repr__(self):
        """String representation of the node, including its name, children, mesh, and transformation matrix."""
        reprstr = f"Node['{self.name}']: ("
        datas = []
        children = self.children
        if len(children) > 0:
            datas.append("children: [" + ",".join([str(i) for i in self.children]) + "]")
        if self._node.mesh is not None:
            datas.append("mesh: " + str(self.mesh))
        if self._node.matrix is not None:
            datas.append("matrix: " + ",".join(np.array_str(self.matrix, precision=3, suppress_small=True).split("\n")))
        reprstr += ",".join(datas)
        reprstr += ")"
        return reprstr


class SceneInterface:
    """
    Interface for accessing data of a single scene in a GLTF model.

    :param gltf: Input GLTF2 object.
    :param scene: The Scene object or index of the scene to interface with.
    """

    def __init__(self, gltf: GLTF2, scene: Union[int, Scene]):
        self._gltf = gltf
        if isinstance(scene, Scene):
            self._scene = scene
        else:
            self._scene = self._gltf.scenes[scene]

    @cached_property
    def nodes(self) -> List["NodeInterface"]:
        """List of NodeInterface objects for each node in the scene."""
        return [NodeInterface(self._gltf, i) for i in self._scene.nodes]

    def __repr__(self):
        """String representation of the scene, including its nodes."""
        return "Nodes: [" + ",".join([str(i) for i in self.nodes]) + "]"


class RootInterface:
    """
    Interface for accessing data of the root level in a GLTF model.

    :param gltf: Input GLTF2 object.
    """

    def __init__(self, gltf: GLTF2):
        self._gltf = gltf

    @cached_property
    def scenes(self) -> List["SceneInterface"]:
        """List of SceneInterface objects for each scene in the model."""
        return [SceneInterface(self._gltf, i) for i in self._gltf.scenes]

    @cached_property
    def scene(self) -> Optional["SceneInterface"]:
        """SceneInterface object for the default scene of the model."""
        if self._gltf.scene is None:
            if len(self._gltf.scenes) == 0:
                return None
            else:
                return SceneInterface(self._gltf, self._gltf.scenes[0])
        return SceneInterface(self._gltf, self._gltf.scenes[self._gltf.scene])

    def __repr__(self):
        """String representation of the root, including its scenes."""
        return "Scenes: [" + ",".join([str(i) for i in self.scenes]) + "]"
