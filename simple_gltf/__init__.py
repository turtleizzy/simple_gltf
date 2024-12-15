import numpy as np
import pygltflib
from pygltflib import GLTF2, Accessor, Scene, Node, Mesh, Primitive, Material
from typing import List, Union, Optional
from functools import cached_property

ACCESSOR_COMPONENT_TYPE_MAPPING = {
    pygltflib.BYTE: np.int8,
    pygltflib.UNSIGNED_BYTE: np.uint8,
    pygltflib.SHORT: np.int16,
    pygltflib.UNSIGNED_SHORT: np.uint16,
    pygltflib.UNSIGNED_INT: np.uint32,
    pygltflib.FLOAT: np.float32
}

ACCESSOR_COMPONENT_COUNT_MAPPING = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16
}

PRIMITIVE_MODE_MAPPING = {
    pygltflib.POINTS: "POINTS",
    pygltflib.LINES: "LINES",
    pygltflib.LINE_LOOP: "LINE_LOOP",
    pygltflib.LINE_STRIP: "LINE_STRIP",
    pygltflib.TRIANGLES: "TRIANGLES",
    pygltflib.TRIANGLE_STRIP: "TRIANGLE_STRIP",
    pygltflib.TRIANGLE_FAN: "TRIANGLE_FAN"
}

def load_data_from_accessor(gltf: GLTF2, accessor: Accessor) -> np.ndarray:
    if hasattr(accessor, "sparse") and accessor.sparse is not None:
        raise NotImplementedError("Sparse accessor is not supported yet")
    bufferView = gltf.bufferViews[accessor.bufferView]
    buffer = gltf.buffers[bufferView.buffer]
    data = gltf.get_data_from_buffer_uri(buffer.uri)
    res_data = np.frombuffer(
        data,
        dtype=ACCESSOR_COMPONENT_TYPE_MAPPING[accessor.componentType],
        count=accessor.count * ACCESSOR_COMPONENT_COUNT_MAPPING[accessor.type],
        offset=bufferView.byteOffset + accessor.byteOffset
    )
    if accessor.type.startswith("VEC"):
        dim = int(accessor.type[3:])
        res_data = res_data.reshape((-1, dim))
    if accessor.type.startswith("MAT"):
        dim = int(accessor.type[3:])
        res_data = res_data.reshape((-1, dim, dim))
    return res_data
        
class PrimitiveInterface:
    def __init__(self, gltf: GLTF2, primitive: Primitive):
        self._gltf = gltf
        self._primitive: Primitive = primitive

    def _load_data_from_accessor(self, accessor_idx: Optional[int]) -> Optional[np.ndarray]:
        if accessor_idx is not None:
            accessor = self._gltf.accessors[accessor_idx]
            return load_data_from_accessor(self._gltf, accessor)
        
    @cached_property
    def position(self) -> Optional[np.ndarray]:
        return self._load_data_from_accessor(self._primitive.attributes.POSITION)
        
    @cached_property
    def normal(self) -> Optional[np.ndarray]:
        return self._load_data_from_accessor(self._primitive.attributes.NORMAL)

    @cached_property
    def tangent(self) -> Optional[np.ndarray]:
        return self._load_data_from_accessor(self._primitive.attributes.TANGENT)

    @cached_property
    def color(self) -> List[np.ndarray]:
        valid_color_idx = [
            self._primitive.attributes.__getattribute__(k) 
                for k in dir(self._primitive.attributes) if k.startswith("COLOR_") 
        ]
        valid_color_idx = [i for i in valid_color_idx if i is not None]
        return [self._load_data_from_accessor(i) for i in valid_color_idx]

    @cached_property
    def texcoords(self) -> List[np.ndarray]:
        valid_texcoord_idx = [
            self._primitive.attributes.__getattribute__(k) 
                for k in dir(self._primitive.attributes) if k.startswith("TEXCOORD_") 
        ]
        valid_texcoord_idx = [i for i in valid_texcoord_idx if i is not None]
        return [self._load_data_from_accessor(i) for i in valid_texcoord_idx]

    @cached_property
    def material(self) -> Optional[Material]:
        if self._primitive.material is not None:
            return self._gltf.materials[self._primitive.material]
        else:
            return None

    @cached_property
    def mode(self) -> str:
        return PRIMITIVE_MODE_MAPPING[self._primitive.mode]

    def __repr__(self):
        reprstr = f"Primitive("
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
    def __init__(self, gltf: GLTF2, mesh: Union[int, Mesh]):
        self._gltf = gltf
        if isinstance(mesh, Mesh):
            self._mesh: Mesh = mesh
        else:
            self._mesh: Mesh = self._gltf.meshes[mesh]

    @cached_property
    def primitives(self) -> List[PrimitiveInterface]:
        return [PrimitiveInterface(self._gltf, i) for i in self._mesh.primitives]

    @cached_property
    def name(self) -> str:
        return self._mesh.name

    def __repr__(self) -> str:
        return f"Mesh['{self.name}']"
        
class NodeInterface:
    def __init__(self, gltf: GLTF2, node: Union[int, Node]):
        self._gltf = gltf
        if isinstance(node, Node):
            self._node: Node = node
        else:
            self._node: Node = self._gltf.nodes[node]

    @cached_property
    def children(self) -> List["NodeInterface"]:
        return [NodeInterface(self._gltf, i) for i in self._node.children]

    @cached_property
    def mesh(self) -> Optional["MeshInterface"]:
        if self._node.mesh is None:
            return None
        else:
            return MeshInterface(self._gltf, self._node.mesh)

    @cached_property
    def matrix(self) -> Optional[np.ndarray]:
        if self._node.matrix is None:
            return None
        else:
            return np.array(self._node.matrix).reshape([4, 4])

    @cached_property
    def name(self) -> str:
        return self._node.name
        
    def __repr__(self):
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
    def __init__(self, gltf: GLTF2, scene: Union[int, Scene]):
        self._gltf = gltf
        if isinstance(scene, Scene):
            self._scene = scene
        else:
            self._scene = self._gltf.scenes[scene]
        
    @cached_property
    def nodes(self) -> List["NodeInterface"]:
        return [NodeInterface(self._gltf, i) for i in self._scene.nodes]

    def __repr__(self):
        return f"Nodes: [" + ",".join([str(i) for i in self.nodes]) + "]"

class RootInterface:
    def __init__(self, gltf: GLTF2):
        self._gltf = gltf
        
    @cached_property
    def scenes(self) -> List["SceneInterface"]:
        return [SceneInterface(self._gltf, i) for i in self._gltf.scenes]

    @cached_property
    def scene(self) -> "SceneInterface":
        return SceneInterface(self._gltf, self._gltf.scenes[self._gltf.scene])

    def __repr__(self):
        return f"Scenes: [" + ",".join([str(i) for i in self.scenes]) + "]"
