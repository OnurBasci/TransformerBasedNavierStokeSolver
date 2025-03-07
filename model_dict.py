from model import Transolver_Irregular_Mesh, Transolver_Structured_Mesh_2D, Transolver_Structured_Mesh_3D, Transolver_Structured_Mesh2D_Encoder


def get_model(args):
    model_dict = {
        'Transolver_Irregular_Mesh': Transolver_Irregular_Mesh, # for PDEs in 1D space or in unstructured meshes
        'Transolver_Structured_Mesh_2D': Transolver_Structured_Mesh_2D,
        'Transolver_Structured_Mesh_3D': Transolver_Structured_Mesh_3D,
        'Transolver_Structured_Mesh2D_Encoder': Transolver_Structured_Mesh2D_Encoder
    }
    return model_dict[args.model]
