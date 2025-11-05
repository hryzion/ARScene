"""Script used for pickling the 3D Future dataset in order to be subsequently
used by our scripts.
"""
import argparse
from concurrent.futures import process
from fileinput import filename
import os
import sys

import pickle
import trimesh
import numpy as np

SOURCE_3D_FUTURE_DIR = '/mnt/disk-1/zhx24/dataset/3dfront/object'

from plyfile import PlyElement, PlyData
import tqdm

def export_pointcloud(vertices, out_file, as_text=True):
    assert(vertices.shape[1] == 3)
    vertices = vertices.astype(np.float32)
    vertices = np.ascontiguousarray(vertices)
    vector_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = vertices.view(dtype=vector_dtype).flatten()
    plyel = PlyElement.describe(vertices, 'vertex')
    plydata = PlyData([plyel], text=as_text)
    plydata.write(out_file)


def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices

def main(argv):
    parser = argparse.ArgumentParser(
        description="Pickle the 3D Future dataset"
    )
    ##
    parser.add_argument(
        "--bbox_padding",
        type=float,
        default=0.0,
        help="The bbox padding, default 0.0 as occnet"
    )
    parser.add_argument(
        '--pointcloud_size', 
        type=int, default=30000,
        help='Size of point cloud.')
    args = parser.parse_args(argv)

    model_meta = np.load('../datasets/model_meta.npy')
    for model in tqdm.tqdm(model_meta):
        # if model == '7465':
        #     continue
        model_dir = os.path.join(SOURCE_3D_FUTURE_DIR, model)
        raw_model_path = os.path.join(model_dir, f'{model}.obj')
        texture_path = os.path.join(model_dir, 'texture.jpg')
        try:
            mesh = trimesh.load(
                raw_model_path,
                process=False,
                force="mesh",
                skip_materials=True,
                skip_texture=True
            )
        except Exception as e:
            print(e)
            print(f"Error loading {raw_model_path}, skipping...")
            continue
        bbox = mesh.bounding_box.bounds

        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)
        size = abs(bbox[1]-bbox[0])

        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)


        # sample point clouds with normals
        points, face_idx = mesh.sample(args.pointcloud_size, return_index=True)
        normals = mesh.face_normals[face_idx]

        # Compress
        dtype = np.float16
        #dtype = np.float32
        points = points.astype(dtype)
        normals = normals.astype(dtype)

        filename = raw_model_path[:-4] + "_norm_pc.npz"
        # print('Writing pointcloud: %s' % filename)
        np.savez(filename, points=points, normals=normals, loc=loc, scale=scale, size=size)



        filename = raw_model_path[:-4] + "_norm_pc.ply"
        export_pointcloud(points, filename)
        # print('Writing pointcloud: %s' % filename)

    

if __name__ == "__main__":
    main(sys.argv[1:])