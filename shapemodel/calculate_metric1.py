import os
import glob
import shapeworks as sw
import pytorch3d
import numpy as np
import torch
from tqdm import tqdm
import shutil

from DataAugmentationUtils import Embedder, Utils
import subprocess
from pytorch3d.loss import chamfer_distance
from data_augmentation import Gaussian_Sampler

def get_particles(model_path):
    f = open(model_path, "r")
    data = []
    for line in f.readlines():
        points = line.split()
        points = [float(i) for i in points]
        data.append(points)
    return(data)

def get_particle_files(model_dir):
    world_particle_list = []
    local_particle_list = []
    for file in os.listdir(model_dir):
        if "meanshape" in file:
            continue
        if "local" in file:
            local_particle_list.append(os.path.join(model_dir, file))
        if "world" in file:
            world_particle_list.append(os.path.join(model_dir, file))

    world_particle_list = sorted(world_particle_list)
    local_particle_list = sorted(local_particle_list)
    return local_particle_list

        

def convertToMesh(dt_path):
    save_path = dt_path.replace("distance_transforms", "mesh")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_dts = sorted(glob.glob(os.path.join(dt_path, "*.nrrd")))
    for i in range(len(all_dts)):
        dt = sw.Image(all_dts[i])
        dt.toMesh(0).remesh(10000, 1.0).write(all_dts[i].replace("distance_transforms", "mesh").replace(".nrrd", ".vtk"))
    return save_path, len(all_dts)
    
    

def calculate_PCA_analysis(dt_path, particle_path, num_modes):
    template_mesh_new = "tmp/template_mesh"+str(num_modes)+".vtk"
    template_particles_new = "tmp/template_particles"+str(num_modes)+".particles"
    shutil.copy(template_mesh, template_mesh_new)
    shutil.copy(template_particles, template_particles_new)
    all_particles = get_particle_files(particle_path)
    particleSystem = sw.ParticleSystem(all_particles)
    all_dts = sorted(glob.glob(os.path.join(dt_path, "*.vtk")))

    comp = sw.ShapeEvaluation.ComputeCompactness(particleSystem=particleSystem, nModes=num_modes)
    
    point_matrix = Utils.create_data_matrix(all_particles)    
    point_embedder = Embedder.PCA_Embbeder(np.array(point_matrix), num_dim=num_modes)
    pca_scores = point_embedder.PCA_scores
    
    all_gens = []
    for i in range(len(all_particles)):
        current_particles = point_embedder.project(pca_scores[i])
        np.savetxt(os.path.join("tmp/p"+str(num_modes)+".particles"), current_particles)
        execCommand = ["shapeworks", 
                    "warp-mesh", "--reference_mesh", template_mesh_new,
                    "--reference_points", template_particles_new,
                    "--target_points", "tmp/p"+str(num_modes)+".particles" ]
        subprocess.check_call(execCommand)
        rMesh = sw.Mesh("tmp/p"+str(num_modes)+".vtk")
        gtMesh = sw.Mesh(all_dts[i])
        rMesh_verts = torch.tensor(rMesh.points()).float().unsqueeze(0)
        gtMesh_verts = torch.tensor(gtMesh.points()).float().unsqueeze(0)
        cd_l2, _ = chamfer_distance(gtMesh_verts, rMesh_verts, norm=2)
        all_gens.append(cd_l2.squeeze().item())
    
    gen = np.mean(np.array(all_gens))
    
    pca_distribution = Gaussian_Sampler()
    pca_distribution.fit(pca_scores)

    
    all_specs = []
    for i in range(1000):
        sampled_pca, closest_index = pca_distribution.sample()
        sampled_points = point_embedder.project(sampled_pca)
        
        np.savetxt(os.path.join("tmp/p"+str(num_modes)+".particles"), sampled_points)
        execCommand = ["shapeworks", 
                    "warp-mesh", "--reference_mesh", template_mesh_new,
                    "--reference_points", template_particles_new,
                    "--target_points", "tmp/p"+str(num_modes)+".particles" ]
        subprocess.check_call(execCommand)
        rMesh = sw.Mesh("tmp/p"+str(num_modes)+".vtk")
        gtMesh = sw.Mesh(all_dts[closest_index])
        rMesh_verts = torch.tensor(rMesh.points()).float().unsqueeze(0)
        gtMesh_verts = torch.tensor(gtMesh.points()).float().unsqueeze(0)
        cd_l2, _ = chamfer_distance(gtMesh_verts, rMesh_verts, norm=2)
        all_specs.append(cd_l2.squeeze().item())
    
    spec = np.mean(np.array(all_specs))
    
    return [comp, gen, spec]
    
    
    
from joblib import Parallel, delayed

template_mesh = "la/gt/groomed/template_mesh.vtk"
template_particles = "la/gt/groomed/template_particles.particles"
dt_path = "la/gt/groomed_test/distance_transforms"
particle_path = "la/gt/shape_models_1024_fd/la_particles"
tqdm.write("--- converting dts to meshes ---")
mp, count = convertToMesh(dt_path)
tqdm.write("---- Calculating Metrics ----")
count -= 1
out = Parallel(n_jobs=count)(delayed(calculate_PCA_analysis)(mp,particle_path,i+1) for i in tqdm(range(count), desc="Iter: "))
# out = [calculate_PCA_analysis(mp,particle_path,i+1) for i in tqdm(range(count), desc="Iter: ")]

tqdm.write("---- Saving Metrics ----")

out = np.array(out)
np.savez("la/gt/groomed_test/stats.npz", comp=out[:,0], gen=out[:,1], spec=out[:,2])
tqdm.write("---- Done ----")




    
    