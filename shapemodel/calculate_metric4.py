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
import json

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

def getDistance(mesh, points):
    all_dists = np.zeros((points.shape[0]))
    for i in range(points.shape[0]):
        cp = mesh.closestPoint(points[i])
        dist = np.linalg.norm(points[i]-cp[0]) 
        all_dists[i] = dist
    return np.mean(all_dists)

def convertToMesh(dt_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_dts = sorted(glob.glob(os.path.join(dt_path, "*.nrrd")))
    def worker(i):
        dt = sw.Image(all_dts[i])
        dt.binarize()
        dt.isolate()
        dt.antialias(30).computeDT(0).gaussianBlur(1.0)
        dt.toMesh(0).remesh(10000, 1.0).write(all_dts[i].replace("distance_transforms", "mesh").replace(".nrrd", ".vtk"))
        
    Parallel(n_jobs=20)(delayed(worker)(i) for i in range(len(all_dts)))
    return len(all_dts)
    

def getGeneralization_test(num_modes, point_embedder):
    all_particles = get_particle_files(test_particles_path)
    particleSystem = sw.ParticleSystem(all_particles)
    all_dts = sorted(glob.glob(os.path.join(test_mesh_path, "*.vtk")))
    
    point_matrix = np.array(Utils.create_data_matrix(all_particles))
    data_matrix_2d = point_matrix.reshape(point_matrix.shape[0], -1).T - point_embedder.mean.reshape(point_embedder.mean.shape[0], 1)
    pca_scores = point_embedder.eigen_vectors.T @ data_matrix_2d
    pca_scores = pca_scores[:num_modes, :].T

    save_dir = os.path.join(tmp_dir, "test", str(num_modes))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    min_dist = np.inf
    max_dist = -np.inf

    all_gens = []
    def worker_gen(i):
        current_particles = point_embedder.project(pca_scores[i])
        np.savetxt(os.path.join(save_dir,"p"+str(i)+".particles"), current_particles)
        # # execCommand = ["shapeworks", 
        # #             "warp-mesh", "--reference_mesh", template_mesh,
        # #             "--reference_points", template_particles,
        # #             "--target_points", os.path.join(tmp_dir, "test" , "p"+str(i)+".particles") ]
        # subprocess.check_call(execCommand)
        # rMesh = sw.Mesh(os.path.join(tmp_dir, "test","p"+str(i)+".vtk"))
        gtMesh = sw.Mesh(all_dts[i])
        # rMesh_verts = torch.tensor(current_particles).float().unsqueeze(0)
        # gtMesh_verts = torch.tensor(gtMesh.points()).float().unsqueeze(0)
        # cd_l2, _ = chamfer_distance(rMesh_verts, gtMesh_verts, norm=2, single_directional=False)
        cd_value = getDistance(gtMesh, current_particles) #cd_l2.squeeze().item()
        info = {}
        info["pred_particles"] = os.path.join(save_dir,"p"+str(i)+".particles")
        info["gt_mesh"] = all_dts[i]
        info["cd"] = cd_value
        info["template_mesh"] = template_mesh
        info["template_particles"] = template_particles

        return info
    
    all_infos = Parallel(n_jobs=20)(delayed(worker_gen)(i) for i in tqdm(range(len(all_particles)), desc="Iter Gen: "))
    all_gens = [i["cd"] for i in all_infos]

    info = {"all_infos": all_infos}
    with open(os.path.join(save_dir, "info.json"), 'w') as f:
        json.dump(info, f, indent=4)
    gen = np.mean(np.array(all_gens))
    return gen

def getGeneralization_train(num_modes, point_embedder):
    all_particles = get_particle_files(train_particles_path)
    particleSystem = sw.ParticleSystem(all_particles)
    all_dts = sorted(glob.glob(os.path.join(train_mesh_path, "*.vtk")))
    
    point_matrix = np.array(Utils.create_data_matrix(all_particles))
    data_matrix_2d = point_matrix.reshape(point_matrix.shape[0], -1).T - point_embedder.mean.reshape(point_embedder.mean.shape[0], 1)
    pca_scores = point_embedder.eigen_vectors.T @ data_matrix_2d
    pca_scores = pca_scores[:num_modes, :].T
    save_dir = os.path.join(tmp_dir, "train", str(num_modes))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_gens = []

    min_dist = np.inf
    max_dist = -np.inf

    def worker_gen(i):
        current_particles = point_embedder.project(pca_scores[i])
        np.savetxt(os.path.join(save_dir, "p"+str(i)+".particles"), current_particles)
        # execCommand = ["shapeworks", 
        #             "warp-mesh", "--reference_mesh", template_mesh,
        #             "--reference_points", template_particles,
        #             "--target_points", os.path.join(tmp_dir, "train", "p"+str(i)+".particles") ]
        # subprocess.check_call(execCommand)
        # rMesh = sw.Mesh(os.path.join(tmp_dir, "train" ,"p"+str(i)+".vtk"))
        gtMesh = sw.Mesh(all_dts[i])
        # rMesh_verts = torch.tensor(current_particles).float().unsqueeze(0)
        # gtMesh_verts = torch.tensor(gtMesh.points()).float().unsqueeze(0)
        # cd_l2, _ = chamfer_distance(rMesh_verts, gtMesh_verts, norm=2, single_directional=False)
        cd_value = getDistance(gtMesh, current_particles) #cd_l2.squeeze().item()
        info = {}
        info["pred_particles"] = os.path.join(save_dir,"p"+str(i)+".particles")
        info["gt_mesh"] = all_dts[i]
        info["cd"] = cd_value
        info["template_mesh"] = template_mesh
        info["template_particles"] = template_particles

        return info
    
    all_infos = Parallel(n_jobs=20)(delayed(worker_gen)(i) for i in tqdm(range(len(all_particles)), desc="Iter Gen: "))
    all_gens = [i["cd"] for i in all_infos]

    info = {"all_infos": all_infos}
    with open(os.path.join(save_dir, "info.json"), 'w') as f:
        json.dump(info, f, indent=4)

    gen = np.mean(np.array(all_gens))
    return gen

def calculate_PCA_analysis(num_modes):
    all_particles = get_particle_files(train_particles_path)
    particleSystem = sw.ParticleSystem(all_particles)
    all_dts = sorted(glob.glob(os.path.join(train_mesh_path, "*.vtk")))

    comp = sw.ShapeEvaluation.ComputeCompactness(particleSystem=particleSystem, nModes=num_modes)
    
    point_matrix = Utils.create_data_matrix(all_particles)    
    point_embedder = Embedder.PCA_Embbeder(np.array(point_matrix), num_dim=num_modes)
    pca_scores = point_embedder.PCA_scores
    
    gen = getGeneralization_test(num_modes, point_embedder)
    gen_train = getGeneralization_train(num_modes, point_embedder) # don;t do this possibly
    
    
    out_dir = os.path.join(train_particles_path.replace("la_particles", "out"), str(num_modes))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    nSpecificity = sw.ShapeEvaluation.ComputeSpecificity(particleSystem=particleSystem, nModes=num_modes, saveTo=out_dir)

    xml_files = sorted(glob.glob(out_dir+'/*.xml'))
    generated_particle_files = sorted(glob.glob(out_dir+'/*.particles'))
    all_specs = []
    def worker_spec(i):
        with open(xml_files[i]) as f:
            first_line = f.readline().strip('\n')
        closest_file = os.path.basename(first_line)
        # execCommand = ["shapeworks", 
        #             "warp-mesh", "--reference_mesh", template_mesh,
        #             "--reference_points", template_particles,
        #             "--target_points", generated_particle_files[i] ]
        # subprocess.check_call(execCommand)
        rMesh = np.loadtxt(generated_particle_files[i]) #sw.Mesh(generated_particle_files[i].replace(".particles", ".vtk"))
        gtMesh = sw.Mesh(os.path.join(train_mesh_path, closest_file.replace("_local.particles", ".vtk")))
        # rMesh_verts = torch.tensor(rMesh).float().unsqueeze(0)
        # gtMesh_verts = torch.tensor(gtMesh.points()).float().unsqueeze(0)
        # cd_l2, _ = chamfer_distance(rMesh_verts, gtMesh_verts, norm=2, single_directional=False)
        cd_value = getDistance(gtMesh, rMesh) #cd_l2.squeeze().item()
        info = {}
        info["pred_particles"] = generated_particle_files[i]
        info["gt_mesh"] = os.path.join(train_mesh_path, closest_file.replace("_local.particles", ".vtk"))
        info["cd"] = cd_value
        info["template_mesh"] = template_mesh
        info["template_particles"] = template_particles

        return info
        
    
    all_infos = Parallel(n_jobs=52)(delayed(worker_spec)(i) for i in tqdm(range(len(xml_files)), desc="Iter Spec: "))
    all_specs = [i["cd"] for i in all_infos]

    info = {"all_infos": all_infos}
    with open(os.path.join(out_dir, "info.json"), 'w') as f:
        json.dump(info, f, indent=4)
    spec = np.mean(np.array(all_specs))
    # shutil.rmtree(out_dir)
    
    
    return [comp, spec, gen, gen_train]
    
    
    
from joblib import Parallel, delayed



# path = "femur_40/bcp"
# template = "n11_R_femur_1x_hip.isores.padded.com.aligned.cropped_pred"

# path = "/home/sci/asmak/Documents/Methods/ssm/run_as_is/20_exp/namic/mt"
# template = "CARMA0415_pred"

# path = "/home/sci/asmak/Documents/Methods/ssm/Ground_Truth/namic/20_percent_gt/1SSM_gt_gt"
# template = "CARMA1092_gt"

# path = "/home/sci/asmak/Documents/Methods/ssm/Ground_Truth/namic/20_percent_gt/2SSM_gt.training_DeSCO.testing"
# template = "CARMA1092_gt"

# path = "/home/sci/asmak/Documents/Methods/ssm/DeSCO/20_percent_namic/3SSM_DeSCO.training_DesCO.testing"
# template = "CARMA0937.h5_pred"

# path = "/home/sci/asmak/Documents/Methods/ssm/Ground_Truth/namic/40_percent_gt/1SSM_gt_gt"
# template = "CARMA1092_gt"

# path = "/home/sci/asmak/Documents/Methods/ssm/Ground_Truth/femur/40_percent_gt/1SSM_gt_gt"
# template = "n11_R_femur_1x_hip_gt"

# Setting I:
# path = "/home/sci/asmak/Documents/Methods/ssm/Ground_Truth/femur/40_percent_gt/2SSM_gt.training.CAML.testing"
# template = "n11_R_femur_1x_hip_gt"

# Setting II:
path = "/home/sci/asmak/Documents/Methods/weaklyssm/BCP/40_percent_femur"
template = "n11_R_femur_1x_hip.isores.padded.com.aligned.cropped_pred"

tmp_dir = os.path.join(path, "tmp")

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
    os.makedirs(os.path.join(tmp_dir, "test"))
    os.makedirs(os.path.join(tmp_dir, "train"))

train_dt_path = os.path.join(path, "groomed", "distance_transforms")
train_mesh_path = os.path.join(path, "groomed", "mesh")
train_particles_path = os.path.join(path, "shape_models_1024", "la_particles")

test_dt_path = os.path.join(path, "groomed_test", "distance_transforms")
test_mesh_path = os.path.join(path, "groomed_test", "mesh")
test_particles_path = os.path.join(path, "shape_models_1024_fd", "la_particles")


template_mesh = os.path.join(train_mesh_path, template+".vtk")
template_particles = os.path.join(train_particles_path, template+"_local.particles")

tqdm.write("--- converting dts to meshes ---")
train_count = convertToMesh(train_dt_path, train_mesh_path)
test_count = convertToMesh(test_dt_path, test_mesh_path)

tqdm.write("---- Calculating Metrics ----")
train_count -= 1
test_count -= 1
# out = Parallel(n_jobs=count)(delayed(calculate_PCA_analysis)(mp,particle_path,i+1) for i in tqdm(range(count), desc="Iter: "))
# calculate_PCA_analysis(train_count)
# quit()
out = [calculate_PCA_analysis(i+1) for i in tqdm(range(train_count), desc="Iter: ")]

tqdm.write("---- Saving Metrics ----")

out = np.array(out)
np.savez(os.path.join(path, "groomed_test", "stats_new.npz"), comp=out[:,0], spec=out[:,1], gen=out[:,2], gen_train=out[:,3])
tqdm.write("---- Done ----")

# shutil.rmtree(tmp_dir)