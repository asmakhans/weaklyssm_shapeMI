import shapeworks as sw
import ShapeCohortGen
import os
import numpy as np
import argparse
import os
import glob
import math
import vtk
import numpy as np
import matplotlib.tri as mtri
import pandas as pd


def get_particles(model_path):
    f = open(model_path, "r")
    data = []
    for line in f.readlines():
        points = line.split()
        points = [float(i) for i in points]
        data.append(points)
    return(data)

def findMeanShape(fileList, shapeModelDir):
    for i in range(len(fileList)):
        if i == 0:
            meanShape = np.loadtxt(fileList[i])
        else:
            meanShape += np.loadtxt(fileList[i])
    meanShape = meanShape / len(fileList)
    nmMS = os.path.join(shapeModelDir, 'meanshape.particles')
    np.savetxt(nmMS, meanShape)

if __name__ == "__main__":
    
    # df = pd.read_excel("/home/sci/janmesh/Projects/original/ssm/la/gt/shape_models_1024/la.xlsx")
    df = pd.read_excel("/home/sci/asmak/Documents/Methods/ssm/run_as_is/20_exp/namic/mt/shape_models_1024/la.xlsx")

    # train_dir = "/home/sci/janmesh/Projects/original/ssm/la/gt"
    train_dir = "/home/sci/asmak/Documents/Methods/ssm/run_as_is/20_exp/namic/mt"

    # test_dir = "/home/sci/janmesh/Projects/original/ssm/la/test"
    test_dir = "/home/sci/asmak/Documents/Methods/ssm/run_as_is/20_exp/namic/test"

    test_file_list = glob.glob(os.path.join(test_dir, "*.nii.gz"))

    spreadsheet_file_name = "la"


    # Groom
    train_groom_dir = os.path.join(train_dir, 'groomed')

    test_groom_dir = os.path.join(test_dir, 'groomed')
    if not os.path.exists(test_groom_dir):
        os.makedirs(test_groom_dir)
        
    ref_seg = sw.Image(os.path.join(train_groom_dir, 'reference.nrrd'))
    ## What is the reference image

    train_groomed_files = df["groomed_1"].tolist()


    world_particle_list = df["world_particles_1"].tolist()
    local_particle_list = df["local_particles_1"].tolist()


    world_particle_list = sorted(world_particle_list)
    local_particle_list = sorted(local_particle_list)

    # sw.utils.findMeanShape(model_dir)
    # findMeanShape(world_particle_list, "/home/sci/janmesh/Projects/original/ssm/la/gt/shape_models_1024")
    findMeanShape(world_particle_list, "/home/sci/asmak/Documents/Methods/ssm/run_as_is/20_exp/namic/mt/shape_models_1024")
    # mean_shape_path = os.path.join("/home/sci/janmesh/Projects/original/ssm/la/gt/shape_models_1024", 'meanshape.particles')
    mean_shape_path = os.path.join("/home/sci/asmak/Documents/Methods/ssm/run_as_is/20_exp/namic/mt/shape_models_1024", 'meanshape.particles')


    test_shape_seg_list = []
    test_shape_names = []
    for shape_filename in test_file_list:
        print('Loading: ' + shape_filename)
        shape_name = shape_filename.split('/')[-1].replace('.nii.gz', '')
        test_shape_names.append(shape_name)
        shape_seg = sw.Image(shape_filename)
        test_shape_seg_list.append(shape_seg)

        # do initial grooming steps
        print("Grooming: " + shape_name)
        iso_value = 0.5  # voxel value for isosurface
        shape_seg.isolate()
        bounding_box = sw.ImageUtils.boundingBox([shape_seg], iso_value).pad(2)
        shape_seg.crop(bounding_box)
        # Resample to isotropic spacing using linear interpolation
        antialias_iterations = 30   # number of iterations for antialiasing
        iso_spacing = [1, 1, 1]     # isotropic spacing
        shape_seg.antialias(antialias_iterations).resample(iso_spacing, sw.InterpolationType.Linear).binarize()
        # Pad segmentations with zeros
        pad_size = 30    # number of voxels to pad for each dimension
        pad_value = 0   # the constant value used to pad the segmentations
        shape_seg.pad(pad_size, pad_value)


    """
    Now we can loop over all of the segmentations again to find the rigid
    alignment transform and compute a distance transform
    """


    test_rigid_transforms = [] # Save rigid transorm matrices
    for shape_seg, shape_name in zip(test_shape_seg_list, test_shape_names):
        # Get rigid transform
        iso_value = 0.5      # voxel value for isosurface
        icp_iterations = 100 # number of ICP iterations
        rigid_transform = shape_seg.createRigidRegistrationTransform(
            ref_seg, iso_value, icp_iterations)
        # Convert to vtk format for optimization
        
        shape_seg.antialias(antialias_iterations)
        
        shape_seg.applyTransform(rigid_transform,
                                ref_seg.origin(),  ref_seg.dims(),
                                ref_seg.spacing(), ref_seg.coordsys(),
                                sw.InterpolationType.Linear)
        shape_seg.binarize()

        bounding_box = sw.ImageUtils.boundingBox([shape_seg], iso_value).pad(2)
        shape_seg.crop(bounding_box).pad(10, 0)

        rigid_transform = sw.utils.getVTKtransform(rigid_transform)
        test_rigid_transforms.append(rigid_transform)

        # Convert segmentations to smooth signed distance transforms
        print("Converting " + shape_name + " to distance transform")
        iso_value = 0   # voxel value for isosurface
        sigma = 1.5     # for Gaussian blur
        shape_seg.antialias(antialias_iterations).computeDT(iso_value).gaussianBlur(sigma)

    # Save distance transforms
    test_groomed_files = sw.utils.save_images(os.path.join(test_groom_dir, 'distance_transforms/'), test_shape_seg_list,
                                    test_shape_names, extension='nrrd', compressed=True, verbose=True)




    subjects = []
    number_domains = 1

    for i in range(len(train_file_list)):
        subject = sw.Subject()
        subject.set_number_of_domains(number_domains)
        rel_seg_files = [train_file_list[i]]#sw.utils.get_relative_paths([os.getcwd() + '/' + file_list[i]], project_location)
        # subject.set_original_filenames(rel_seg_files)
        rel_groom_files = [train_groomed_files[i]]#sw.utils.get_relative_paths([os.getcwd() + '/' + groomed_files[i]], project_location)
        subject.set_groomed_filenames(rel_groom_files)
        transform = [ np.eye(4).flatten() ]
        # subject.set_groomed_transforms(transform)
        subject.set_local_particle_filenames([local_particle_list[i]])
        subject.set_world_particle_filenames([local_particle_list[i]])
        subject.set_extra_values({"fixed": "yes"})
        subjects.append(subject)

    for i in range(len(test_file_list)):
        subject = sw.Subject()
        subject.set_number_of_domains(number_domains)
        rel_seg_files = [test_file_list[i]]#sw.utils.get_relative_paths([os.getcwd() + '/' + file_list[i]], project_location)
        # subject.set_original_filenames(rel_seg_files)
        rel_groom_files = [test_groomed_files[i]]#sw.utils.get_relative_paths([os.getcwd() + '/' + groomed_files[i]], project_location)
        subject.set_groomed_filenames(rel_groom_files)
        transform = [test_rigid_transforms[i].flatten()]#[ np.eye(4).flatten() ]
        # subject.set_groomed_transforms(transform)
        subject.set_local_particle_filenames([mean_shape_path])
        subject.set_world_particle_filenames([mean_shape_path])
        subject.set_extra_values({"fixed": "no"})
        subjects.append(subject)

        
    
    import subprocess
    # Create project spreadsheet
    project_location = os.path.join(train_dir, "shape_models_1024_fd")
    if not os.path.exists(project_location):
        os.makedirs(project_location)

    # Set project
    project = sw.Project()
    project.set_subjects(subjects)
    parameters = sw.Parameters()

    # Create a dictionary for all the parameters required by optimization

    parameter_dictionary = {
        "number_of_particles" : 1024,
        "use_normals": 0,
        "checkpointing_interval" : 200,
        "keep_checkpoints" : 0,
        "iterations_per_split" : 1000,
        "optimization_iterations" : 500,
        "starting_regularization" : 1000,
        "ending_regularization" : 10,
        "recompute_regularization_interval" : 2,
        "domains_per_shape" : 1,
        "relative_weighting" : 1,
        "initial_relative_weighting" : 0.05,
        "save_init_splits" : 0,
        "verbosity" : 1,
        "procrustes" : 1,
        "use_fixed_subjects": 1,
        "narrow_band": 1e50,
    }

    parameter_dictionary['use_normals'] = 0
    parameter_dictionary['verbosity'] = 1
    parameter_dictionary['narrow_band'] = 1e50

    # Add param dictionary to spreadsheet
    for key in parameter_dictionary:
        parameters.set(key,sw.Variant([parameter_dictionary[key]]))
    parameters.set("domain_type", sw.Variant(1))
    project.set_parameters("optimize",parameters)
    spreadsheet_file = os.path.join(project_location, "la.xlsx")
    project.save(spreadsheet_file)

    analyze_cmd = ('ShapeWorksStudio ' + spreadsheet_file).split()
    subprocess.check_call(analyze_cmd)

    # # Run optimization
    # optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
    # subprocess.check_call(optimize_cmd)
    

    

    
    
