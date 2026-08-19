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


def get_particles(model_path):
    f = open(model_path, "r")
    data = []
    for line in f.readlines():
        points = line.split()
        points = [float(i) for i in points]
        data.append(points)
    return(data)

if __name__ == "__main__":
    


    train_dir = "/home/sci/janmesh/Projects/original/ssm/femur/pln"
    file_list = glob.glob(os.path.join(train_dir, "*.nii.gz"))

    # Groom
    groom_dir = os.path.join(train_dir, 'groomed')
    if not os.path.exists(groom_dir):
        os.makedirs(groom_dir)
        
    shape_seg_list = []
    shape_names = []
    for shape_filename in file_list:
        print('Loading: ' + shape_filename)
        shape_name = shape_filename.split('/')[-1].replace('.nii.gz', '')
        shape_names.append(shape_name)
        shape_seg = sw.Image(shape_filename)
        shape_seg_list.append(shape_seg)

        # do initial grooming steps
        print("Grooming: " + shape_name)
        iso_value = 0.5  # voxel value for isosurface
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
        
    ref_index = sw.find_reference_image_index(shape_seg_list)
    ref_seg = shape_seg_list[ref_index].write(os.path.join(groom_dir, 'reference.nrrd'))
    ref_name = shape_names[ref_index]
    ## What is the reference image
    print("Reference found: " + ref_name)

    """
    Now we can loop over all of the segmentations again to find the rigid
    alignment transform and compute a distance transform
    """
    rigid_transforms = [] # Save rigid transorm matrices
    for shape_seg, shape_name in zip(shape_seg_list, shape_names):
        print('Finding alignment transform from ' + shape_name + ' to ' + ref_name)
        # Get rigid transform
        iso_value = 0.5      # voxel value for isosurface
        icp_iterations = 100 # number of ICP iterations
        rigid_transform = shape_seg.createRigidRegistrationTransform(
            ref_seg, iso_value, icp_iterations)
        # Convert to vtk format for optimization
        rigid_transform = sw.utils.getVTKtransform(rigid_transform)
        rigid_transforms.append(rigid_transform)

        # Convert segmentations to smooth signed distance transforms
        print("Converting " + shape_name + " to distance transform")
        iso_value = 0   # voxel value for isosurface
        sigma = 1.5     # for Gaussian blur
        shape_seg.antialias(antialias_iterations).computeDT(iso_value).gaussianBlur(sigma)

    # Save distance transforms
    groomed_files = sw.utils.save_images(os.path.join(groom_dir, 'distance_transforms/'), shape_seg_list,
                                    shape_names, extension='nrrd', compressed=True, verbose=True)


    # Get data input (meshes if running in mesh mode, else distance transforms)
    ## What to take for mesh mode
    domain_type, groomed_files = sw.data.get_optimize_input(groomed_files, False)

    
    
    import subprocess
    # Create project spreadsheet
    project_location = os.path.join(train_dir, "shape_models_1024")
    if not os.path.exists(project_location):
        os.makedirs(project_location)
    # Set subjects
    subjects = []
    number_domains = 1
    for i in range(len(file_list)):
        subject = sw.Subject()
        subject.set_number_of_domains(number_domains)
        rel_seg_files = [file_list[i]]#sw.utils.get_relative_paths([os.getcwd() + '/' + file_list[i]], project_location)
        subject.set_original_filenames(rel_seg_files)
        rel_groom_files = [groomed_files[i]]#sw.utils.get_relative_paths([os.getcwd() + '/' + groomed_files[i]], project_location)
        subject.set_groomed_filenames(rel_groom_files)
        transform = [rigid_transforms[i].flatten()]#[ np.eye(4).flatten() ]
        subject.set_groomed_transforms(transform)
        subjects.append(subject)
    # Set project
    project = sw.Project()
    project.set_subjects(subjects)
    parameters = sw.Parameters()

    # Create a dictionary for all the parameters required by optimization

    parameter_dictionary = {
        "number_of_particles" : 1024,
        "use_normals": 1,
        "checkpointing_interval" : 200,
        "keep_checkpoints" : 0,
        "iterations_per_split" : 1000,
        "optimization_iterations" : 500,
        "starting_regularization" : 1000,
        "ending_regularization" : 10,
        "relative_weighting" : 1,
        "initial_relative_weighting" : 0.05,
        "save_init_splits" : 0,
        "geodesics_enabled": 0,
        "verbosity" : 1,
        "procrustes" : 1,
        "narrow_band": 1e20,
    }

    # Add param dictionary to spreadsheet
    for key in parameter_dictionary:
        parameters.set(key,sw.Variant([parameter_dictionary[key]]))
    parameters.set("domain_type", sw.Variant(domain_type[0]))
    project.set_parameters("optimize",parameters)
    spreadsheet_file = os.path.join(train_dir, "shape_models_1024", "la.xlsx")
    project.save(spreadsheet_file)

    # Run optimization
    optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
    subprocess.check_call(optimize_cmd)
    

    import subprocess
    # Create project spreadsheet
    project_location = os.path.join(train_dir, "shape_models_256")
    if not os.path.exists(project_location):
        os.makedirs(project_location)
    # Set subjects
    subjects = []
    number_domains = 1
    for i in range(len(file_list)):
        subject = sw.Subject()
        subject.set_number_of_domains(number_domains)
        rel_seg_files = [file_list[i]]#sw.utils.get_relative_paths([os.getcwd() + '/' + file_list[i]], project_location)
        subject.set_original_filenames(rel_seg_files)
        rel_groom_files = [groomed_files[i]]#sw.utils.get_relative_paths([os.getcwd() + '/' + groomed_files[i]], project_location)
        subject.set_groomed_filenames(rel_groom_files)
        transform = [rigid_transforms[i].flatten()]#[ np.eye(4).flatten() ]
        subject.set_groomed_transforms(transform)
        subjects.append(subject)
    # Set project
    project = sw.Project()
    project.set_subjects(subjects)
    parameters = sw.Parameters()

    # Create a dictionary for all the parameters required by optimization

    parameter_dictionary = {
        "number_of_particles" : 256,
        "use_normals": 1,
        "checkpointing_interval" : 200,
        "keep_checkpoints" : 0,
        "iterations_per_split" : 1000,
        "optimization_iterations" : 500,
        "starting_regularization" : 1000,
        "ending_regularization" : 10,
        "relative_weighting" : 1,
        "initial_relative_weighting" : 0.05,
        "save_init_splits" : 0,
        "geodesics_enabled": 0,
        "verbosity" : 1,
        "procrustes" : 1,
        "narrow_band": 1e20,
    }

    # Add param dictionary to spreadsheet
    for key in parameter_dictionary:
        parameters.set(key,sw.Variant([parameter_dictionary[key]]))
    parameters.set("domain_type", sw.Variant(domain_type[0]))
    project.set_parameters("optimize",parameters)
    spreadsheet_file = os.path.join(train_dir, "shape_models_256", "la.xlsx")
    project.save(spreadsheet_file)

    # Run optimization
    optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
    subprocess.check_call(optimize_cmd)
    

    
    
