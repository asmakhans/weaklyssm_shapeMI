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
    for i in range(len(fileList)):
        if i == 0:
            meanShape = np.loadtxt(fileList[i])
        else:
            meanShape += np.loadtxt(fileList[i])
    meanShape = meanShape / len(fileList)
    nmMS = os.path.join(shapeModelDir, 'meanshape_local.particles')
    np.savetxt(nmMS, meanShape)

if __name__ == "__main__":
    # janmesh:
    # train_dir = "/home/sci/janmesh/Projects/original/ssm/femur_40/bcp"

    # how i did it previously
    # train_dir = "/home/sci/asmak/Documents/Methods/ssm/run_as_is/20_exp/namic/mt"

    # train_dir = "/home/sci/asmak/Documents/Methods/ssm/Ground_Truth/20_percent_gt"

    train_dir = "/home/sci/asmak/Documents/Methods/weaklyssm/MT/20_percent_femur"
    train_file_list = glob.glob(os.path.join(train_dir, "*.nii.gz"))


    spreadsheet_file_name = "la"


    # Groom
    train_groom_dir = os.path.join(train_dir, 'groomed')
    if not os.path.exists(train_groom_dir):
        os.makedirs(train_groom_dir)

        
    train_shape_seg_list = []
    train_shape_names = []
    for shape_filename in train_file_list:
        print('Loading: ' + shape_filename)
        shape_name = shape_filename.split('/')[-1].replace('.nii.gz', '')
        train_shape_names.append(shape_name)
        shape_seg = sw.Image(shape_filename)
        train_shape_seg_list.append(shape_seg)

        # do initial grooming steps
        print("Grooming: " + shape_name)
        iso_value = 0.5  # voxel value for isosurface
        bounding_box = sw.ImageUtils.boundingBox([shape_seg], iso_value).pad(2)
        shape_seg.crop(bounding_box)
        # Resample to isotropic spacing using linear interpolation
        antialias_iterations = 30   # number of iterations for antialiasing
        iso_spacing = [1, 1, 1]     # isotropic spacing
        # shape_seg.isolate()
        shape_seg.antialias(antialias_iterations).resample(iso_spacing, sw.InterpolationType.Linear).binarize()
        # Pad segmentations with zeros
        pad_size = 30    # number of voxels to pad for each dimension
        pad_value = 0   # the constant value used to pad the segmentations
        shape_seg.pad(pad_size, pad_value)
        
    ref_index = sw.find_reference_image_index(train_shape_seg_list)
    ref_name = train_shape_names[ref_index]

    ref_seg = train_shape_seg_list[ref_index].write(os.path.join(train_groom_dir, ref_name+'.nrrd'))
    ref_seg.write(os.path.join(train_groom_dir, 'reference.nrrd'))
    ## What is the reference image
    print("Reference found: " + ref_name)

    train_rigid_transforms = [] # Save rigid transorm matrices
    for shape_seg, shape_name in zip(train_shape_seg_list, train_shape_names):
        print('Finding alignment transform from ' + shape_name + ' to ' + ref_name)
        # Get rigid transform
        iso_value = 0.5      # voxel value for isosurface
        icp_iterations = 100 # number of ICP iterations
        # shape_seg.isolate()
        shape_seg.antialias(antialias_iterations)
        rigidTransform = shape_seg.createRigidRegistrationTransform(ref_seg, iso_value, 200)
        
        shape_seg.applyTransform(rigidTransform,
                                ref_seg.origin(),  ref_seg.dims(),
                                ref_seg.spacing(), ref_seg.coordsys(),
                                sw.InterpolationType.Linear)
        shape_seg.binarize()

        bounding_box = sw.ImageUtils.boundingBox([shape_seg], iso_value).pad(2)
        shape_seg.isolate()
        shape_seg.crop(bounding_box).pad(10, 0)
        

        # Convert segmentations to smooth signed distance transforms
        print("Converting " + shape_name + " to distance transform")
        iso_value = 0   # voxel value for isosurface
        sigma = 1.5     # for Gaussian blur
        
        shape_seg.antialias(antialias_iterations).computeDT(iso_value).gaussianBlur(sigma)

    # Save distance transforms
    train_groomed_files = sw.utils.save_images(os.path.join(train_groom_dir, 'distance_transforms/'), train_shape_seg_list,
                                    train_shape_names, extension='nrrd', compressed=True, verbose=True)


    # Get data input (meshes if running in mesh mode, else distance transforms)
    ## What to take for mesh mode
    domain_type, train_groomed_files = sw.data.get_optimize_input(train_groomed_files, False)

    import subprocess
    # Create project spreadsheet
    project_location = os.path.join(train_dir, "shape_models_1024")
    if not os.path.exists(project_location):
        os.makedirs(project_location)
    # Set subjects
    subjects = []
    number_domains = 1
    for i in range(len(train_file_list)):
        subject = sw.Subject()
        subject.set_number_of_domains(number_domains)
        rel_seg_files = [train_file_list[i]]#sw.utils.get_relative_paths([os.getcwd() + '/' + file_list[i]], project_location)
        rel_groom_files = [train_groomed_files[i]]
        subject.set_original_filenames(rel_groom_files)
        #sw.utils.get_relative_paths([os.getcwd() + '/' + groomed_files[i]], project_location)
        subject.set_groomed_filenames(rel_groom_files)
        transform = [ np.eye(4).flatten() ]
        subject.set_groomed_transforms(transform)
        subjects.append(subject)
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
        "optimization_iterations" : 1000,
        "starting_regularization" : 1000,
        "ending_regularization" : 1,
        "relative_weighting" : 1.0,
        "initial_relative_weighting" : 0.05,
        "save_init_splits" : 0,
        "verbosity" : 1,
        "procrustes" : 0,
        # "narrow_band": 1e20,
    }

    # Add param dictionary to spreadsheet
    for key in parameter_dictionary:
        parameters.set(key,sw.Variant([parameter_dictionary[key]]))
    parameters.set("domain_type", sw.Variant(domain_type[0]))
    project.set_parameters("optimize",parameters)
    spreadsheet_file = os.path.join(train_dir, "shape_models_1024", "la.xlsx")
    project.save(spreadsheet_file)


    # # # Run optimization
    optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
    subprocess.check_call(optimize_cmd)

    

    

    
    
