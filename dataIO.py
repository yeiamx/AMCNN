import trimesh
import os
import os.path
import numpy as np

chair_file_path = 'E:/ML/Data/chair/chair_aligned_obj_obb_200'
#bicycle_file_path = 'E:/ML/Data/Grass data/bicycle_aligned_obj_obb_100'
#candel_file_path = 'E:/ML/Data/Grass data/candel_aligned_obj_obb'
#excavator_file_path = 'E:/ML/Data/Grass data/excavator_aligned_obj_obb_61'
#plane_file_path = 'E:/ML/Data/Grass data/plane_aligned_obj_obb'
#airplane_file_path1='E:/ML/Data/airplane/test'
#airplane_file_path2='E:/ML/Data/airplane/train'
bathtub_file_path1='E:/ML/Data/bathtub/train'
segmentation_mesh_path='E:/ML/Data/SegData(small)/mesh'
segmentation_seg_path='E:/ML/Data/SegData(small)/seg'
#bathtub_file_path2='E:/ML/Data/bathtub/test'

seg_data_index = ['52', '125', '200', '341', '342', '343', '398']

def meshToAdjacencyMatrix(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    num_vertices = len(vertices)
    num_faces = len(faces)

    contain_list = [[] for i in range(num_vertices)]
    for i in range(num_faces):
        contain_list[faces[i][0]].append(faces[i][1])
        contain_list[faces[i][0]].append(faces[i][2])
        contain_list[faces[i][1]].append(faces[i][0])
        contain_list[faces[i][1]].append(faces[i][2])
        contain_list[faces[i][2]].append(faces[i][0])
        contain_list[faces[i][2]].append(faces[i][1])

    adjancencyMatrix = np.zeros((num_vertices, num_vertices))
    for i in range(num_vertices):
        for j in range(num_vertices):
            if j in contain_list[i]:
                adjancencyMatrix[i][j] = np.linalg.norm(vertices[i]-vertices[j])
    return adjancencyMatrix

def meshToFeatures(mesh):
    return mesh.vertices

def list_mesh_to_features(path, format, limit=2500):
    less_than_thousand_count = 0
    path_list = os.listdir(path)
    index = 0

    for filename in path_list:
        index+=1
        print(str(index)+'/'+str(len(path_list)))

        if os.path.splitext(filename)[1] == format:
            try:
                mesh = trimesh.load(path+'/'+filename)
            except UnicodeDecodeError:
                print("code format Error:"+filename)
            if (len(mesh.vertices) <= limit):
                #print(path+'/'+filename)
                less_than_thousand_count+=1
                print('processing...')
                features = meshToFeatures(mesh)
                np.save(path+'/'+filename.split('.')[0]+"_feature.npy", features)
                #print(less_than_thousand_count)
            else:
                pass
                #print(len(mesh.vertices))
    print('total small '+format+' file: ', less_than_thousand_count)

""" origin seg format: N x 1
    label format: N x C """

def seg_to_label(path, format='seg'):
    print('converting seg to labels...')

    labels = []
    for seg_index in seg_data_index:
        filename = path + '/' + seg_index + '/' + seg_index + '_0.seg'
        print(filename)
        with open(filename,'r') as f:
            result = []
            for line in f.readlines():
                result.append(int(line.strip('\n').strip('\r')))
            label_array = np.array(result)
            class_num = np.max(label_array) + 1
            one_hot_label_array = np.eye(class_num)[label_array]
            labels.append(one_hot_label_array)
    return labels

def list_mesh_to_am(path, format, limit=2500):
    less_than_thousand_count = 0
    path_list = os.listdir(path)
    index = 0

    for filename in path_list:
        index+=1
        print(str(index)+'/'+str(len(path_list)))

        if os.path.splitext(filename)[1] == format:
            try:
                mesh = trimesh.load(path+'/'+filename)
            except UnicodeDecodeError:
                print("code format Error:"+filename)
            if (len(mesh.vertices) <= limit):
                #print(path+'/'+filename)
                less_than_thousand_count+=1
                print('processing...')
                adjacencyMatrix = meshToAdjacencyMatrix(mesh)
                np.save(path+'/'+filename.split('.')[0]+"_am.npy", adjacencyMatrix)
                #print(less_than_thousand_count)
            else:
                pass
                #print(len(mesh.vertices))
    print('total small '+format+' file: ', less_than_thousand_count)

#be carefully before open below
#list_mesh_to_am(airplane_file_path1, '.off', 2500)
#list_mesh_to_am(airplane_file_path2, '.off', 2500)
#list_mesh_to_am(bathtub_file_path1, '.off', 2500)
#list_mesh_to_am(bathtub_file_path2, '.off', 2500)
#list_mesh_to_am(plane_file_path, ".obj")
#list_mesh_to_am(chair_file_path, '.obj', 2500)


#list_mesh_to_features(bathtub_file_path1, '.off', 2500)
#list_mesh_to_features(chair_file_path, '.obj', 2500)

#list_mesh_to_am(segmentation_mesh_path, '.off', 5000)
#list_mesh_to_features(segmentation_mesh_path, '.off', 5000)

#list_mesh_to_features('data',  '.obj')
#print(np.load('data/toydata_feature.npy').shape)
