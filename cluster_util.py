import pickle
import os
import numpy as np

class Classification_cluster:
    def __init__(self,cluster_path,cluster_name,image_folder):
        self.path = str(cluster_path)
        self.img_path = str(image_folder)
        self.cluster_path = str(cluster_path + '/' + cluster_name)
        self.image_groups = []
        
        if not os.path.exists(self.path):            
            os.makedirs(self.path)
        elif os.path.exists(self.cluster_path):
            f = open(self.cluster_path, 'rb')
            self.image_groups = pickle.load(f)
            f.close()
        
    def match_face(self,embedding):
        least_distance = 99
        least_group_index = 0
        grp_no = 0

        for group in self.image_groups:
            distance = np.sum(np.square(embedding - group[0])) 
            if least_distance > distance:
                least_distance = distance
                least_group_index = grp_no 
            grp_no += 1
        return least_distance, least_group_index
      
    def add_vec(self,embedding,image=[],image_name=None,group_no=None, save_image=False):
        
        if group_no == None:
            self.image_groups.append([embedding,1,[embedding]])
            group_no = len(self.image_groups) - 1
            
        else:
            group = self.image_groups[group_no]
            group[2].append(embedding)
            group[1] += 1
            group[0] = ((group[0]*(group[1]-1)) + embedding)/group[1] 
            
        if save_image == True  and image_name!=None:
            path = self.img_path + '/' + str((group_no + 1)) + '/' + str(image_name)
            cv2.imwrite(path, image)
            
            
    
    def save_cluster(self):
        if os.path.exists(self.cluster_path):
            os.remove(self.cluster_path)
        f = open(self.cluster_path,'wb')
        pickle.dump(self.image_groups,f)
        f.close() 
        
    def get_cluster(self):
        return(self.image_groups)
    
