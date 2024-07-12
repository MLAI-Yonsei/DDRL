from torch import nn
import torch
from torchvision.utils import save_image
import numpy as np
import cv2
from copy import deepcopy
import random

class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, img_size, return_box=False, show_position=False, point_radius=1, point_color=1, point_circle=True, line_thickness=2, line_color=0.5, box_color=0.75, max_points=100):
        self.data_file = data_file
        self.img_size = img_size
        self.return_box = return_box
        self.point_radius = point_radius
        self.point_color = point_color
        self.point_circle = point_circle
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.box_color = box_color
        self.max_points = max_points
        self.show_position = show_position
        
        self.file_lines = open(data_file).read().splitlines()
        print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')
        
    def __len__(self):
        return len(self.file_lines)
    
    def rasterize(self, idx, return_img = True):
        # Select sample
        line = self.file_lines[idx]
        # Clear leading/trailing characters
        line = line.strip()

        # Extract points
        points = line.split(' output ')[0]
        points = points.split(' ')
        points = np.array([[float(points[i]), float(points[i+1])] for i in range(0,len(points),2)])
        # Extract tour
        tour = line.split(' output ')[1]
        tour = tour.split(' ')
        tour = np.array([int(t) for t in tour])
        if len(line.split(' output '))>2:
            # Extract box
            box = line.split(' output ')[2]
            box = box.split(' ')
            box = np.array([float(t) for t in box])
        
        if self.return_box:
            img = self.draw_tour(tour=tour, points=points, box=box)
            return img, points, tour, box
        
        if return_img:
            img = self.draw_tour(tour=tour, points=points)
            return img, points, tour
        # # Rasterize lines
        # img = np.zeros((self.img_size, self.img_size))
        # for i in range(tour.shape[0]-1):
        #     from_idx = tour[i]-1
        #     to_idx = tour[i+1]-1

        #     cv2.line(img, 
        #              tuple(((self.img_size-1)*points[from_idx,::-1]).astype(int)), 
        #              tuple(((self.img_size-1)*points[to_idx,::-1]).astype(int)), 
        #              color=self.line_color, thickness=self.line_thickness)

        # # Rasterize points
        # for i in range(points.shape[0]):
        #     if self.point_circle:
        #         cv2.circle(img, tuple(((self.img_size-1)*points[i,::-1]).astype(int)), 
        #                    radius=self.point_radius, color=self.point_color, thickness=-1)
        #     else:
        #         row = round((self.img_size-1)*points[i,0])
        #         col = round((self.img_size-1)*points[i,1])
        #         img[row,col] = self.point_color
            
        # # Rescale image to [-1,1]
        # img = 2*(img-0.5)
        return points, tour

    def draw_tour(self, tour, points, box = None, edges = None):
        img = np.zeros((self.img_size, self.img_size))
        # Rasterize lines
        for i in range(tour.shape[0]-1):
            if tour.min()==1:
                from_idx = tour[i]-1
                to_idx = tour[i+1]-1
            elif tour.min()==0:
                from_idx = tour[i]
                to_idx = tour[i+1]
            cv2.line(img, 
                        tuple(((self.img_size-1)*points[from_idx,::-1]).astype(int)), 
                        tuple(((self.img_size-1)*points[to_idx,::-1]).astype(int)), 
                        color=self.line_color, thickness=self.line_thickness)

        # Rasterize points
        for i in range(points.shape[0]):
            point_coords = tuple(((self.img_size-1)*points[i,::-1]).astype(int))
            if self.point_circle:
                cv2.circle(img, point_coords, 
                            radius=self.point_radius, color=self.point_color, thickness=-1)
            else:
                row = round((self.img_size-1)*points[i,0])
                col = round((self.img_size-1)*points[i,1])
                img[row,col] = self.point_color
                
                    # Conditionally add text to image
            if self.show_position:
                text = f'({points[i, 1]:.2f}, {points[i, 0]:.2f})'
                cv2.putText(img, text, (point_coords[0] + 5, point_coords[1] - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, self.point_color, 1, cv2.LINE_AA)
        
        # Rasterize box
        if box is not None:
            y_bottom = int(box[0] * (self.img_size - 1))
            y_top = int(box[1] * (self.img_size - 1))
            x_left = int(box[2] * (self.img_size - 1))
            x_right = int(box[3] * (self.img_size - 1))
            img[y_bottom:y_top, x_left:x_right] = self.box_color

            if self.show_position:
                corners = [(x_left, y_bottom), (x_right, y_bottom), (x_left, y_top), (x_right, y_top)]
                for (x, y) in corners:
                    text = f'({x/(self.img_size-1):.2f}, {y/(self.img_size-1):.2f})'
                    cv2.putText(img, text, (x + 5, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.point_color, 1, cv2.LINE_AA)

        if edges is not None:
            for edge in edges:
                from_idx, to_idx = edge
                cv2.line(img, 
                         tuple(((self.img_size - 1) * points[from_idx, ::-1]).astype(int)), 
                         tuple(((self.img_size - 1) * points[to_idx, ::-1]).astype(int)), 
                         self.box_color, thickness=self.line_thickness)  # Different color for added lines

        
        
        # Rescale image to [-1,1]
        img = 2*(img-0.5)
        return img


    def __getitem__(self, idx):
        if self.return_box:
            img, points, tour, box = self.rasterize(idx)
            return img[np.newaxis,:,:], points, tour, idx, box
        else:
            img, points, tour = self.rasterize(idx)
            return img[np.newaxis,:,:], points, tour, idx
            

class Model_x0(nn.Module):
    def __init__(self, batch_size, num_points, img_size, line_color, line_thickness, xT):
        super(Model_x0, self).__init__()
        
        # Latent variables (b,v,v) matrix
        self.latent = nn.Parameter(torch.randn(batch_size, num_points, num_points)) # (B, 50, 50)
        self.latent.requires_grad = True
        
        self.num_points = num_points
        self.batch_size = batch_size
        self.img_size = img_size
        self.line_color = line_color
        self.line_thickness = line_thickness
        self.xT = xT

    def reset(self):
        nn.init.normal_(self.latent)
        # self.latent = nn.Parameter(torch.randn(self.batch_size, self.num_points, self.num_points)) # (B, 50, 50)
        # self.latent.requires_grad = True

    def compute_edge_images(self, points, img_query):
        # Pre-compute edge images
        self.img_query = img_query
        self.edge_images = []
        for i in range(points.shape[0]):
            node_edges = []
            for j in range(points.shape[0]):
                edge_img = np.zeros((self.img_size, self.img_size)) # (64, 64)
                cv2.line(edge_img, 
                         tuple(((self.img_size-1)*points[i,::-1]).astype(int)), # city position in 50x50 ex) (2, 39)
                         tuple(((self.img_size-1)*points[j,::-1]).astype(int)), 
                         color=self.line_color, thickness=self.line_thickness)
                edge_img = torch.from_numpy(edge_img).float().to(self.latent.device)

                node_edges.append(edge_img)
            node_edges = torch.stack(node_edges, dim=0)
            self.edge_images.append(node_edges)
        self.edge_images = torch.stack(self.edge_images, dim=0) # (50, 50, 64, 64) -> all edge connection image for each city
                        
    def encode(self, sampling=False):
        # Compute permutation matrix
        adj_mat = normalize(self.latent) # [1, 50, 50] -> [1, 50, 50]
        if sampling:
            adj_mat = normalize(deepcopy(self.latent)) #TODO: need to apply random
        adj_mat_ = adj_mat
        all_edges = self.edge_images.view(1,-1,self.img_size,self.img_size).to(adj_mat.device)
        img = all_edges * adj_mat_.view(self.batch_size,-1,1,1) # [1, 2500, 64, 64] * [1, 50, 50] -> [1, 2500, 64, 64]
        img = torch.sum(img, dim=1, keepdims=True) # [1, 2500, 64, 64] -> [1, 1, 64, 64]
        
        img = 2*(img-0.5)               
        
        # Draw fixed points
        img[self.img_query.tile(self.batch_size,1,1,1) == 1] = 1
        
        return img
    
    def save_image(self, path):
        model_encode = deepcopy(torch.clamp(self.encode(), -1, 1).cpu().detach())
        model_encode -= model_encode.min()
        model_encode /= model_encode.max()
        save_image(model_encode[0,0,:,:], path)
        
        
    
def normalize(cost, entropy_reg=0.1, n_iters=20, eps=1e-6):
    # Cost matrix is exp(-lambda*C)
    cost_matrix = -entropy_reg * cost # 0.1 * [1, 50, 50] (latent)
        
    cost_matrix -= torch.eye(cost_matrix.shape[-1], device=cost_matrix.device)*100000 # COST = COST - 100000*I
    cost_matrix = cost_matrix - torch.logsumexp(cost_matrix, dim=-1, keepdim=True)
    assignment_mat = torch.exp(cost_matrix)
    
    return assignment_mat # [1, 50, 50] (adj_mat)