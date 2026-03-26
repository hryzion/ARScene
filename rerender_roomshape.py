import json
import tqdm
import os
from config import DATA_DIR, OBJ_DIR
from utils import load_scene_json, divide_scene_json_to_rooms, get_room_attributes, CRASHED_ROOM, render_room_shape_image_fixed_scale, centralize_room
import random
import numpy as np
TARGET_DIR = './datasets/processed'
import math


def rerender_roomshape(target_dir, filter = '', type_filter = "bedroom"):
    target_dir = os.path.join(target_dir, filter)
    for room_name in tqdm.tqdm(os.listdir(target_dir)):
        # print(room_name)
        scene_origin = room_name.split('_')[0]
        room_id = int(room_name.split('_')[1].split('room')[-1])
        scene_json = load_scene_json(os.path.join(DATA_DIR, scene_origin+'.json'))
        rooms = divide_scene_json_to_rooms(scene_json)
        room = rooms[room_id]
        # print(room['roomShape'])
        centralize_room(room)
        if type_filter == 'bedroom':
            fixed_size = 3.5
        elif type_filter == 'livingroom' or type_filter == 'diningroom':
            fixed_size = 5
        elif type_filter == 'library':
            fixed_size = 3.5
        else:
            fixed_size=10
        scaled_roomshape_rendered = render_room_shape_image_fixed_scale(room, fixed_size=fixed_size)
        scaled_roomshape_rendered.save(os.path.join(target_dir, room_name, 'room_mask.png'))
        # print(room['roomShape'])
        # room_shape = np.array(room['roomShape'])
        # boxes = calc_box_from_polygon(room_shape)
        # v, f = polygon_to_mesh(room_shape)
        # np.savez_compressed(
        #     os.path.join(target_dir, room_name, 'room_shape.npz'),
        #     room_shape = room_shape,
        #     boxes = boxes,
        #     vertices = v,
        #     faces = f
        # )


def calc_box_from_polygon( points, S=50):
    l = len(points)
    boxes = {"translation":[],"size":[]}
    for i in range(l):
        # i = 6
        # if i==6:
        #     continue
        x = points[i]
        j = (i+1)%l
        k = (i+2)%l
        vij = np.array(points[j]-points[i])
        vjk = np.array(points[k]-points[j])
        # x->y, x->z
        vxz = np.array([-vij[1],vij[0]])
        # z = x + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*S
        

        #####calc y
        C = x
        D = x + vij/math.sqrt(vij[0]*vij[0]+vij[1]*vij[1])*S  #y
        line2 = D-C
        intersection_lst = []
        for t in range(1,l-1):
            A = points[(i+t)%l]
            B = points[(i+t+1)%l]
            line1 = B-A
            if is_perpendicular(line1,line2) and np.cross(line2,line1)>0: #chuizhi & left 
                intersection = segments_intersect(A,B,C,D) #qiu xianduan jiaodian
                if intersection:
                    intersection_lst.append(np.array([intersection[0],intersection[1]]))

        #find closest intersction
        y = D.copy()
        if len(intersection_lst)>0:
            min_dist = S*S
            for intersection in intersection_lst:
                d = dist2(intersection,x)
                if d<min_dist:
                    y = intersection.copy()
                    min_dist = d
        # cross = np.cross(vij,vjk)
        # if cross > 0:  #-| limit
        #     y = points[j]
        # else: # -- or -|
        #     y = x + vij/math.sqrt(vij[0]*vij[0]+vij[1]*vij[1])*S

        #####calc z
        C = x
        Dxz = x + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*S
        Dyz = y + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*S

        line2 = Dxz-C
        intersection_lst = []
        dmin2 = S*S
        for t in range(1,l-1):
            A = points[(i+t)%l]
            B = points[(i+t+1)%l]
            line1 = B-A
            if is_perpendicular(line1,line2) and np.cross(line2,line1)>0: #chuizhi & left
                #xz
                intersection = segments_intersect(A,B,x,Dxz) #qiu xianduan jiaodian
                if intersection:
                    dmin2 = min(dmin2,dist2(intersection,x))
                #yz
                intersection = segments_intersect(A,B,y,Dyz) #qiu xianduan jiaodian
                if intersection:
                    dmin2 = min(dmin2,dist2(intersection,y))
                    

        #find closest intersction
        d = np.sqrt(dmin2)
        z = x + vxz/math.sqrt(vxz[0]*vxz[0]+vxz[1]*vxz[1])*d

        # cross = np.cross(vij,vjk)
        # if cross > 0:  #-| limit
        #     y = points[j]
        # else: # -- or -|
        #     y = x + vij/math.sqrt(vij[0]*vij[0]+vij[1]*vij[1])*S

        sizez = S
        meanz = 0
        if x[0]==y[0]:
            sizey = abs(x[1]-y[1])
            sizex = abs(x[0]-z[0])
            meany = (x[1]+y[1])/2
            meanx = (x[0]+z[0])/2
        else:
            sizex = abs(x[0]-y[0])
            sizey = abs(x[1]-z[1])
            meany = (x[1]+z[1])/2
            meanx = (x[0]+y[0])/2

        
        translation = np.array([meanx,meany,meanz])
        size = np.array([sizex,sizey,sizez])# half size
        boxes["translation"].append(translation)
        boxes["size"].append(size)

    boxes["translation"] = np.array(boxes["translation"])
    boxes["size"] = np.array(boxes["size"])
    # 
    gt_boxes = [[boxes["translation"][i][0],boxes["translation"][i][1],boxes["translation"][i][2],boxes["size"][i][0],boxes["size"][i][1],boxes["size"][i][2],0] for i in range(len(boxes["translation"]))]
    gt_boxes = np.array(gt_boxes)
    # print(gt_boxes)
    # vis = draw_box_label(gt_boxes, (0, 0, 1))

    return  boxes

def dist2(A,B):
    return (A[0]-B[0])*(A[0]-B[0])+(A[1]-B[1])*(A[1]-B[1])


def is_perpendicular(line1,line2):
    #k
    # slope1 = line1[1]/ line1[0]
    # slope2 = line2[1] / line2[0]

    if line1[1]*line2[1]==-line1[0]*line2[0]:
        return True
    else:
        return False
def segments_intersect(A,B,C,D):
    #check bbox 
    if max(A[0],B[0])<min(C[0],D[0]) or max(C[0],D[0])<min(A[0],B[0]) or \
        max(A[1],B[1])<min(C[1],D[1]) or max(C[1],D[1])<min(A[1],B[1]):
        return False
    
    #calc cross product
    def cross_product(p1,p2,p3):
        out = (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0])
        return out
    
    #chech whether two segs are crossed
    if cross_product(A,B,C) * cross_product(A,B,D) <= 0 and \
        cross_product(C,D,A) * cross_product(C,D,B) <= 0:
        a = cross_product(C,D,A)
        b = cross_product(C,D,B)
        t = a/(a-b+0.0000001)
        x = A[0] + t*(B[0]-A[0])
        y = A[1] + t*(B[1]-A[1])
        # if (Room.dist(A,(x,y))<0.0001) or (Room.dist(B,(x,y))<0.0001):
        #     return False
        return (x,y)
    else:
            return False

def polygon_to_mesh(points, H=3.0):
    vertices = []
    
    # bottom
    for x, z in points:
        vertices.append([x, 0, z])
        
    # top
    for x, z in points:
        vertices.append([x, H, z])

    faces = []
    n = len(points)

    # side faces
    for i in range(n):
        j = (i+1)%n
        
        # quad → 2 triangles
        faces.append([i, j, j+n])
        faces.append([i, j+n, i+n])

    return np.array(vertices), np.array(faces)
    


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_objlat', action="store_true",help='Experiment number for visualization folder')
    parser.add_argument('--filter_fn', type=str,default='', help="room type for training dataset")
    parser.add_argument('--data_filter', type=str, default='train')
    parser.add_argument('--out_dir', type=str, default='./datasets/processed', help = "output dir")
    args = parser.parse_args()

    target_dir = args.out_dir
    # if args.atiss:
    #     target_dir = './datasets/atiss'
    if args.filter_fn != '':
        target_dir+=f'_{args.filter_fn}'
    if not args.use_objlat:
        target_dir += "_wo_lat"
    
    rerender_roomshape(target_dir=target_dir,  filter=args.data_filter, type_filter = "bedroom")


    