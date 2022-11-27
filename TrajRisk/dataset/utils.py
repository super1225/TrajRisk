import math
import sys
from time import time
import numpy as np
import time, datetime

from torch import norm


def clip(n, min_value, max_value):
    return min(max(n, min_value), max_value)


def to_pixel_x(longitude, zoom):
    lon = clip(longitude, -180.0, 180.0)
    x = (lon + 180.0) / 360.0
    map_size = 256 << zoom
    return int(clip(x * map_size, 0, map_size - 1))


def to_tile_x(longitude, zoom):
    return int(to_pixel_x(longitude, zoom) / 256)


def to_pixel_y(latitude, zoom):
    lat = clip(latitude, -85.05112878, 85.05112878)
    sin_lat = math.sin(lat * math.pi / 180.0)
    y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4.0 * math.pi)
    map_size = 256 << zoom
    return int(clip(y * map_size, 0, map_size - 1))


def to_tile_y(latitude, zoom):
    return int(to_pixel_y(latitude, zoom) / 256)


def get_tile_index(longtitude, latitude, zoom=17):
    assert 5 <= zoom <= 20, "zoom should in range [5, 20]"
    return to_tile_x(longtitude, zoom), to_tile_y(latitude, zoom)


def time_process(time):
    dateArray = datetime.datetime.fromtimestamp(time)
    otherStyleTime = dateArray.strftime("%H")
    return int(otherStyleTime)
            
      
def to_grids(max_x, max_y, x, y):
        return 300*int(300*x/max_x)+int(300*y/max_y)

def norm(line,max_value,min_value):
    new_line = [0]*len(line)
    max_value=max_value
    min_value=min_value
    for i in range(0,len(line)):
        if(line[i] < min_value[i]):
                new_line[i] = min_value[i]
        elif(line[i] > max_value[i]):
                new_line[i] = max_value[i]
        else:
            new_line[i]=(line[i]-min_value[i])/(max_value[i]-min_value[i])  
    return new_line 

def line_preprocess_traj(config, line):
    new_line = []
    norm_line = []
    max_value=[config["time_num"]+2,config["dis_max"]+2,config["avg_speed_max"]+2]
    min_value=[0,config["dis_min"],config["avg_speed_min"]]
    for point in line:
        time = time_process(point["begin_time"])
        dist = point["route"] if point["route"] < config["dis_max"] else config["dis_max"]  
        avg_speed = point["avg_speed"] if point["avg_speed"] < config["avg_speed_max"] else config["avg_speed_max"] 
        feature_select = [time,dist,avg_speed]
        norm_line= norm(feature_select,max_value,min_value)
        if(len(norm_line)>0):
            new_line.append(norm_line)
        else:
            continue 
    return new_line
            
def dict_2_list(dictlist):
    result = []
    for i in range(0,len(dictlist)):
        time = dictlist[i]["begin_time"]
        result.append([dictlist[i]["route"],dictlist[i]["route"]/dictlist[i]["num_points"], dictlist[i]["max_speed"]-dictlist[i]["min_speed"],dictlist[i]["min_speed"],dictlist[i]["avg_angle"],time_process(time)])
    
    return result

def staticinfo_norm(config,line):
    new_line = []
    norm_line = []
    max=[0,0,config["route_max"],config["offset_max"],config["max_speed_max"],config["min_speed_max"],config["avg_speed_max"],config["max_r_max"],config["min_r_max"],config["avg_r_max"]]
    min=[0,0,config["route_min"],config["offset_min"],config["max_speed_min"],config["min_speed_min"],config["avg_speed_min"],config["max_r_min"],config["min_r_min"],config["avg_r_min"]]
    for point in line:
        norm_line= norm(point,max,min)
        if(len(norm_line)>0):
            new_line.append([norm_line[2], norm_line[3], norm_line[4],norm_line[5], norm_line[6], norm_line[7],norm_line[8],norm_line[9]]) 
        else:
            continue 
    
    return new_line
    


def event_norm(line):
    new_line = [0,0,0,0,0,0,0,0,0]
    max=[600,809500,600,11000,40000000,20000000,500,700000,500]
    min=[0,0,0,0,0,0,0,0,0]
    for i in range(0,len(line)):
        if(line[i] < min[i]):
                new_line[i]=min[i]
        elif(line[i] > max[i]):
                new_line[i]=max[i]
        else:
            new_line[i]=(line[i]-min[i])/(max[i]-min[i])  
    return new_line  
    
def line_preprocess(line, distance_num, last_num, zoom=17):
    cur_begin = line[0][0]
    cur_end = line[0][0]
    old_begin = line[0][0]
    cur_x, cur_y = get_tile_index(line[0][1], line[0][2], zoom)
    new_line = []

    cur_speed = []
    cur_direction = []

    for point in line:
        x, y = get_tile_index(point[1], point[2], zoom)
        cur_speed.append(point[3])
        if x==cur_x and y==cur_y:
            cur_end = point[0]
        else:
            cur_speed = np.array(cur_speed)
            new_line.append([cur_x, cur_y, time_distance_to_grid(cur_begin-old_begin, distance_num
                ), time_distance_to_grid(cur_end-cur_begin, last_num), cur_speed.max(), cur_speed.min(
                ), cur_speed.mean()])
            cur_speed = []
            cur_x, cur_y = x, y
            old_begin = cur_begin
            cur_begin = point[0]
            cur_end = point[0]
    return new_line

def time_distance_to_grid(time_distance, grid_num):
    threshold_1 = grid_num-100
    threshold_2 = 99
    if time_distance<=threshold_1:
        return time_distance
    elif time_distance>threshold_1 and time_distance<threshold_1+10*threshold_2+9:
        return int((time_distance-threshold_1)/10) + threshold_1
    else:
        return grid_num

if __name__ == '__main__':
    time_process(1641487856, 10)