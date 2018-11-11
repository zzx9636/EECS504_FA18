import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage import color
from skimage.measure import regionprops
import eta.core.image as etai
from collections import deque


def display_save(img, S, img_tag):
    '''
    Function to display superpixels and its overlay on img
    Args:
    -----
    img: input image
    S: Super pixel mask. Each pixel location will have the superpixel label corresponding to it
    img_tag: number to represent the question number for saving figures
    '''
    fig = plt.figure()
    plt.subplot(131)
    plt.imshow(img)
    plt.axis('off')
    plt.title("input image")

    plt.subplot(132)
    plt.imshow(S,cmap='hsv')
    
    plt.axis('off')
    plt.title("superpixel mask")
    plt.subplot(133)
    
    out2 = color.label2rgb(S, img)
    plt.imshow(out2)
    plt.axis('off')
    plt.title("superpixel overlay")
    #plt.show()
    plt.savefig(img_tag + '_Superpixels..png')
    plt.close()


def get_superpixel(img, segments):
    '''
    Function to get superpixels and its centroid locations
    Args:
    -----
    img: input image
    segments: parameter for SLIC function
    Return:
    -------
    S: superpixel mask of size of the size of image. 
       Each pixel location will have the superpixel label corresponding to it
    C: List of coordinates of centroid of superpixels
    '''
    S = slic(img,n_segments= segments)
    A = S + 1
    regions= regionprops(A)
    centroids  = []
    for props in regions:
        centroids.append(props.centroid)
    return S,centroids

def get_correspondences(img1):
    # Function to pick key superpixel
    pts = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    coords = []
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print("The current point is: ")
        print (ix, iy)

        coords.append((ix, iy))

        if len(coords) == 1:
            fig.canvas.mpl_disconnect(cid)
            plt.close()
        return coords

    ax.imshow(img1)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    pts = coords
    return pts

def img_reduce(img, S, C):
    '''
    Wrapper for function to compute feature descriptors on the segments
    '''
    segments  = len(C)
    nbins = 10
    segment_val = np.zeros((segments,nbins*img.shape[-1]))
    for i in range(segments):
        segment_val[i,:] = histvec(img, S ==i ,nbins)
    return segment_val

def bfs_augment_path(start, target, current_flow, capacity, n):

    WHITE = 0
    GRAY = 1
    BLACK = 2
    color = [WHITE for i in range(n)]
    q = deque([])
    augment_path = []
    q.append(start)
    color[start] = GRAY

    pred = np.zeros((1,n))
    while len(q) != 0:
        u = q.popleft()
        color[u] = BLACK

        for v in range(n):
            if (color[v] == WHITE and capacity[u,v] > current_flow[u,v]):
                q.append(v)
                color[v] = GRAY
                pred[0][v] = u

    if color[target] == BLACK:
        temp = target
        while (pred[0][int(temp)] != start):
            augment_path = [pred[0][int(temp)]] + augment_path
            temp = pred[0][int(temp)]

        augment_path = [start] + augment_path + [target]
    else:
        augment_path = []

    return augment_path

def bfs_residual_reachable(start, current_flow, capacity):

    num_node = capacity.shape[0]-2
    q = deque([])
    q.append(start)
    node_hist = np.zeros(num_node,dtype=bool)
    visited_node = []

    while len(q) != 0:
        u = q.popleft()
        for i in range(num_node):
            if~node_hist[i]:
                if current_flow[u,i]<capacity[u,i]:
                    node_hist[i]=True
                    visited_node.append(i)
                    q.append(i) 
    return  visited_node   


def ff_max_flow(source, sink, capacity, nodes_number):
    # Function to implement Ford-Fulkerson Algorithm
    current_flow = np.zeros((nodes_number, nodes_number))
    max_flow = 0
    augment_path = bfs_augment_path(source, sink, current_flow, capacity,
                                    nodes_number)
    while len(augment_path) != 0:
        increment = float("inf")
        for i in range(len(augment_path) - 1):
            pos_values = [increment,
                          (capacity[int(augment_path[i]), int(augment_path[i+1])] -
                          current_flow[int(augment_path[i]), int(augment_path[i+1])])]
            increment = min(pos_values)
        for i in range(len(augment_path) - 1):
            current_flow[int(augment_path[i]), int(augment_path[i+1])] += increment
            current_flow[int(augment_path[i+1]), int(augment_path[i])] -= increment

        max_flow += increment

        augment_path = bfs_augment_path(
            source, sink, current_flow, capacity, nodes_number)

    return (max_flow, current_flow)


def histvec(img,mask,b):
    '''
    Function to find the color histogram of the image.

    Args:
    -----
    img: input image
    mask: Super pixel mask. Each pixel location will have the superpixel label corresponding to it
    b: number of bins in the histogram
    Return:
    -------
    hist_vector: 1-D vector having the histogram of all three channels appended
    '''
    '''
        For each channel in the image, compute a b-bin histogram (uniformly space
    bins in the range 0:255) of the pixels in image where the mask is true. 
    Then, concatenate the vectors together into one column vector (first
    channel at top).
    
    normalize the histogram of each channel so that it sums to 1.
    Function to take a superpixel set and a keyindex and convert to a 
%  foreground/background segmentation.
    You CAN use the cv2 functions.
    You MAY loop over the channels.
    
    '''
    img_in_SP = img[mask,:].astype(dtype=np.float)
    total_location = img_in_SP.shape[0]
    hist_vector = np.zeros(3*b)
    ub_unit = 255.0/b
    for i in range(b):
        ub_cur = np.ceil((i+1)*ub_unit)
        for j in range(3):
            cur_idx = np.argwhere(img_in_SP[:,j]<=ub_cur)
            hist_vector[j*b+i]+=len(cur_idx)
            img_in_SP[cur_idx,j] = 300
    '''Normalize Histogram'''
    hist_vector=hist_vector/total_location    
    
    return hist_vector
    
    

def seg_neighbor(svMap):
    '''
    Function to find adjacency matrix
    Args:
    ----
    svMap: Super pixel mask. Each pixel location will have the superpixel label 
            corresponding to it.  svMap is an integer image with the value of each pixel being
           the id of the superpixel with which it is associated
    Return:
    ------
    Bmap:  Bmap is a binary adjacency matrix NxN (N being the number of superpixels
            in svMap).  Bmap has a 1 in cell i,j if superpixel i and j are neighbors.
            Otherwise, it has a 0.  Superpixels are neighbors if any of their
            pixels are neighbors.
    '''
    '''
    Implement the code to compute the adjacency matrix for the superpixel graph
    captured by svMap

    '''
    segmentList = np.unique(svMap)
    segmentNum = segmentList.shape[0]
    # FILL IN THE CODE HERE to calculate the adjacency
    Bmap = np.zeros([segmentNum, segmentNum])
    height,width = svMap.shape
    for i in range(height):
        for j in range(width):
            '''check eight connectivity'''
            y_u = min(i+1, height-1) 
            x_u = min(j+1, width-1)
            x_l = max(j-1, 0)
            ''' check lower'''
            if svMap[i,j] != svMap[y_u,j]: 
                Bmap[svMap[i,j],svMap[y_u,j]] = 1
                Bmap[svMap[y_u,j],svMap[i,j]] = 1
            ''' check left'''
            if svMap[i,j] != svMap[i,x_u]:
                Bmap[svMap[i,j],svMap[i,x_u]] = 1
                Bmap[svMap[i,x_u],svMap[i,j]] = 1
            ''' check lower left'''
            if svMap[i,j] != svMap[y_u,x_u]:
                Bmap[svMap[i,j],svMap[y_u,x_u]] = 1
                Bmap[svMap[y_u,x_u],svMap[i,j]] = 1
            ''' check lower right'''
            if svMap[i,j] != svMap[y_u,x_l]:
                Bmap[svMap[i,j],svMap[y_u,x_l]] = 1
                Bmap[svMap[y_u,x_l],svMap[i,j]] = 1

    return Bmap
    
def hist_intersect(a, b):
    return np.sum(np.minimum(a,b))

def graphcut(S,C,hist_values, keyindex):
    '''
    Function to take a superpixel set and a keyindex and convert to a 
    foreground/background segmentation.

    keyindex is the index to the superpixel we wish to use as foreground and
    find its relevant neighbors that would be in the same macro-segment

    Similarity is computed based on segments(i).fv (which is a color histogram)
    and spatial proximity.
    
    Args:
    ----
    S: Super pixel mask. Each pixel location will have the superpixel label 
        corresponding to it.
    C: List of coordinates of centroid of superpixels
    hist_values: reduced feature for the segmentation (histograms)
    keyindex : keyindex is the index to the superpixel we wish to use as foreground
    Return:
    -------
    B: B is a binary image with 1's for those pixels connected to the
      source node and hence in the same segment as the keyindex.
        B has 0's for those nodes connected to the sink.
    '''
    
    #Compute basic adjacency information of superpixels
    adjacency = seg_neighbor(S)
    '''
    Normalization for distance calculation based on the image size.
    For points (x1,y1) and (x2,y2), distance is exp(-||(x1,y1)-(x2,y2)||^2/dnorm)
    Thinking of this like a Gaussian and considering the Std-Dev of the Gaussian 
    to be roughly half of the total number of pixels in the image. Just a guess.
    '''
    dnorm = 2*np.square(np.prod(np.divide(S.shape,2)))
    
    k = len(C)
    # Generate capacity matrix
    capacity = np.zeros((k+2,k+2)) # initialize the zero-valued capacity matrix
    source = k # set the index of the source node
    sink = k+1 # set the index of the sink node
    capacity[k+1,keyindex]=0
    capacity[keyindex,k+1]=0
    capacity[k,keyindex]=3
    capacity[keyindex,k]=3
    '''
    This is a single planar graph with an extra source and sink.

    Capacity of a present edge in the graph (adjacency) is to be defined as the product of
    1:  the histogram similarity between the two color histogram feature vectors.
       use the provided histintersect function below to compute this similarity 
    2:  the spatial proximity between the two superpixels connected by the edge.
       use exp(-D(a,b)/dnorm) where D is the euclidean distance between superpixels a and b,
          dnorm is given above.

     * Source gets connected to every node except sink:
       capacity is with respect to the keyindex superpixel
     * Sink gets connected to every node except source:
       capacity is opposite that of the corresponding source-connection (from each superpixel)
       in our case, the max capacity on an edge is 3; so, 3 minus corresponding capacity
     * Other superpixels get connected to each other based on computed adjacency
         matrix:
      capacity defined as above. EXCEPT THAT YOU ALSO NEED TO MULTIPLY BY A SCALAR 0.25 for
      adjacent superpixels.
    '''
    # FILL IN CODE HERE to generate the capacity matrix using the description above.
    for i in range(k):
        for j in range((i+1),k):
            His_sim_cur = hist_intersect(hist_values[i],hist_values[j])
            dis_cen = np.array(C[i])-np.array(C[j])
            Spa_sim_cur = np.exp(-1*np.linalg.norm(dis_cen, ord=2)/dnorm)
            Capa_cur = His_sim_cur*Spa_sim_cur
            if adjacency[i,j]==1:
                capacity[i,j] = 0.25*Capa_cur
                capacity[j,i] = 0.25*Capa_cur          
            if i==keyindex:
                capacity[k,j] = Capa_cur
                capacity[j,k] = Capa_cur
                capacity[k+1,j] = 3-Capa_cur
                capacity[j,k+1] = 3-Capa_cur
            elif j==keyindex:
                capacity[k,i] = Capa_cur
                capacity[i,k] = Capa_cur
                capacity[k+1,i] = 3-Capa_cur
                capacity[i,k+1] = 3-Capa_cur

    #Compute the cut (this code is provided to you)
    _,current_flow = ff_max_flow(source, sink, capacity, k+2)
    '''
    Extract the two-class segmentation.
    The cut will separate all nodes into those connected to the
    source and those connected to the sink.
    The current_flow matrix contains the necessary information about
    the max-flow through the graph.

     Populate the binary matrix B with 1's for those nodes that are connected
      to the source (and hence are in the same big segment as our keyindex) in the
      residual graph.
 
     You need to compute the set of reachable nodes from the source.  Recall, from
      lecture that these are any nodes that can be reached from any path from the
      source in the graph with residual capacity (original capacity - current flow) 
      being positive.
    '''
    reachable_node = bfs_residual_reachable(source, current_flow, capacity)
    

    '''create mask'''
    B = np.zeros_like(S)
    for i in range(len(reachable_node)):
        cur_idx = (S==reachable_node[i])
        B[cur_idx] = 1
    return B
   

def q1():
    img = etai.read('veggie-stand.jpg')
    #first compute the superpixels on the image we loaded
    S, C = get_superpixel(img,230)
    display_save(img,S,'q1')
    # compute histograms on a superpixel

    v = histvec(img, S==115,10)
   
    # plot and compare
    plt.figure()
    plt.subplot(131)
    plt.bar(np.arange(len(v)),v)

    plt.title("Student_output")
    solution_hist = np.load('q1_histogram.npy')
    plt.subplot(132)
    plt.bar(np.arange(len(solution_hist)),solution_hist)

    plt.title("Solution_output")
    plt.subplot(133)
    plt.plot(np.arange(len(v)),abs(v-solution_hist))
    plt.title("Error")
    #plt.show()
    plt.savefig('q1_result.png')
    plt.close()
    
    
def q2():
    img = etai.read('porch1.png')[:,:,:3]
    #first compute the superpixels on the image we loaded
    S, C = get_superpixel(img,300)
    display_save(img,S,'q2')
    #compute adjacency matrix
    student_A = seg_neighbor(S)

    solution_A = np.load("Adjacency.npy")
    # plot and compare
    plt.figure()
    plt.subplot(131)
    plt.imshow(student_A)
    plt.title("student output")
    plt.subplot(132)
    plt.imshow(solution_A)
    plt.title("solution output")
    plt.subplot(133)
    plt.imshow(student_A-solution_A)
    plt.title("Error")
    #plt.show()
    plt.savefig('q2_result.png')
    plt.close()


def q3():
    img = etai.read('flower1.jpg')
    # first compute the superpixels on the image we loaded
    S, C = get_superpixel(img,180)
    display_save(img,S,'q3')
    # next compute the feature reduction for the segmentation (histograms)
    hist_values = img_reduce(img,S,C)
    print('Please click on the superpixel you want to be the key \n on which to base the foreground extraction.\n\n')
    select_keyindex =True
    if select_keyindex:

        key_coordinates = get_correspondences(img)[0]
        x = int(key_coordinates[0])
        y = int(key_coordinates[1])
        keyindex = S[y,x]
   
    # Perform graph-cut
    output_student = graphcut(S,C,hist_values, keyindex)
    # plot and compare
    solution_output =np.load("solution_q3_mask.npy")

    plt.figure()
    plt.subplot(131)
    plt.imshow(output_student)
    plt.title("student output")
    plt.subplot(132)
    plt.imshow(solution_output)
    plt.title("solution output")
    plt.subplot(133)
    plt.imshow(output_student-solution_output)
    plt.title("Error")
    #plt.show()
    plt.savefig('q3_result.png')
    plt.close()


def example(file_name):
    # Uncomment the images according to the question
    img = etai.read(file_name)[:,:,:3]
    file_name_w_ext = file_name.split('.')[0]
    
    # first compute the superpixels on the image we loaded
    S, C = get_superpixel(img,180)
    display_save(img,S,file_name_w_ext)
    # next compute the feature reduction for the segmentation (histograms)
    hist_values = img_reduce(img,S,C)
    print('Please click on the superpixel you want to be the key \n on which to base the foreground extraction.\n\n')
    select_keyindex =True
    if select_keyindex:

        key_coordinates = get_correspondences(img)[0]
        x = int(key_coordinates[0])
        y = int(key_coordinates[1])
        keyindex = S[y,x]
    
    # Perform graph-cut
    output_student = graphcut(S,C,hist_values, keyindex)
    plt.figure()
    plt.subplot(131)
    img2 = img.copy()
    img2[S!=keyindex] = 0
    plt.imshow(img2)
    plt.title("keyindex")
    plt.subplot(132)
    plt.imshow(output_student)
    plt.title("graphcut mask")
    plt.subplot(133)
    out_img = img.copy()
    out_img[output_student != 1] =255
    plt.imshow(mark_boundaries(out_img,S))
    plt.title("Superpixels and fg")
    #plt.show()
    plt.savefig(file_name_w_ext+'_result.png')
    plt.close()


def main():
    # These are the functions to do Question 1 Part 1-4. 
    q1()
    q2()
    q3()
    namelist = ['flower1.jpg','porch1.png','flag1.jpg']
    for file_name in namelist:
        example(file_name)
    

if(__name__=="__main__"):
    main()
